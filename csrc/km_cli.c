/* ホスト CLI: komimi_cli model.kmm a.wav [b.wav ...]  → 1 行ずつ "path<TAB>text"。16 kHz mono PCM16 WAV のみ。
 * stderr に compute_s=<秒> (読み込みを除いた計算時間の合計) と audio_s。 */
#include "km.h"
#include "km_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t *read_file(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)sz + 64); if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    fclose(f); *n = (size_t)sz; return b;
}

/* 最小限の RIFF 解析: fmt の ch/sr/bits を見て data を取り出す。 */
static float *read_wav(const char *p, int *n_out) {
    size_t n; uint8_t *b = read_file(p, &n); if (!b || n < 44 || memcmp(b, "RIFF", 4) || memcmp(b + 8, "WAVE", 4)) { free(b); return NULL; }
    size_t off = 12; int ch = 1, bits = 16; const uint8_t *data = NULL; size_t dlen = 0;
    while (off + 8 <= n) {
        uint32_t len = b[off+4] | (b[off+5] << 8) | (b[off+6] << 16) | ((uint32_t)b[off+7] << 24);
        if (!memcmp(b + off, "fmt ", 4)) { ch = b[off+10] | (b[off+11] << 8); bits = b[off+22] | (b[off+23] << 8); }
        else if (!memcmp(b + off, "data", 4)) { data = b + off + 8; dlen = len; if (off + 8 + dlen > n) dlen = n - off - 8; break; }
        off += 8 + len + (len & 1);
    }
    if (!data || bits != 16) { free(b); return NULL; }
    int frames = (int)(dlen / (2 * ch)); float *x = (float *)malloc(sizeof(float) * (size_t)frames);
    for (int i = 0; i < frames; i++) { int32_t acc = 0; for (int c = 0; c < ch; c++) { int16_t s = (int16_t)(data[(i*ch+c)*2] | (data[(i*ch+c)*2+1] << 8)); acc += s; } x[i] = (float)acc / ch / 32768.f; }
    free(b); *n_out = frames; return x;
}

static void dump_tap(void *ctx, int frame, const float *lg, int vocab) { (void)frame; fwrite(lg, sizeof(float), (size_t)vocab, (FILE *)ctx); }

int main(int argc, char **argv) {
    int stream = 0, chunk = -1, left = -1, piece = 1600;      /* chunk/left の省略時は .kmm ヘッダの値 (無ければ 16/128) */ const char *dump = NULL; float silence_db = 0.f; int dedup = 1;
    while (argc > 1 && argv[1][0] == '-') {                       /* --stream [--chunk C --left L --piece N --dump logits.f32] */
        if (!strcmp(argv[1], "--stream")) stream = 1;
        else if (!strcmp(argv[1], "--chunk") && argc > 2) { chunk = atoi(argv[2]); argv++; argc--; }
        else if (!strcmp(argv[1], "--left") && argc > 2) { left = atoi(argv[2]); argv++; argc--; }
        else if (!strcmp(argv[1], "--piece") && argc > 2) { piece = atoi(argv[2]); argv++; argc--; }
        else if (!strcmp(argv[1], "--dump") && argc > 2) { dump = argv[2]; argv++; argc--; }
        else if (!strcmp(argv[1], "--silence-db") && argc > 2) { silence_db = (float)atof(argv[2]); argv++; argc--; }
        else if (!strcmp(argv[1], "--no-dedup")) dedup = 0;
        else { fprintf(stderr, "unknown option %s\n", argv[1]); return 2; }
        argv++; argc--;
    }
    if (argc < 3) { fprintf(stderr, "usage: %s [--stream [--chunk C --left L] --piece N --dump f --silence-db D --no-dedup] model.kmm a.wav ...  (chunk/left の省略時は .kmm ヘッダの値)\n", argv[0]); return 2; }
    size_t msz; uint8_t *mb = read_file(argv[1], &msz); if (!mb) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    static km_model m; int rc = km_load(&m, mb, msz); if (rc) { fprintf(stderr, "km_load: %d\n", rc); return 1; }
    if (chunk < 0) chunk = m.chunk > 0 ? m.chunk : 16;
    if (left < 0) left = m.left >= 0 ? m.left : 128;
    double comp = 0, audio = 0; int ids[4096]; char text[16384];
    for (int a = 2; a < argc; a++) {
        int n; float *x = read_wav(argv[a], &n); if (!x) { printf("%s\t\n", argv[a]); continue; }
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        int T, k;
        if (stream) {
            km_stream *st = km_stream_new(&m, chunk, left);
            if (!st) { fprintf(stderr, "stream needs fixed normalization in the model\n"); return 1; }
            FILE *df = dump ? fopen(dump, "wb") : NULL;
            if (df) km_stream_set_tap(st, dump_tap, df);
            if (silence_db != 0.f) km_stream_set_silence_gate(st, silence_db);
            km_stream_set_dedup_prefix(st, dedup);
            for (int off = 0; off < n; off += piece) km_stream_feed(st, x + off, n - off < piece ? n - off : piece);
            km_stream_finish(st);
            const int *sid; k = km_stream_tokens(st, &sid); memcpy(ids, sid, sizeof(int) * (size_t)k); T = km_stream_frames(st);
            if (df) fclose(df);
            if (a == 2) fprintf(stderr, "stream workspace %.2f MB\n", km_stream_bytes(st) / 1e6);
            if (silence_db != 0.f) fprintf(stderr, "skipped chunks: %d\n", km_stream_skipped(st));
            km_stream_free(st);
        } else k = km_recognize(&m, x, n, ids, 4096, &T);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        comp += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9; audio += n / (double)m.sr;
        km_detok(&m, ids, k < 0 ? 0 : k, text, sizeof text);
        printf("%s\t%s\n", argv[a], text); fflush(stdout); free(x);
    }
    fprintf(stderr, "compute_s=%.3f audio_s=%.3f rtf=%.3f\n", comp, audio, audio > 0 ? comp / audio : 0);
    return 0;
}
