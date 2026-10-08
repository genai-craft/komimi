/* ブラウザ (WebAssembly) 用の薄い API。csrc/ のエンジンをそのまま使う。端末と同じ int8 演算 (KM_ATT_INT8) なので結果は端末と一致する。 */
#include <stdlib.h>
#include <string.h>
#include <emscripten/emscripten.h>
#include "km.h"
#include "km_stream.h"

typedef struct { km_model m; uint8_t *buf; km_stream *s; char text[8192]; int chunk, left; } ctx_t;

EMSCRIPTEN_KEEPALIVE void *kmw_load(uint8_t *buf, int n) {
    ctx_t *c = (ctx_t *)calloc(1, sizeof *c);
    c->buf = (uint8_t *)malloc((size_t)n + 64); memcpy(c->buf, buf, (size_t)n);
    if (km_load(&c->m, c->buf, (size_t)n) != 0) { free(c->buf); free(c); return NULL; }
    return c;
}
EMSCRIPTEN_KEEPALIVE int kmw_layers(void *p) { return ((ctx_t *)p)->m.n_layers; }
EMSCRIPTEN_KEEPALIVE int kmw_sub_ch(void *p) { return ((ctx_t *)p)->m.sub_ch; }
EMSCRIPTEN_KEEPALIVE int kmw_default_chunk(void *p) { return ((ctx_t *)p)->m.chunk; }
EMSCRIPTEN_KEEPALIVE int kmw_stream_new(void *p, int chunk, int left, float gate_db) {
    ctx_t *c = (ctx_t *)p;
    if (c->s) km_stream_free(c->s);
    c->chunk = chunk; c->left = left;
    c->s = km_stream_new(&c->m, chunk, left);
    if (!c->s) return -1;
    if (gate_db != 0.f) km_stream_set_silence_gate(c->s, gate_db);
    c->text[0] = 0;
    return 0;
}
EMSCRIPTEN_KEEPALIVE int kmw_feed(void *p, float *pcm, int n) { ctx_t *c = (ctx_t *)p; return c->s ? km_stream_feed(c->s, pcm, n) : -1; }
EMSCRIPTEN_KEEPALIVE void kmw_finish(void *p) { ctx_t *c = (ctx_t *)p; if (c->s) km_stream_finish(c->s); }
EMSCRIPTEN_KEEPALIVE void kmw_reset(void *p) { ctx_t *c = (ctx_t *)p; if (c->s) km_stream_reset(c->s); c->text[0] = 0; }
EMSCRIPTEN_KEEPALIVE const char *kmw_text(void *p) {
    ctx_t *c = (ctx_t *)p; const int *ids; int n = c->s ? km_stream_tokens(c->s, &ids) : 0;
    km_detok(&c->m, ids, n, c->text, sizeof c->text); return c->text;
}
EMSCRIPTEN_KEEPALIVE int kmw_frames(void *p) { ctx_t *c = (ctx_t *)p; return c->s ? km_stream_frames(c->s) : 0; }
EMSCRIPTEN_KEEPALIVE int kmw_skipped(void *p) { ctx_t *c = (ctx_t *)p; return c->s ? km_stream_skipped(c->s) : 0; }
EMSCRIPTEN_KEEPALIVE void kmw_set_dedup(void *p, int on) { ctx_t *c = (ctx_t *)p; if (c->s) km_stream_set_dedup_prefix(c->s, on); }
