/* ESP32-S3 (PIE) / ESP32-P4 の SIMD 内積。inline asm は xtensa gcc 13 の -O3 で ICE を起こすので、この TU だけ -O2 で別にコンパイルする。 */
#include <stdint.h>
#if defined(KM_KERNEL_ESP32S3)
int32_t km_dot_s8_pie(const int8_t *pa, const int8_t *pb, int n16) {
    int32_t acc; int zero = 0;
    __asm__ volatile(
        "ee.zero.accx\n"
        "loopnez %[cnt], .Lkm_dot_end%=\n"
        "  ee.vld.128.ip q0, %[pa], 16\n"
        "  ee.vld.128.ip q1, %[pb], 16\n"
        "  ee.vmulas.s8.accx q0, q1\n"
        ".Lkm_dot_end%=:\n"
        "ee.srs.accx %[acc], %[zero], 0\n"
        : [acc] "=r"(acc), [pa] "+r"(pa), [pb] "+r"(pb)
        : [cnt] "r"(n16), [zero] "r"(zero)
        : "memory");
    return acc;
}
/* 候補 2: 積和とロードを融合した命令で 2 段展開 (32 MAC あたり 6 命令)。末尾で 16 byte 先読みするので
 * 両バッファに 16 byte の余裕が要る (重み blob は 64 byte 詰め、作業バッファは +16 以上確保している)。n16 は偶数前提。 */
int32_t km_dot_s8_pie2(const int8_t *pa, const int8_t *pb, int n16) {
    int32_t acc; int zero = 0; int cnt = n16 >> 1;
    if (n16 & 1) {                                    /* 奇数なら先頭 1 ブロックを先に */
        __asm__ volatile(
            "ee.zero.accx\n"
            "ee.vld.128.ip q0, %[pa], 16\n"
            "ee.vld.128.ip q1, %[pb], 16\n"
            "ee.vmulas.s8.accx q0, q1\n"
            : [pa] "+r"(pa), [pb] "+r"(pb) : : "memory");
    } else {
        __asm__ volatile("ee.zero.accx\n" ::: "memory");
    }
    if (cnt == 0) { __asm__ volatile("ee.srs.accx %[acc], %[zero], 0\n" : [acc] "=r"(acc) : [zero] "r"(zero)); return acc; }
    __asm__ volatile(
        "ee.vld.128.ip q0, %[pa], 16\n"
        "ee.vld.128.ip q1, %[pb], 16\n"
        "loopnez %[cnt], .Lkm_dot2_end%=\n"
        "  ee.vmulas.s8.accx.ld.ip q2, %[pa], 16, q0, q1\n"
        "  ee.vld.128.ip q3, %[pb], 16\n"
        "  ee.vmulas.s8.accx.ld.ip q0, %[pa], 16, q2, q3\n"
        "  ee.vld.128.ip q1, %[pb], 16\n"
        ".Lkm_dot2_end%=:\n"
        "ee.srs.accx %[acc], %[zero], 0\n"
        : [acc] "=r"(acc), [pa] "+r"(pa), [pb] "+r"(pb)
        : [cnt] "r"(cnt), [zero] "r"(zero)
        : "memory");
    return acc;
}

int32_t km_dot_s8_pie3(const int8_t *pa, const int8_t *pb) {
    int32_t acc; int zero = 0;
    __asm__ volatile(
        "ee.zero.accx\n"
        "ee.vld.128.ip q0, %[pa], 16\n"
        "ee.vld.128.ip q1, %[pb], 16\n"
        "ee.vmulas.s8.accx.ld.ip q2, %[pa], 16, q0, q1\n"
        "ee.vld.128.ip q3, %[pb], 16\n"
        "ee.vmulas.s8.accx.ld.ip q0, %[pa], 16, q2, q3\n"
        "ee.vld.128.ip q1, %[pb], 16\n"
        "ee.vmulas.s8.accx q0, q1\n"
        "ee.srs.accx %[acc], %[zero], 0\n"
        : [acc] "=r"(acc), [pa] "+r"(pa), [pb] "+r"(pb) : [zero] "r"(zero) : "memory");
    return acc;
}
#elif defined(KM_KERNEL_ESP32P4)
int32_t km_dot_s8_p4(const int8_t *pa, const int8_t *pb, int n16) {
    int32_t acc; int zero = 0;
    /* P4 では 40 bit アキュムレータは XACC (S3 の ACCX)。ゼロ化は下位/上位を別々に書く。 */
    __asm__ volatile(
        "esp.movx.w.xacc.l %[zero]\n"
        "esp.movx.w.xacc.h %[zero]\n"
        "1:\n"
        "  esp.vld.128.ip q0, %[pa], 16\n"
        "  esp.vld.128.ip q1, %[pb], 16\n"
        "  esp.vmulas.s8.xacc q0, q1\n"
        "  addi %[cnt], %[cnt], -1\n"
        "  bnez %[cnt], 1b\n"
        "esp.srs.s.xacc %[acc], %[zero]\n"
        : [acc] "=r"(acc), [pa] "+r"(pa), [pb] "+r"(pb), [cnt] "+r"(n16)
        : [zero] "r"(zero)
        : "memory");
    return acc;
}
int32_t km_dot_s8_p4_2(const int8_t *pa, const int8_t *pb, int n16) {
    int32_t acc; int zero = 0; int cnt = n16 >> 1;
    __asm__ volatile("esp.movx.w.xacc.l %[zero]\nesp.movx.w.xacc.h %[zero]\n" : : [zero] "r"(zero) : "memory");
    if (n16 & 1) {                                    /* 奇数なら先頭 1 ブロックを先に */
        __asm__ volatile(
            "esp.vld.128.ip q0, %[pa], 16\n"
            "esp.vld.128.ip q1, %[pb], 16\n"
            "esp.vmulas.s8.xacc q0, q1\n"
            : [pa] "+r"(pa), [pb] "+r"(pb) : : "memory");
    }
    if (cnt == 0) { __asm__ volatile("esp.srs.s.xacc %[acc], %[zero]\n" : [acc] "=r"(acc) : [zero] "r"(zero)); return acc; }
    __asm__ volatile(
        "esp.vld.128.ip q0, %[pa], 16\n"
        "esp.vld.128.ip q1, %[pb], 16\n"
        "1:\n"
        "  esp.vmulas.s8.xacc.ld.ip q2, %[pa], 16, q0, q1\n"
        "  esp.vld.128.ip q3, %[pb], 16\n"
        "  esp.vmulas.s8.xacc.ld.ip q0, %[pa], 16, q2, q3\n"
        "  esp.vld.128.ip q1, %[pb], 16\n"
        "  addi %[cnt], %[cnt], -1\n"
        "  bnez %[cnt], 1b\n"
        "esp.srs.s.xacc %[acc], %[zero]\n"
        : [acc] "=r"(acc), [pa] "+r"(pa), [pb] "+r"(pb), [cnt] "+r"(cnt)
        : [zero] "r"(zero)
        : "memory");
    return acc;
}

int32_t km_dot_s8_p4_3(const int8_t *pa, const int8_t *pb) {
    int32_t acc; int zero = 0;
    __asm__ volatile(
        "esp.movx.w.xacc.l %[zero]\n"
        "esp.movx.w.xacc.h %[zero]\n"
        "esp.vld.128.ip q0, %[pa], 16\n"
        "esp.vld.128.ip q1, %[pb], 16\n"
        "esp.vmulas.s8.xacc.ld.ip q2, %[pa], 16, q0, q1\n"
        "esp.vld.128.ip q3, %[pb], 16\n"
        "esp.vmulas.s8.xacc.ld.ip q0, %[pa], 16, q2, q3\n"
        "esp.vld.128.ip q1, %[pb], 16\n"
        "esp.vmulas.s8.xacc q0, q1\n"
        "esp.srs.s.xacc %[acc], %[zero]\n"
        : [acc] "=r"(acc), [pa] "+r"(pa), [pb] "+r"(pb) : [zero] "r"(zero) : "memory");
    return acc;
}
#endif
