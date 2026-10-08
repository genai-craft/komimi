CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE
LDFLAGS ?= -lm
SRC = km_model.c km_feat.c km_conformer.c km_stream.c
all: komimi_cli komimi_cli_avx2 komimi_cli_att8
komimi_cli: $(SRC) km_cli.c km.h km_kernels.h km_feat.h km_stream.h km_internal.h
	$(CC) $(CFLAGS) -o $@ $(SRC) km_cli.c $(LDFLAGS)
komimi_cli_avx2: $(SRC) km_cli.c km.h km_kernels.h km_feat.h km_stream.h km_internal.h
	$(CC) $(CFLAGS) -mavx2 -DKM_KERNEL_AVX2 -o $@ $(SRC) km_cli.c $(LDFLAGS)
clean:
	rm -f komimi_cli komimi_cli_avx2 komimi_cli_kv8 komimi_cli_att8
komimi_cli_kv8: $(SRC) km_cli.c km.h km_kernels.h km_feat.h km_stream.h km_internal.h
	$(CC) $(CFLAGS) -mavx2 -DKM_KERNEL_AVX2 -DKM_KV_INT8 -o $@ $(SRC) km_cli.c $(LDFLAGS)
komimi_cli_att8: $(SRC) km_cli.c km.h km_kernels.h km_feat.h km_stream.h km_internal.h
	$(CC) $(CFLAGS) -mavx2 -DKM_KERNEL_AVX2 -DKM_ATT_INT8 -o $@ $(SRC) km_cli.c $(LDFLAGS)
