#!/bin/bash
# WebAssembly 版をビルド (Emscripten は docker イメージで)。出力: web/dist/komimi.js, komimi.wasm
set -e
cd "$(dirname "$0")/.."
mkdir -p web/dist
docker run --rm -u $(id -u):$(id -g) -v "$PWD":/src -w /src emscripten/emsdk:latest emcc -O3 -std=c11 -D_GNU_SOURCE -DKM_ATT_INT8 -DKM_KERNEL_WASM_SIMD \
  -msimd128 -I csrc csrc/km_model.c csrc/km_feat.c csrc/km_conformer.c csrc/km_stream.c web/komimi_wasm.c \
  -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=64MB -s MODULARIZE=1 -s EXPORT_NAME=createKomimi \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAPU8","UTF8ToString"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free"]' -o web/dist/komimi.js
ls -la web/dist/
