# csrc/ をそのまま IDF コンポーネントとして使う (ホストと同じソース)
set(KM_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../csrc)
idf_component_register(SRCS ${KM_SRC}/km_model.c ${KM_SRC}/km_feat.c ${KM_SRC}/km_conformer.c ${KM_SRC}/km_stream.c ${KM_SRC}/km_kernels_esp.c
                       INCLUDE_DIRS ${KM_SRC} REQUIRES heap)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffast-math -Wno-unused-parameter -DKM_ATT_INT8)   # 端末は int8 の KV/射影キャッシュ + int8 の注意スコア (KM_ATT_INT8 は KM_KV_INT8 を含む)
if(CONFIG_IDF_TARGET_ESP32S3)
  # エンジン本体と、km_kernels.h を include する側 (main の自己診断) の両方に効かせる
  target_compile_options(${COMPONENT_LIB} PUBLIC -DKM_KERNEL_ESP32S3)
endif()
if(CONFIG_IDF_TARGET_ESP32P4)
  target_compile_options(${COMPONENT_LIB} PUBLIC -DKM_KERNEL_ESP32P4)
endif()
# xtensa gcc 13 は km_stream.c を -O3 で ICE (extract_constrain_insn) を起こすので、このファイルだけ -O2
set_source_files_properties(${KM_SRC}/km_kernels_esp.c PROPERTIES COMPILE_OPTIONS "-O2")
