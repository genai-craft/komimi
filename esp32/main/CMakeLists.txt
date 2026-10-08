idf_component_register(SRCS main.c INCLUDE_DIRS . REQUIRES komimi esp_partition esp_timer spi_flash)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=misleading-indentation)
