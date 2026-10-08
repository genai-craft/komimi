idf_component_register(SRCS app_main.c audio.c lcd.c pm1.c INCLUDE_DIRS . REQUIRES komimi es8311 esp_lcd driver esp_timer esp_partition spi_flash)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=misleading-indentation -Wno-unused-function)
