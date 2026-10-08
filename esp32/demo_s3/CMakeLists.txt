cmake_minimum_required(VERSION 3.16)
set(EXTRA_COMPONENT_DIRS ../components)          # komimi コンポーネント (csrc/ をそのまま)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(komimi_demo_s3)
