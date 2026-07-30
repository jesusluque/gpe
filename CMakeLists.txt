cmake_minimum_required(VERSION 3.24)
project(gpe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# --- which backend ------------------------------------------------------------
#
# One per platform, decided here and nowhere else. There is no build in which
# both are compiled: a machine has the one it has, and a client that names
# neither does not care which it got.
if(APPLE)
    set(GPE_BACKEND "Metal")
else()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(GPE_BACKEND "CUDA")
    else()
        set(GPE_BACKEND "none")
    endif()
endif()
message(STATUS "gpe backend: ${GPE_BACKEND}")

# --- the library --------------------------------------------------------------
#
# Sources are added by the phase that writes them. Until then this is the
# headers, which is enough to compile a client against and enough to test the
# vocabulary types -- and it means every phase ends with something that builds
# rather than with a target that cannot link until three phases later.
set(GPE_SOURCES)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/pool.cpp")
    list(APPEND GPE_SOURCES src/pool.cpp)
endif()
if(GPE_BACKEND STREQUAL "Metal" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/device_metal.mm")
    enable_language(OBJCXX)
    list(APPEND GPE_SOURCES src/metal/device_metal.mm)
elseif(GPE_BACKEND STREQUAL "CUDA" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/device_cuda.cpp")
    list(APPEND GPE_SOURCES src/cuda/device_cuda.cpp)
endif()

if(GPE_SOURCES)
    add_library(gpe STATIC ${GPE_SOURCES})
else()
    add_library(gpe INTERFACE)
endif()

if(GPE_SOURCES)
    target_include_directories(gpe
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(gpe PUBLIC cxx_std_20)
    target_compile_options(gpe PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>)
    if(GPE_BACKEND STREQUAL "Metal")
        target_link_libraries(gpe PRIVATE
            "-framework Metal" "-framework Foundation" "-framework QuartzCore")
    elseif(GPE_BACKEND STREQUAL "CUDA")
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(gpe PRIVATE CUDA::cudart)
    endif()
else()
    target_include_directories(gpe INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(gpe INTERFACE cxx_std_20)
endif()

# --- tests --------------------------------------------------------------------

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
