cmake_minimum_required(VERSION 3.24)
project(gpe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Where this project is, as opposed to whichever project is including it.
#
# CMAKE_SOURCE_DIR is the *top* of the build, which is this directory only when
# gpe is built on its own. Consumed as a subproject it is the consumer's, and
# every path built from it points at the wrong tree -- which is what happened
# the first time openFXplayer added gpe, and is the sort of thing only
# integration finds.
set(GPE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(GPE_BUILD "${CMAKE_CURRENT_BINARY_DIR}")

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

# The card the PTX is generated for.
#
# A floor and not a ceiling: PTX built for one virtual architecture runs on that
# one and everything above it, and the driver JITs the difference. Which is
# exactly why a hardcoded sm_89 was the wrong default -- Ada is a high floor,
# and every card below it, an Ampere laptop included, answers
# cuModuleLoadData with CUDA_ERROR_INVALID_PTX. Every kernel in the build
# failed to load, on a machine whose GPU worked perfectly.
#
# So the default is what this machine actually has, asked of the driver, and
# nvidia-smi is the right thing to ask: it ships with the driver, and a machine
# with no driver has nothing to run the PTX on either. An explicit
# -DGPE_CUDA_ARCH still wins, which is what a build that has to *ship* a bundle
# wants -- an embedded blob compiled for the machine that built it is a plugin
# that refuses to load on an older card, and choosing that floor is a decision
# somebody should make on purpose rather than inherit from a build host.
set(GPE_CUDA_ARCH_DEFAULT "sm_89")
if(NOT DEFINED GPE_CUDA_ARCH)
    find_program(GPE_NVIDIA_SMI nvidia-smi)
    if(GPE_NVIDIA_SMI)
        execute_process(
            COMMAND "${GPE_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _caps RESULT_VARIABLE _asked
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        # One card is the ordinary case. With two, the PTX has to clear the
        # lower of them, and the one nvidia-smi lists first is that only by
        # luck -- so the lowest is taken rather than the first.
        if(_asked EQUAL 0 AND _caps)
            string(REPLACE "\n" ";" _caps "${_caps}")
            set(_lowest "")
            foreach(_cap IN LISTS _caps)
                string(STRIP "${_cap}" _cap)
                string(REPLACE "." "" _cap "${_cap}")
                if(_cap MATCHES "^[0-9]+$")
                    if(_lowest STREQUAL "" OR _cap LESS _lowest)
                        set(_lowest "${_cap}")
                    endif()
                endif()
            endforeach()
            if(_lowest)
                set(GPE_CUDA_ARCH_DEFAULT "sm_${_lowest}")
            endif()
        endif()
    endif()

    # ...and then clamped to what this nvcc can build for, which is not the
    # same question.
    #
    # A card newer than the toolkit is the ordinary case for the first months
    # of a generation, and asking for an architecture nvcc has never heard of
    # is `nvcc fatal : Unsupported gpu architecture` -- every kernel in the
    # build failing to compile, which is the same total failure the wrong floor
    # gave and arrives by the opposite road. Clamping down is safe and is the
    # whole point of a floor: PTX for compute_90 runs on an sm_120 card, the
    # driver JITs the difference.
    #
    # Clamping *up* -- a card older than the toolkit's minimum, which newer
    # CUDA versions do drop -- produces PTX that card cannot run. There is
    # nothing better available, so it is done and said out loud rather than
    # left to fail at load time with CUDA_ERROR_INVALID_PTX and no explanation.
    if(CMAKE_CUDA_COMPILER)
        execute_process(
            COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
            OUTPUT_VARIABLE _known RESULT_VARIABLE _asked
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        # `--list-gpu-arch` arrived in CUDA 11.1. An older nvcc answers
        # non-zero and this is skipped, which is where this started.
        if(_asked EQUAL 0 AND _known)
            string(REGEX MATCHALL "compute_([0-9]+)" _known "${_known}")
            set(_supported "")
            foreach(_one IN LISTS _known)
                string(REGEX REPLACE "compute_" "" _one "${_one}")
                list(APPEND _supported "${_one}")
            endforeach()
            list(SORT _supported COMPARE NATURAL)
            if(_supported)
                list(GET _supported 0 _floor)
                list(GET _supported -1 _ceiling)
                string(REGEX REPLACE "^sm_" "" _wanted "${GPE_CUDA_ARCH_DEFAULT}")
                if(_wanted GREATER _ceiling)
                    message(STATUS
                        "CUDA: this card is sm_${_wanted} and this nvcc stops "
                        "at compute_${_ceiling}; building for the latter, "
                        "which the card runs")
                    set(GPE_CUDA_ARCH_DEFAULT "sm_${_ceiling}")
                elseif(_wanted LESS _floor)
                    message(WARNING
                        "CUDA: this card is sm_${_wanted} and this nvcc starts "
                        "at compute_${_floor}; building for the latter, which "
                        "this card cannot run -- the toolkit has dropped it")
                    set(GPE_CUDA_ARCH_DEFAULT "sm_${_floor}")
                endif()
            endif()
        endif()
    endif()
endif()
set(GPE_CUDA_ARCH "${GPE_CUDA_ARCH_DEFAULT}" CACHE STRING
    "virtual architecture for nvcc -ptx; a floor, so the lowest card it must run on")

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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/arena.cpp")
    list(APPEND GPE_SOURCES src/arena.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/player.cpp")
    list(APPEND GPE_SOURCES src/player.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/display.cpp")
    list(APPEND GPE_SOURCES src/display.cpp)
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
    add_library(gpe STATIC
    src/kernel_registry.cpp ${GPE_SOURCES})
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
        # metal-cpp: header-only, Apache 2.0, vendored so a checkout builds
        # without a trip to developer.apple.com.
        target_include_directories(gpe PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/metal-cpp)
        option(GPE_METAL_CPP_IMPLEMENTATION
               "Define metal-cpp's selector tables in gpe (off when another library in the program does)"
               ON)
        if(NOT GPE_METAL_CPP_IMPLEMENTATION)
            target_compile_definitions(gpe PRIVATE GPE_NO_METAL_CPP_IMPLEMENTATION=1)
        endif()
    elseif(GPE_BACKEND STREQUAL "CUDA")
        find_package(CUDAToolkit REQUIRED)
        # The driver API, not the runtime: the kernels are PTX in a blob and
        # loading a module from memory is cuModuleLoadData. The runtime API's
        # <<<>>> launch needs the kernel compiled into this binary by nvcc,
        # which is the thing embedding a blob is instead of.
        target_link_libraries(gpe PRIVATE CUDA::cuda_driver)
        target_include_directories(gpe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
else()
    target_include_directories(gpe INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(gpe INTERFACE cxx_std_20)
endif()

# --- kernels ------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(NOT GPE_BACKEND STREQUAL "none")
    include(CompileSlang)
    gpe_compile_slang(scale ENTRY scaleMain)
    gpe_compile_slang(display ENTRY displayMain)
    gpe_compile_slang(iota ENTRY iotaMain)
    gpe_write_kernel_header(gpe)
else()
    # A table with nothing in it, so the registry still compiles.
    #
    # `kernel_registry.cpp` is in the library whatever the backend is, and it
    # includes the generated header -- so with no backend the build stopped at
    # `gpe_kernels.h: No such file or directory`, on the one configuration
    # whose whole point is that a machine without a GPU can still build a
    # client against this.
    #
    # Written here rather than by the generator because it is not the same
    # header with zero entries: `Blob kAll[]` with no elements is not valid
    # C++, so the empty case has a shape of its own -- `find` that finds
    # nothing, which is the truth when nothing was compiled.
    set(GPE_KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/kernels")
    file(MAKE_DIRECTORY "${GPE_KERNEL_DIR}")
    file(WRITE "${GPE_KERNEL_DIR}/gpe_kernels.h"
"// Generated for a build with no backend. Do not edit.
#pragma once

#include <cstddef>
#include <string_view>

namespace gpe::kernels {

/// One compiled kernel, embedded. None were: this build has no backend to
/// compile them for.
struct Blob {
    std::string_view     name;
    std::string_view     entry;
    const unsigned char* data;
    size_t               size;
};

[[nodiscard]] inline const Blob* find(std::string_view) { return nullptr; }

}   // namespace gpe::kernels
")
    if(GPE_SOURCES)
        target_include_directories(gpe PRIVATE "${GPE_KERNEL_DIR}")
    endif()
endif()

# --- viewer -------------------------------------------------------------------

add_subdirectory(viewer)

# --- tests --------------------------------------------------------------------

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
