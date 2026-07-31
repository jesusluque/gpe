# Copyright (c) 2026 gpe contributors.
#
# Kernels are compiled at build time and embedded as binary blobs.
#
# Not loaded from disk at run time: a compositor that reads its kernels from a
# directory beside the executable has a directory beside the executable to ship,
# to sign, and to be broken by. The blob is in the binary and cannot go missing.
#
# The route differs per backend, and neither is Slang's one-shot path:
#
#   Metal   .slang --slangc--> .metal --metal--> .air --metallib--> .metallib
#   CUDA    .slang --slangc--> .cu    --nvcc -ptx--> .ptx
#
# Slang can emit PTX directly, through nvrtc, and Metal libraries directly,
# through the Metal toolchain -- but only when it can dlopen them, which makes
# the build depend on where a shared library happens to be rather than on the
# compilers CMake already found. Going through the source form costs one extra
# step and uses the toolchain the machine actually has.

find_program(GPE_SLANGC slangc
    HINTS $ENV{SLANG_ROOT}/bin /Users/muriel/tools/slang/bin
    DOC "Slang compiler")
if(NOT GPE_SLANGC)
    message(FATAL_ERROR
        "slangc not found. conda: `conda env create -f environment.yml`, "
        "or unpack a release from github.com/shader-slang/slang and set "
        "SLANG_ROOT.")
endif()

if(GPE_BACKEND STREQUAL "Metal")
    # Through xcrun, not by path.
    #
    # The Metal compiler is not in the Command Line Tools and is not in
    # /usr/bin either: it ships as a downloadable toolchain and lives under a
    # versioned cryptex mount whose path changes when Apple updates it. xcrun
    # is the only stable way to name it, and it also fails with a readable
    # message on a machine that has only the Command Line Tools -- which is
    # what this machine had an hour ago.
    find_program(GPE_XCRUN xcrun REQUIRED)
    execute_process(COMMAND ${GPE_XCRUN} -f metal
                    RESULT_VARIABLE metal_found OUTPUT_QUIET ERROR_QUIET)
    if(NOT metal_found EQUAL 0)
        message(FATAL_ERROR
            "the Metal compiler is not available. It needs full Xcode, not the "
            "Command Line Tools: install Xcode, then "
            "`sudo xcodebuild -license accept && "
            "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.")
    endif()
    set(GPE_METAL ${GPE_XCRUN} -sdk macosx metal)
    set(GPE_METALLIB ${GPE_XCRUN} -sdk macosx metallib)
elseif(GPE_BACKEND STREQUAL "CUDA")
    find_program(GPE_NVCC nvcc HINTS ${CUDAToolkit_BIN_DIR} REQUIRED)
endif()

set(GPE_KERNEL_DIR "${GPE_BUILD}/kernels")
file(MAKE_DIRECTORY "${GPE_KERNEL_DIR}")

# Visible to whoever added this project, so a plugin built alongside it can
# compile its own kernels with the same toolchain rather than searching for one
# again -- and, more to the point, so it cannot end up with a different one.
#
# A no-op when gpe is the top-level project, which is why it is guarded: there
# is no parent to tell.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GPE_BACKEND   "${GPE_BACKEND}"   PARENT_SCOPE)
    set(GPE_SLANGC    "${GPE_SLANGC}"    PARENT_SCOPE)
    set(GPE_METAL     "${GPE_METAL}"     PARENT_SCOPE)
    set(GPE_METALLIB  "${GPE_METALLIB}"  PARENT_SCOPE)
    set(GPE_NVCC      "${GPE_NVCC}"      PARENT_SCOPE)
    set(GPE_CUDA_ARCH "${GPE_CUDA_ARCH}" PARENT_SCOPE)
endif()

# gpe_compile_slang(<name> ENTRY <entry>)
#
# Compiles kernels/<name>.slang for this backend and leaves the blob at
# ${GPE_KERNEL_DIR}/<name>.blob. Records it so that gpe_write_kernel_header()
# can put them all in one table.
function(gpe_compile_slang name)
    cmake_parse_arguments(ARG "" "ENTRY" "" ${ARGN})
    if(NOT ARG_ENTRY)
        message(FATAL_ERROR "gpe_compile_slang(${name}) needs ENTRY")
    endif()

    set(source "${GPE_ROOT}/kernels/${name}.slang")
    set(blob "${GPE_KERNEL_DIR}/${name}.blob")

    # Every kernel imports common.slang, so a change to it has to rebuild all of
    # them. Listed as a dependency rather than globbed: a glob is re-run at
    # configure time and not at build time, which is how a stale kernel survives
    # an edit.
    set(deps "${source}" "${GPE_ROOT}/kernels/common.slang")

    if(GPE_BACKEND STREQUAL "Metal")
        set(msl "${GPE_KERNEL_DIR}/${name}.metal")
        set(air "${GPE_KERNEL_DIR}/${name}.air")
        add_custom_command(
            OUTPUT "${blob}"
            COMMAND ${GPE_SLANGC} "${source}" -target metal
                    -entry ${ARG_ENTRY} -stage compute -o "${msl}"
            COMMAND ${GPE_METAL} -c "${msl}" -o "${air}"
            COMMAND ${GPE_METALLIB} "${air}" -o "${blob}"
            DEPENDS ${deps}
            WORKING_DIRECTORY "${GPE_ROOT}/kernels"
            COMMENT "slang -> metallib: ${name}"
            VERBATIM)
    elseif(GPE_BACKEND STREQUAL "CUDA")
        set(cu "${GPE_KERNEL_DIR}/${name}.cu")
        # -ptx, not -cubin: PTX is forward compatible, so a binary built here
        # runs on a card that did not exist when it was built. The driver JITs
        # it once and caches the result.
        add_custom_command(
            OUTPUT "${blob}"
            COMMAND ${GPE_SLANGC} "${source}" -target cuda
                    -entry ${ARG_ENTRY} -stage compute -o "${cu}"
            COMMAND ${GPE_NVCC} -ptx "${cu}" -o "${blob}"
                    -arch=${GPE_CUDA_ARCH}
            DEPENDS ${deps}
            WORKING_DIRECTORY "${GPE_ROOT}/kernels"
            COMMENT "slang -> ptx: ${name}"
            VERBATIM)
    else()
        message(FATAL_ERROR "no backend to compile ${name} for")
    endif()

    set_property(GLOBAL APPEND PROPERTY GPE_KERNEL_NAMES "${name}")
    set_property(GLOBAL APPEND PROPERTY GPE_KERNEL_ENTRIES "${ARG_ENTRY}")
    set_property(GLOBAL APPEND PROPERTY GPE_KERNEL_BLOBS "${blob}")
endfunction()

# gpe_write_kernel_header(<target>)
#
# Generates gpe_kernels.h -- every blob as a byte array plus one table to look
# them up by name -- and hangs its generation off <target>.
function(gpe_write_kernel_header target)
    get_property(names GLOBAL PROPERTY GPE_KERNEL_NAMES)
    get_property(entries GLOBAL PROPERTY GPE_KERNEL_ENTRIES)
    get_property(blobs GLOBAL PROPERTY GPE_KERNEL_BLOBS)
    if(NOT names)
        message(FATAL_ERROR "gpe_write_kernel_header: no kernels compiled")
    endif()

    set(header "${GPE_KERNEL_DIR}/gpe_kernels.h")

    # Written by a script at build time rather than by configure_file here:
    # the blobs do not exist at configure time, and a header generated from
    # files that are not there yet is a header full of nothing.
    set(script "${GPE_KERNEL_DIR}/write_kernels_header.cmake")
    file(WRITE "${script}" "\
# Generated by gpe_write_kernel_header. Do not edit.
set(names \"${names}\")
set(entries \"${entries}\")
set(blobs \"${blobs}\")
set(out \"${header}\")

set(body \"// Generated at build time from kernels/*.slang. Do not edit.\\n\")
string(APPEND body \"#pragma once\\n\\n#include <cstddef>\\n#include <string_view>\\n\\n\")
string(APPEND body \"namespace gpe::kernels {\\n\\n\")

list(LENGTH names count)
math(EXPR last \"\${count} - 1\")
foreach(i RANGE \${last})
    list(GET names \${i} name)
    list(GET blobs \${i} blob)
    file(READ \"\${blob}\" hex HEX)
    string(REGEX REPLACE \"(..)\" \"0x\\\\1,\" bytes \"\${hex}\")
    string(REGEX REPLACE \"(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)\" \"\\\\1\\n    \" bytes \"\${bytes}\")
    string(APPEND body \"inline constexpr unsigned char k_\${name}[] = {\\n    \${bytes}\\n};\\n\\n\")
endforeach()

string(APPEND body \"/// One compiled kernel, embedded.\\nstruct Blob {\\n\")
string(APPEND body \"    std::string_view     name;\\n\")
string(APPEND body \"    std::string_view     entry;\\n\")
string(APPEND body \"    const unsigned char* data;\\n\")
string(APPEND body \"    size_t               size;\\n};\\n\\n\")
string(APPEND body \"inline constexpr Blob kAll[] = {\\n\")
foreach(i RANGE \${last})
    list(GET names \${i} name)
    list(GET entries \${i} entry)
    string(APPEND body \"    {\\\"\${name}\\\", \\\"\${entry}\\\", k_\${name}, sizeof(k_\${name})},\\n\")
endforeach()
string(APPEND body \"};\\n\\n\")
string(APPEND body \"/// The blob called `name`, or null. Linear over a table of a few entries.\\n\")
string(APPEND body \"[[nodiscard]] inline const Blob* find(std::string_view name) {\\n\")
string(APPEND body \"    for (const Blob& blob : kAll) {\\n        if (blob.name == name) {\\n            return &blob;\\n        }\\n    }\\n    return nullptr;\\n}\\n\\n\")
string(APPEND body \"}   // namespace gpe::kernels\\n\")

file(WRITE \"\${out}\" \"\${body}\")
")

    add_custom_command(
        OUTPUT "${header}"
        COMMAND ${CMAKE_COMMAND} -P "${script}"
        DEPENDS ${blobs} "${script}"
        COMMENT "embedding ${names}"
        VERBATIM)

    add_custom_target(gpe_kernels DEPENDS "${header}")
    add_dependencies(${target} gpe_kernels)
    target_include_directories(${target} PRIVATE "${GPE_KERNEL_DIR}")
endfunction()
