# Copyright (c) 2026 gpe contributors.
#
# What a kernel blob says about itself.
#
# WHY THIS EXISTS
#
# A blob used to be bytes and a name. Everything else a dispatch depends on --
# how many threads a group holds, how big one element of each buffer is, how
# many bytes the uniform block has -- lived in the .slang file and nowhere the
# backend could read it. So each backend assumed: Metal launched 16x16 groups
# whatever the kernel declared, CUDA picked a block shape from the grid, the
# host told CUDA every element was one byte so its bounds check could not
# truncate, and a dispatch handed the wrong number of buffers rendered
# something rather than failing (openFXplayer D30). Every one of those is a
# guess standing in for a fact slangc already knows.
#
# So slangc is asked (-reflection-json), and the answer is appended to the
# embedded blob as a trailer the registry strips before any backend sees the
# bytes: a Metal library and a PTX module are both formats that would reject
# anything after their end.
#
# THE LAYOUT, all uint32 little-endian, read from the end:
#
#     for each entry:  nameLength, name (padded to 4), groupX, groupY, groupZ
#     bufferCount, elementBytes[bufferCount], uniformBytes
#     entryCount, trailerBytes, version (1), 'G' 'P' 'E' 'K'
#
# A kernel whose parameters are not structured buffers plus at most one
# ConstantBuffer gets no trailer, and runs exactly as a blob without one
# always has. That is also every blob compiled before this existed.

# gpe_kernel_trailer_hex(<json file> "<entry;entry>" <out var>)
#
# Script-mode safe: used from the header generators at build time.
function(_gpe_u32_hex value out)
    math(EXPR hex "${value}" OUTPUT_FORMAT HEXADECIMAL)
    string(SUBSTRING "${hex}" 2 -1 hex)
    string(LENGTH "${hex}" len)
    while(len LESS 8)
        string(PREPEND hex "0")
        math(EXPR len "${len} + 1")
    endwhile()
    string(SUBSTRING "${hex}" 6 2 b0)
    string(SUBSTRING "${hex}" 4 2 b1)
    string(SUBSTRING "${hex}" 2 2 b2)
    string(SUBSTRING "${hex}" 0 2 b3)
    set(${out} "${b0}${b1}${b2}${b3}" PARENT_SCOPE)
endfunction()

function(gpe_kernel_trailer_hex json_file entries out)
    set(${out} "" PARENT_SCOPE)
    if(NOT EXISTS "${json_file}")
        return()
    endif()
    file(READ "${json_file}" json)

    set(buffers "")
    set(buffer_count 0)
    set(uniform_bytes 0)
    set(constant_buffers 0)
    string(JSON param_count ERROR_VARIABLE failed LENGTH "${json}" parameters)
    if(failed)
        return()
    endif()
    if(param_count GREATER 0)
        math(EXPR last "${param_count} - 1")
        foreach(i RANGE ${last})
            string(JSON kind GET "${json}" parameters ${i} type kind)
            if(kind STREQUAL "resource")
                string(JSON shape ERROR_VARIABLE failed GET "${json}" parameters ${i} type baseShape)
                if(failed OR NOT shape STREQUAL "structuredBuffer")
                    return()   # a texture or a sampler: not a shape this describes
                endif()
                string(JSON element ERROR_VARIABLE failed
                       GET "${json}" parameters ${i} type resultType sizes 0 value)
                if(failed)
                    return()
                endif()
                _gpe_u32_hex(${element} h)
                string(APPEND buffers "${h}")
                math(EXPR buffer_count "${buffer_count} + 1")
            elseif(kind STREQUAL "constantBuffer")
                string(JSON bytes ERROR_VARIABLE failed
                       GET "${json}" parameters ${i} type elementType sizes 0 value)
                if(failed)
                    return()
                endif()
                set(uniform_bytes ${bytes})
                math(EXPR constant_buffers "${constant_buffers} + 1")
            else()
                return()   # a loose uniform: Slang wraps those, and not predictably
            endif()
        endforeach()
    endif()
    if(constant_buffers GREATER 1)
        return()
    endif()

    set(body "")
    set(entry_count 0)
    string(JSON ep_count ERROR_VARIABLE failed LENGTH "${json}" entryPoints)
    if(failed)
        return()
    endif()
    foreach(entry IN LISTS entries)
        set(found FALSE)
        if(ep_count GREATER 0)
            math(EXPR ep_last "${ep_count} - 1")
            foreach(e RANGE ${ep_last})
                string(JSON name GET "${json}" entryPoints ${e} name)
                if(name STREQUAL entry)
                    string(JSON gx GET "${json}" entryPoints ${e} threadGroupSize 0)
                    string(JSON gy GET "${json}" entryPoints ${e} threadGroupSize 1)
                    string(JSON gz GET "${json}" entryPoints ${e} threadGroupSize 2)
                    set(found TRUE)
                endif()
            endforeach()
        endif()
        if(NOT found)
            return()
        endif()
        string(LENGTH "${entry}" name_len)
        _gpe_u32_hex(${name_len} h)
        string(APPEND body "${h}")
        string(HEX "${entry}" name_hex)
        math(EXPR pad "(4 - ${name_len} % 4) % 4")
        while(pad GREATER 0)
            string(APPEND name_hex "00")
            math(EXPR pad "${pad} - 1")
        endwhile()
        string(APPEND body "${name_hex}")
        foreach(v IN ITEMS ${gx} ${gy} ${gz})
            _gpe_u32_hex(${v} h)
            string(APPEND body "${h}")
        endforeach()
        math(EXPR entry_count "${entry_count} + 1")
    endforeach()

    _gpe_u32_hex(${buffer_count} h)
    string(APPEND body "${h}${buffers}")
    _gpe_u32_hex(${uniform_bytes} h)
    string(APPEND body "${h}")
    _gpe_u32_hex(${entry_count} h)
    string(APPEND body "${h}")

    # trailerBytes counts everything, the footer included. entryCount is
    # already in the body, so the rest of the footer is three words.
    string(LENGTH "${body}" body_hex_len)
    math(EXPR total "${body_hex_len} / 2 + 12")
    _gpe_u32_hex(${total} h)
    string(APPEND body "${h}")
    _gpe_u32_hex(1 h)
    string(APPEND body "${h}")
    string(HEX "GPEK" magic)
    string(APPEND body "${magic}")
    set(${out} "${body}" PARENT_SCOPE)
endfunction()
