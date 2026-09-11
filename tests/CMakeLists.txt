# Tests are plain executables with an exit code. No framework: the whole suite
# is a handful of files, and a framework here would be a dependency to install
# on two platforms before anybody can run anything.
function(gpe_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE gpe)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    # The kernel table is generated at build time and lives with the blobs, so
    # a test that looks kernels up has to be told where it landed.
    if(GPE_KERNEL_DIR)
        target_include_directories(${name} PRIVATE ${GPE_KERNEL_DIR})
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

gpe_test(test_types)
gpe_test(test_kernel_registry)

# The viewer test needs Qt, so it exists only where the viewer does.
if(TARGET gpe_viewer AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_viewer.cpp")
    add_executable(test_viewer test_viewer.cpp)
    target_link_libraries(test_viewer PRIVATE gpe_viewer gpe)
    add_test(NAME test_viewer COMMAND test_viewer)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_drop.cpp")
    gpe_test(test_drop)
endif()

if(NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_kernels)
    add_dependencies(test_kernels gpe_kernels)
    target_include_directories(test_kernels PRIVATE "${GPE_KERNEL_DIR}")
endif()

if(NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_kernel_info)
    add_dependencies(test_kernel_info gpe_kernels)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_pool.cpp")
    gpe_test(test_pool)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_arena.cpp")
    gpe_test(test_arena)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_ring.cpp")
    gpe_test(test_ring)
    find_package(Threads REQUIRED)
    target_link_libraries(test_ring PRIVATE Threads::Threads)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_roundtrip.cpp" AND
   NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_roundtrip)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_realtime.cpp" AND
   NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_realtime)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_player.cpp" AND
   NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_player)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_display.cpp" AND
   NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_display)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_shared.cpp" AND
   NOT GPE_BACKEND STREQUAL "none")
    add_executable(bench_shared bench_shared.cpp)
    target_link_libraries(bench_shared PRIVATE gpe)
    target_include_directories(bench_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()

if(NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_dispatch_order)
    add_dependencies(test_dispatch_order gpe_kernels)
endif()
