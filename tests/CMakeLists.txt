# Tests are plain executables with an exit code. No framework: the whole suite
# is a handful of files, and a framework here would be a dependency to install
# on two platforms before anybody can run anything.
function(gpe_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE gpe)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

gpe_test(test_types)

if(NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_kernels)
    add_dependencies(test_kernels gpe_kernels)
    target_include_directories(test_kernels PRIVATE "${GPE_KERNEL_DIR}")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_pool.cpp")
    gpe_test(test_pool)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_roundtrip.cpp" AND
   NOT GPE_BACKEND STREQUAL "none")
    gpe_test(test_roundtrip)
endif()
