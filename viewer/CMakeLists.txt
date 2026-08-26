# The viewer: the only part of the tree that knows Qt exists.
#
# Optional. A machine without Qt builds the engine and every test that does not
# need a window, which is all of them but this one -- the CUDA build machine has
# no Qt and no reason to grow one.
find_package(Qt6 6.6 COMPONENTS Gui GuiPrivate QUIET)
if(NOT Qt6_FOUND)
    message(STATUS "gpe viewer: Qt6 not found, skipping")
    return()
endif()
message(STATUS "gpe viewer: Qt ${Qt6_VERSION}")

# GuiPrivate, not Gui. QRhi is under the versioned private include path and its
# own header says the API has limited compatibility guarantees -- so the Qt
# version is pinned here rather than discovered, and moving it is a decision
# somebody makes on purpose.
add_library(gpe_viewer STATIC rhi_presenter.cpp)
# GPE_ROOT, never CMAKE_SOURCE_DIR: consumed as a subproject the latter is the
# *consumer's* top and this pointed at their tree. It is the same mistake the
# root CMakeLists spends eight lines warning about, made in the one directory
# that was written after it.
target_include_directories(gpe_viewer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
           ${GPE_ROOT}/include)
target_link_libraries(gpe_viewer PUBLIC Qt6::Gui Qt6::GuiPrivate)
target_compile_features(gpe_viewer PUBLIC cxx_std_20)
set(GPE_VIEWER_QT_VERSION "${Qt6_VERSION}" CACHE INTERNAL "")
