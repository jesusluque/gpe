// Copyright (c) 2026 gpe contributors.
//
// Kernels that were not compiled into this library.
//
// The kernels gpe ships with are compiled by its own build and embedded in the
// binary, and `Device::load` finds them by name in a table generated at build
// time. That is the right arrangement for them and no arrangement at all for
// anybody else's: a plugin loaded from a bundle at run time has its kernels in
// the bundle, and the table was written long before the bundle existed.
//
// So a blob can be handed over at run time and answered for by name from then
// on. Deliberately a free function rather than a ninth method on `Device`: the
// device interface is eight methods and stays eight methods. Registration is a
// property of the process, not of a device -- two devices in one process would
// find the same kernels, which is what anybody would expect.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace gpe {

/// Makes `blob` findable as `name` by every device in this process.
///
/// `entry` is the name of the function inside the blob -- Slang's entry point,
/// which is not usually the same as the kernel's name.
///
/// **The blob is not copied.** It must outlive every dispatch that uses it,
/// which for a plugin means the bundle must stay loaded. Copying instead would
/// double the memory for every kernel to protect against a caller unloading a
/// bundle it is still using, and that caller has a worse problem than this.
///
/// Registering a name twice replaces the earlier one, so a bundle reloaded
/// during development does what a person expects rather than keeping the first
/// version forever.
///
/// **A name this library compiled in is refused**, with a line on stderr. It
/// used to be accepted and then never found -- the build's own table is looked
/// up first -- so a plugin that called a kernel `display` ran gpe's display
/// pass over its buffers with its uniforms, and said nothing about it.
///
/// A blob carrying a gpe kernel trailer (cmake/KernelTrailer.cmake) is
/// stripped of it here and the trailer becomes the kernel's `KernelInfo`.
///
/// False if the name or the blob is empty, or the name is taken by the build.
/// Not thread-safe: this belongs where plugins are loaded, which is before
/// there is a second thread.
bool registerKernel(std::string_view name, std::string_view entry,
                    const void* blob, size_t bytes);

/// What a kernel's own reflection says about it, when its blob carried a
/// trailer. Empty for a blob compiled without one.
///
/// Facts a dispatch depends on and a caller could otherwise only guess:
///
///   - `threadGroup`: the kernel's [numthreads]. Both backends launch
///     ceil(grid / threadGroup) whole groups of exactly this shape, which is
///     what D3D and Vulkan do and what a kernel using shared group memory needs.
///   - `elementBytes`: one element of each structured buffer, in declaration
///     order, for the target the blob was compiled for. CUDA's bounds check
///     counts in these.
///   - `uniformBytes`: the ConstantBuffer's size. A dispatch whose uniform
///     block is a different size is refused rather than run.
struct KernelInfo {
    std::array<uint32_t, 3> threadGroup{1, 1, 1};
    std::vector<uint32_t>   elementBytes;
    uint32_t                uniformBytes = 0;
};

[[nodiscard]] std::optional<KernelInfo> kernelInfo(std::string_view name);

/// Splits a blob into its payload and trailer. `payloadBytes` is `bytes` when
/// there is no trailer. Exposed for the host that wants to check a dispatch
/// against a kernel before queueing it, and for tests.
struct ParsedKernelBlob {
    size_t                    payloadBytes = 0;
    std::optional<KernelInfo> info;
};
[[nodiscard]] ParsedKernelBlob parseKernelBlob(const void* blob, size_t bytes,
                                               std::string_view entry);

/// Forgets one. For a bundle that failed to load after registering some of its
/// kernels: leaving a name pointing into memory that is about to be unmapped is
/// worse than having no kernel at all.
void unregisterKernel(std::string_view name);

}   // namespace gpe
