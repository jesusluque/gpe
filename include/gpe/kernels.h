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

#include <cstddef>
#include <string_view>

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
/// False if the name or the blob is empty. Not thread-safe: this belongs where
/// plugins are loaded, which is before there is a second thread.
bool registerKernel(std::string_view name, std::string_view entry,
                    const void* blob, size_t bytes);

/// Forgets one. For a bundle that failed to load after registering some of its
/// kernels: leaving a name pointing into memory that is about to be unmapped is
/// worse than having no kernel at all.
void unregisterKernel(std::string_view name);

}   // namespace gpe
