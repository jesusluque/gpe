// Copyright (c) 2026 gpe contributors.
//
// Sharing a device with whoever is drawing.
//
// `Device::create()` finds a GPU and drives it. That is the right answer when
// gpe owns the machine, and the wrong one when something else already does: a
// renderer with its own device means two devices on one GPU, and everything
// that crosses between them is an interop problem -- a shared handle, a fence,
// a synchronisation neither side can express in the other's vocabulary.
//
// The way out is not to solve it. On macOS, gpe takes the device the renderer
// already made. Compute and render then sit on the same MTLDevice, a buffer
// written by one is a buffer read by the other, and there is nothing to
// synchronise across because there is no across.
//
// This is deliberately not part of the eight, and not part of `create`. It is a
// start-up decision made by whoever owns the window, once, and everything after
// it is the same interface as always.
#pragma once

#include <memory>

#include "gpe/device.h"

namespace gpe {

/// Drives `mtlDevice` -- an `id<MTLDevice>` -- instead of finding one.
///
/// The caller keeps ownership: it made the device and something else is
/// drawing with it. Null on a build without the Metal backend, or if the
/// pointer is not a device.
///
/// Deliberately `void*`. The declaration would otherwise drag Metal's headers
/// into every translation unit that includes this, and the whole point of the
/// interface is that a client names no backend.
[[nodiscard]] std::unique_ptr<Device> adoptMetalDevice(void* mtlDevice);

}   // namespace gpe
