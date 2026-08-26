# gpe

A GPU compute engine for video: allocate on the device, run kernels on the
device, and send only the picture that is actually looked at back across the
bus.

One interface — `gpe::Device`, eight virtual calls and a destructor — with a
CUDA backend on Linux and Windows and a Metal backend on macOS. Client code
names no backend and includes no backend header.

```cpp
#include "gpe/device.h"

auto device = gpe::Device::create();      // null if there is no GPU to drive
const gpe::BufferId plate = device->alloc(width * height * 16);
device->upload(plate, pixels, bytes);

const gpe::KernelId scale = device->load("scale");
device->dispatch(scale, grid, &args, sizeof(args));
device->sync();                            // once a frame, not once a dispatch
```

## What is in it

| | |
|---|---|
| `include/gpe/device.h` | The whole engine boundary: allocate, upload, download, fill, load, dispatch, sync, and the device pointer for somebody else's runtime |
| `include/gpe/presenter.h` | Where a finished frame goes, and the seam that keeps Qt out of the engine |
| `include/gpe/kernels.h` | Registering a kernel blob that was not compiled into this library — what a plugin loaded at run time needs |
| `include/gpe/adopt.h` | Driving a `MTLDevice` somebody else already made, so compute and render share one device and there is nothing to synchronise across |
| `src/pool.cpp`, `src/arena.cpp` | The allocator. `cudaMalloc` synchronises the device implicitly and costs about 100 µs, so an allocation per operation is an allocation per operation too many |
| `src/player.cpp` | The playback loop: the clock is the truth, frames are dropped, the timeline never slips |
| `src/display.cpp` | The display pass — transform, downsample and pack on the device, so what crosses to the CPU is the widget-sized 8-bit picture rather than a full-size float plate |

Everything in `src/` is internal today: only `include/gpe/` is on the public
include path, and `Device::create()` hands back the bare backend. So the pool,
the arenas, the player and the display pass are gpe's own and not yet a
consumer's — `device.h` describes the device *as the pool wraps it*, which is
true inside this library and not true of what `create()` returns. Making that
surface public, or returning a pooled device from `create()`, is a decision
that has not been taken.
| `kernels/*.slang` | Written once in Slang, compiled at build time to PTX or to a Metal library |
| `viewer/` | The QRhi presenter, on the far side of the `Presenter` interface |

## Building

Needs CMake 3.24, a C++20 compiler, and `slangc` on the path to compile the
kernels. The pinned environment brings all three:

```sh
conda env create -f environment.yml && conda activate gpe
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build
```

The backend is decided by the machine and nowhere else: Metal on Apple, CUDA
where `nvcc` is found, and `none` otherwise — in which case `gpe` is still a
target and still compiles clients, it just has no device to give them.
`GPE_CUDA_ARCH` (default `sm_89`, an Ada card) is a floor rather than a
ceiling, because PTX is forward compatible.

`environment.yml` pins the build toolchain — conda-forge only — so two machines
building the same commit use the same CMake and the same Ninja.

## Using it from another project

```cmake
add_subdirectory(../gpe ${CMAKE_BINARY_DIR}/gpe EXCLUDE_FROM_ALL)
target_link_libraries(your_target PRIVATE gpe)
```

It is written to be consumed this way: nothing in the build reads
`CMAKE_SOURCE_DIR`, which is the top of *the consumer's* build and points at
the wrong tree in every path derived from it.

## Design notes

The headers carry the reasoning, and they are the documentation — each one says
what it does and why it is shaped that way before it says how. Four of the
decisions worth knowing about up front:

**Null, not an exception, when there is no GPU.** What a host does about a
missing device — fall back to the CPU, tell the user, refuse to start — is the
host's decision, and none of those want to be written as a `catch`.

**The pool is not an optimisation to add later.** Every allocation goes through
it, `alloc` does not usually reach the driver and `release` does not free.

**`devicePointer()` and `stream()` exist for exactly one kind of caller**: a
third-party runtime that compiles and launches its own work — an inference
engine — which cannot go through `dispatch` because what it holds is not a
kernel this library loaded. Enqueuing on *this* stream is what makes the
ordering free. A caller that takes them and then uses a stream of its own has
to synchronise by hand, and will get it wrong.

**Presenting is a readback today.** The interface takes a frame rather than a
texture precisely so that interop — adopting QRhi's `MTLDevice` on macOS, D3D11
sharing on Windows — can arrive behind it later, be measured, and be kept only
if it wins.
