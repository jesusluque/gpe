// An allocation another process could open, opened here.
//
// The claim is one sentence: what `exportShareable` hands out and
// `importShareable` takes back name the *same device memory*, not a copy of
// it. Proved by writing through one handle and reading through the other,
// both ways, because a copy would pass the first check and fail the second.
//
// In one process on purpose. Two would prove the same thing and would need a
// child, a pipe and a way to hand an NT handle across, none of which belong in
// a test of the allocator -- the driver does not care whether the importer is
// this process or another, and the thing that could be wrong here is the
// mapping.
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "gpe/device.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::fprintf(stderr, "%s: %s\n", ok ? "  ok  " : "FAIL", what.c_str());
    if (!ok) {
        ++failures;
    }
}

/// The handle is the caller's to close, and this test is a caller.
void closeHandle(uint64_t handle) {
#if defined(_WIN32)
    ::CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
    ::close(static_cast<int>(handle));
#endif
}

constexpr size_t kCount = 4096;
constexpr size_t kBytes = kCount * sizeof(uint32_t);

}   // namespace

int main() {
    using namespace gpe;

    std::unique_ptr<Device> device = Device::create();
    if (device == nullptr) {
        std::puts("test_shareable: skipped, no device");
        return 0;
    }
    if (device->backend() != Device::Backend::CUDA) {
        // Metal shares by other means entirely, and the base class says so by
        // answering kInvalidBuffer. That is the contract, and it is what the
        // caller falls back from.
        check(device->allocShareable(kBytes) == kInvalidBuffer,
              "a backend without this says so rather than pretending");
        std::puts("test_shareable: skipped, not CUDA");
        return failures > 0 ? 1 : 0;
    }

    // An ordinary allocation cannot be exported, and says zero rather than
    // failing: the caller asked whether this buffer can cross, and it cannot.
    const BufferId plain = device->alloc(kBytes);
    check(plain != kInvalidBuffer, "an ordinary allocation still works");
    check(device->exportShareable(plain) == 0,
          "and cannot be exported, which is the answer and not an error");
    device->release(plain);

    const BufferId shared = device->allocShareable(kBytes);
    check(shared != kInvalidBuffer, "a shareable allocation is made");
    if (shared == kInvalidBuffer) {
        return 1;
    }

    std::vector<uint32_t> written(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        written[i] = static_cast<uint32_t>(i * 2654435761u);
    }
    device->upload(shared, written.data(), kBytes);
    device->sync();

    const uint64_t handle = device->exportShareable(shared);
    check(handle != 0, "and exports an operating-system handle");
    if (handle == 0) {
        device->release(shared);
        return 1;
    }

    const BufferId opened = device->importShareable(handle, kBytes);
    check(opened != kInvalidBuffer, "which opens again on this device");
    if (opened == kInvalidBuffer) {
        closeHandle(handle);
        device->release(shared);
        return 1;
    }
    check(opened != shared, "as a handle of its own");

    // What went in one end comes out the other.
    std::vector<uint32_t> read(kCount, 0);
    device->download(read.data(), opened, kBytes);
    device->sync();
    check(read == written, "the imported handle reads what was uploaded");

    // And the other way, which is the half a copy would fail: written through
    // the import, read through the original.
    std::vector<uint32_t> again(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        again[i] = static_cast<uint32_t>(kCount - i);
    }
    device->upload(opened, again.data(), kBytes);
    device->sync();
    std::vector<uint32_t> back(kCount, 0);
    device->download(back.data(), shared, kBytes);
    device->sync();
    check(back == again, "and a write through it is the same memory, not a copy");

    // Releasing the import unmaps and leaves the exporter's allocation alone,
    // the way adopting gives up a claim rather than freeing.
    device->release(opened);
    std::vector<uint32_t> after(kCount, 0);
    device->download(after.data(), shared, kBytes);
    device->sync();
    check(after == again, "releasing the import leaves the original standing");

    closeHandle(handle);
    device->release(shared);

    if (failures > 0) {
        std::fprintf(stderr, "test_shareable: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("test_shareable: ok");
    return 0;
}
