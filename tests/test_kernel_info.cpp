// What a kernel says about itself, and a device that listens.
//
// The trailer is only worth having if every fact in it is the one the kernel
// actually has, and if each backend then does what the fact says: launches
// the kernel's own group shape, counts the kernel's own element size, and
// refuses a dispatch that does not match. `scale` and `iota` differ in every
// one of those facts, which is why both are here.
//
// And the handles a second runtime on the same device needs: the backend's
// buffer object, the same buffer adopted back, and a readback that does not
// wait.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "gpe/args.h"
#include "gpe/device.h"
#include "gpe/kernels.h"
#include "gpe/pool.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct IotaUniforms {
    uint32_t count = 0;
    uint32_t first = 0;
};
static_assert(sizeof(IotaUniforms) == 8);

}   // namespace

int main() {
    using namespace gpe;

    // --- the trailer, without a device ---------------------------------------

    const std::optional<KernelInfo> scale = kernelInfo("scale");
    check(scale.has_value(), "scale carries a trailer");
    if (scale) {
        check(scale->threadGroup == std::array<uint32_t, 3>{16, 16, 1},
              "scale's group is its [numthreads(16, 16, 1)]");
        check(scale->elementBytes == std::vector<uint32_t>{16, 16},
              "scale's two buffers are float4, sixteen bytes an element");
        check(scale->uniformBytes == 16, "scale's uniforms are sixteen bytes");
    }
    const std::optional<KernelInfo> iota = kernelInfo("iota");
    check(iota.has_value(), "iota carries a trailer");
    if (iota) {
        check(iota->threadGroup == std::array<uint32_t, 3>{64, 1, 1},
              "iota's group is its [numthreads(64, 1, 1)]");
        check(iota->elementBytes == std::vector<uint32_t>{4},
              "iota's one buffer is uint, four bytes an element");
        check(iota->uniformBytes == 8, "iota's uniforms are eight bytes");
    }
    check(!kernelInfo("no-such-kernel").has_value(), "an unknown kernel has no info");

    const unsigned char plain[64] = {'n', 'o', 't', 'a', 't', 'r', 'a', 'i', 'l', 'e', 'r'};
    const ParsedKernelBlob none = parseKernelBlob(plain, sizeof(plain), "x");
    check(none.payloadBytes == sizeof(plain) && !none.info,
          "a blob without a trailer is all payload");

    check(!registerKernel("scale", "scaleMain", plain, sizeof(plain)),
          "a kernel may not be registered under a name gpe compiled in");

    // --- a device ------------------------------------------------------------

    std::unique_ptr<Device> native = Device::create();
    if (native == nullptr) {
        std::puts("test_kernel_info: device half skipped, no device");
        return failures == 0 ? 0 : 1;
    }
    PooledDevice device(std::move(native), size_t{256} << 20);

    const KernelId kernel = device.load("iota");
    check(kernel != kInvalidKernel, "iota loads, its trailer stripped");

    // 1000 is not a multiple of 64: whole groups run past it, and the kernel's
    // bounds check is what keeps the last 24 slots of the buffer untouched.
    constexpr uint32_t kCount = 1000;
    constexpr uint32_t kSlots = 1024;
    const BufferId values = device.alloc(kSlots * sizeof(uint32_t));
    std::vector<uint32_t> sentinel(kSlots, 0xDEADBEEF);
    device.upload(values, sentinel.data(), sentinel.size() * sizeof(uint32_t));

    Args args;
    args.buffer(values, sizeof(uint32_t)).uniforms(IotaUniforms{kCount, 7});
    device.dispatch(kernel, Grid{kCount, 1, 1}, args.data(), args.size());
    device.sync();

    std::vector<uint32_t> out(kSlots, 0);
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    bool ran = true;
    for (uint32_t i = 0; i < kCount; ++i) {
        ran = ran && out[i] == 7 + i;
    }
    check(ran, "iota wrote first + i over the whole grid");
    bool clipped = true;
    for (uint32_t i = kCount; i < kSlots; ++i) {
        clipped = clipped && out[i] == 0xDEADBEEF;
    }
    check(clipped, "and nothing past it");

    // One byte per element -- what a host relaying a plugin's buffers passes --
    // must not truncate the kernel's writes: reflection knows the element.
    Args oneByte;
    oneByte.buffer(values, 1).uniforms(IotaUniforms{kCount, 100});
    device.dispatch(kernel, Grid{kCount, 1, 1}, oneByte.data(), oneByte.size());
    device.sync();
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[kCount - 1] == 100 + kCount - 1,
          "a one-byte element size from the caller does not truncate the write");

    // The wrong number of buffers is refused, not run.
    Args wrong;
    wrong.buffer(values, sizeof(uint32_t)).buffer(values, sizeof(uint32_t))
        .uniforms(IotaUniforms{kCount, 5000});
    device.dispatch(kernel, Grid{kCount, 1, 1}, wrong.data(), wrong.size());
    device.sync();
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[0] == 100, "a dispatch with too many buffers is refused");

    // And the wrong size of uniform block.
    Args shortUniforms;
    const uint32_t justCount = kCount;
    shortUniforms.buffer(values, sizeof(uint32_t)).uniforms(justCount);
    device.dispatch(kernel, Grid{kCount, 1, 1}, shortUniforms.data(), shortUniforms.size());
    device.sync();
    device.download(out.data(), values, out.size() * sizeof(uint32_t));
    check(out[0] == 100, "a dispatch with a short uniform block is refused");

    // --- handles for a second runtime -----------------------------------------

    check(device.backendDevice() != 0, "the backend device is reachable");
    check(device.backendQueue() != 0, "and its queue");
    const uint64_t object = device.backendBuffer(values);
    check(object != 0, "a buffer's backend object is reachable");
    const BufferId again = device.adopt(object, kSlots * sizeof(uint32_t));
    check(again != kInvalidBuffer, "and can be adopted back");
    if (again != kInvalidBuffer) {
        std::vector<uint32_t> through(kSlots, 0);
        device.download(through.data(), again, through.size() * sizeof(uint32_t));
        check(through == out, "the adopted handle reads the same memory");
        device.release(again);
    }

    std::vector<uint32_t> later(kSlots, 0);
    std::atomic<int> arrived{0};
    device.downloadAsync(values, later.data(), later.size() * sizeof(uint32_t),
                         [&arrived](bool fine) { arrived.store(fine ? 1 : -1); });
    for (int i = 0; i < 2000 && arrived.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(arrived.load() == 1, "an asynchronous download completes");
    check(later == out, "with the same bytes");

    device.release(values);
    device.sync();

    if (failures == 0) {
        std::puts("test_kernel_info: ok");
    }
    return failures == 0 ? 0 : 1;
}
