// A kernel that was not compiled into this library can still be found.
//
// The kernels gpe ships with are embedded by its own build and looked up in a
// table generated at build time. A plugin loaded from a bundle at run time has
// its kernels in the bundle, and that table was written long before the bundle
// existed. Without a way to hand one over, an effect SDK cannot exist at all.
//
// What is deliberately *not* here: a ninth method on Device. The device
// interface is eight methods. Registration is a property of the process, not of
// a device -- two devices in one process find the same kernels, which is what
// anybody would expect and what a per-device call would get wrong.
#include <cstdio>
#include <cstring>
#include <string>

#include "gpe/kernels.h"
#include "kernel_registry.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

/// Stands in for a plugin's blob. Its contents do not matter here -- what is
/// under test is whether the name resolves to these bytes, not whether a
/// device can run them.
const unsigned char kPretend[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11};
const unsigned char kOther[] = {0x01, 0x02, 0x03};

}   // namespace

int main() {
    using namespace gpe;

    check(kernels::lookup("no.such.kernel") == nullptr,
          "an unknown name resolves to nothing");

    // The build's own kernels are still found, and by the same call. A second
    // lookup path that the shipped kernels did not go through would be a path
    // nothing tests.
    check(kernels::lookup("scale") != nullptr,
          "a kernel compiled into the library is found");

    check(registerKernel("plugin.blur", "blurMain", kPretend, sizeof(kPretend)),
          "a blob can be registered at run time");

    const kernels::Blob* found = kernels::lookup("plugin.blur");
    check(found != nullptr, "and found by name afterwards");
    if (found != nullptr) {
        check(found->name == "plugin.blur", "under the name it was given");
        check(found->entry == "blurMain",
              "carrying its entry point, which is not the same as its name");
        check(found->data == kPretend, "pointing at the caller's bytes");
        check(found->size == sizeof(kPretend), "with the right length");
    }

    // Registering the same name again replaces it. A bundle rebuilt and
    // reloaded during development should do what a person expects rather than
    // keep the first version for the life of the process.
    check(registerKernel("plugin.blur", "other", kOther, sizeof(kOther)),
          "registering the same name again is allowed");
    found = kernels::lookup("plugin.blur");
    check(found != nullptr && found->data == kOther && found->size == sizeof(kOther),
          "and replaces what was there");

    // A plugin must not be able to shadow a kernel gpe ships with. The build's
    // table is consulted first, so naming a kernel `scale` gets you nowhere.
    check(registerKernel("scale", "notScaleMain", kOther, sizeof(kOther)),
          "a name that collides with a built-in one registers");
    const kernels::Blob* scale = kernels::lookup("scale");
    check(scale != nullptr && scale->data != kOther,
          "but does not shadow the kernel this library compiled in");

    // Nonsense is refused rather than stored. A null blob found later would be
    // a null dereference inside a driver.
    check(!registerKernel("", "e", kPretend, sizeof(kPretend)),
          "an empty name is refused");
    check(!registerKernel("x", "e", nullptr, 4), "a null blob is refused");
    check(!registerKernel("x", "e", kPretend, 0), "and so is an empty one");
    check(kernels::lookup("x") == nullptr, "none of which registered anything");

    // Withdrawing one. For a bundle that failed to load after registering some
    // of its kernels: a name pointing into memory about to be unmapped is worse
    // than no kernel at all.
    unregisterKernel("plugin.blur");
    check(kernels::lookup("plugin.blur") == nullptr, "a kernel can be withdrawn");
    unregisterKernel("plugin.blur");
    check(kernels::lookup("plugin.blur") == nullptr,
          "and withdrawing it twice is not a crash");

    if (failures == 0) {
        std::puts("test_kernel_registry: ok");
    }
    return failures == 0 ? 0 : 1;
}
