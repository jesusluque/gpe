// The vocabulary compiles and its arithmetic is right.
//
// Small, but not nothing: `Image::bytes` is what every allocation size in the
// engine will be computed from, and it is the kind of expression that is wrong
// by a factor of four for a week before anybody notices.
#include <cstdio>
#include <type_traits>

#include "gpe/device.h"
#include "gpe/types.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}   // namespace

int main() {
    using namespace gpe;

    static_assert(kBytesPerPixel == 16, "float32 RGBA is sixteen bytes");
    static_assert(kInvalidBuffer == 0);
    static_assert(kInvalidKernel == 0);

    // A 1920x1080 plate with no padding.
    constexpr Image hd{1, 1920, 1080, 1920};
    static_assert(hd.bytes() == 1920ull * 1080ull * 16ull);

    // Padded rows count from the stride, not the width. Getting this the other
    // way round reads past the end of every padded buffer in the engine.
    constexpr Image padded{1, 1920, 1080, 2048};
    static_assert(padded.bytes() == 2048ull * 1080ull * 16ull);
    static_assert(padded.bytes() > hd.bytes());

    // A default Image is empty and points at nothing, so an uninitialised one
    // asks for a zero-byte allocation rather than a wild one.
    constexpr Image none{};
    static_assert(none.bytes() == 0);
    static_assert(none.buf == kInvalidBuffer);

    constexpr Grid grid{1920, 1080, 1};
    static_assert(grid.z == 1, "a 2D dispatch still has a z of one");

    // The interface is abstract and stays that way: eight virtuals is the
    // contract, and a Device you can instantiate means somebody added state.
    static_assert(!std::is_default_constructible_v<Device>);
    static_assert(std::has_virtual_destructor_v<Device>);

    check(failures == 0, "no runtime checks failed");
    if (failures == 0) {
        std::puts("test_types: ok");
    }
    return failures == 0 ? 0 : 1;
}
