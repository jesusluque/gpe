// The kernel blobs are in the binary, and they are what they claim to be.
//
// "The artifact exists" is not enough to check. A zero-byte file exists, and so
// does a Metal library that slangc wrote before the metal compiler failed. So
// this looks inside: the container has the right magic, and the entry point the
// table names is actually in it.
#include <cstdio>
#include <cstring>
#include <string_view>

#include "gpe_kernels.h"

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
    using namespace gpe::kernels;

    const Blob* scale = find("scale");
    check(scale != nullptr, "the scale kernel is in the table");
    if (scale == nullptr) {
        return 1;
    }
    check(find("no-such-kernel") == nullptr, "and an unknown name is not");

    check(scale->entry == "scaleMain", "the entry point is recorded");
    check(scale->size > 0, "the blob is not empty");
    check(scale->data != nullptr, "and it points somewhere");

    const std::string_view bytes(reinterpret_cast<const char*>(scale->data),
                                 scale->size);

#if defined(__APPLE__)
    // A Metal library starts with "MTLB". Checked because slangc writing a
    // .metal file and the metal compiler then failing leaves a build that
    // succeeded and a blob that is a text file.
    check(scale->size >= 4, "big enough to have a header");
    check(bytes.substr(0, 4) == "MTLB", "the blob is a Metal library");
    // Around four kilobytes for a kernel this small. A blob of a few dozen
    // bytes is a container with no function in it, which loads and then finds
    // nothing at dispatch -- much later, and much harder to read.
    check(scale->size > 512, "and it has something in it");
#else
    // PTX is text, and the entry point is declared in it by name. This is the
    // check that catches a kernel compiled with the wrong entry: the file is a
    // valid PTX module and simply does not contain what the table promises.
    check(bytes.find(".version") != std::string_view::npos,
          "the blob is a PTX module");
    check(bytes.find(".visible .entry scaleMain") != std::string_view::npos,
          "and scaleMain is in it");
    check(bytes.find(".target sm_") != std::string_view::npos,
          "compiled for a real architecture");
#endif

    if (failures == 0) {
        std::printf("test_kernels: ok (%.*s, %zu bytes)\n",
                    static_cast<int>(scale->name.size()), scale->name.data(),
                    scale->size);
    }
    return failures == 0 ? 0 : 1;
}
