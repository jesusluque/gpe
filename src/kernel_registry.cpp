// Copyright (c) 2026 gpe contributors.
#include "kernel_registry.h"

#include <string>
#include <vector>

#include "gpe/kernels.h"

namespace gpe {
namespace {

/// A registered kernel. The strings are owned because `Blob` holds views of
/// them and the caller's may be temporaries; the blob is not, because it lives
/// in whatever loaded it and copying every kernel to guard against a caller
/// unmapping one it is still using would double the memory for nothing.
struct Registered {
    std::string   name;
    std::string   entry;
    kernels::Blob blob;
};

/// Pointers, not values, and that is not a style choice: `Blob` holds
/// `string_view`s into `name` and `entry`. A vector of values that reallocated
/// would leave every previously returned Blob pointing at freed strings; a
/// vector of pointers can grow all it likes because what the views point into
/// never moves.
std::vector<Registered*>& registry() {
    static std::vector<Registered*> kAll;
    return kAll;
}

Registered* findRegistered(std::string_view name) {
    for (Registered* entry : registry()) {
        if (entry->name == name) {
            return entry;
        }
    }
    return nullptr;
}

}   // namespace

bool registerKernel(std::string_view name, std::string_view entry,
                    const void* blob, size_t bytes) {
    if (name.empty() || blob == nullptr || bytes == 0) {
        return false;
    }
    Registered* slot = findRegistered(name);
    if (slot == nullptr) {
        // Leaked on purpose: a registered kernel lives for the process, and the
        // views inside its Blob must stay valid for as long as anything might
        // still hold one. There are a handful of these, not thousands.
        slot = new Registered();
        registry().push_back(slot);
    }
    slot->name = std::string(name);
    slot->entry = std::string(entry);
    slot->blob = kernels::Blob{slot->name, slot->entry,
                               static_cast<const unsigned char*>(blob), bytes};
    return true;
}

void unregisterKernel(std::string_view name) {
    std::vector<Registered*>& all = registry();
    for (size_t i = 0; i < all.size(); ++i) {
        if (all[i]->name == name) {
            // The Registered itself is not freed: a device may have already
            // loaded a module from this blob and be holding the view. Emptying
            // the name is enough to make it unfindable, which is what was
            // asked for.
            all[i]->name.clear();
            all.erase(all.begin() + static_cast<long>(i));
            return;
        }
    }
}

namespace kernels {

const Blob* lookup(std::string_view name) {
    // The build's own table first. A plugin must not be able to shadow
    // `display` or `scale` by naming a kernel after one of them.
    if (const Blob* compiled = find(name); compiled != nullptr) {
        return compiled;
    }
    const Registered* registered = findRegistered(name);
    return registered == nullptr ? nullptr : &registered->blob;
}

}   // namespace kernels
}   // namespace gpe
