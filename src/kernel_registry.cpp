// Copyright (c) 2026 gpe contributors.
#include "kernel_registry.h"

#include <cstdio>
#include <cstring>
#include <mutex>
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
    std::string        name;
    std::string        entry;
    kernels::Blob      raw;
    kernels::Resolved  resolved;
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

/// The build's own kernels, resolved once each. Same leak-on-purpose rule.
std::vector<Registered*>& compiledResolved() {
    static std::vector<Registered*> kAll;
    return kAll;
}

/// Guards the two tables above. `registerKernel` is documented single-threaded,
/// but `resolve` is reached from `load`, which a host may call from a render
/// thread while a plugin loads on another.
std::mutex& tableGuard() {
    static std::mutex kGuard;
    return kGuard;
}

Registered* findRegistered(std::string_view name) {
    for (Registered* entry : registry()) {
        if (entry->name == name) {
            return entry;
        }
    }
    return nullptr;
}

uint32_t readU32(const unsigned char* at) {
    uint32_t value = 0;
    std::memcpy(&value, at, sizeof(value));   // little-endian on every target
    return value;
}

}   // namespace

ParsedKernelBlob parseKernelBlob(const void* blob, size_t bytes, std::string_view entry) {
    ParsedKernelBlob parsed;
    parsed.payloadBytes = bytes;
    if (blob == nullptr || bytes < 16) {
        return parsed;
    }
    const auto* data = static_cast<const unsigned char*>(blob);
    const unsigned char* footer = data + bytes - 16;
    if (std::memcmp(footer + 12, "GPEK", 4) != 0 || readU32(footer + 8) != 1) {
        return parsed;
    }
    const uint32_t trailerBytes = readU32(footer + 4);
    const uint32_t entryCount = readU32(footer);
    if (trailerBytes < 16 || trailerBytes > bytes) {
        return parsed;
    }
    // Everything below is bounds-checked against the trailer's own extent: a
    // blob that claims a trailer and does not hold one is treated as having
    // none, never read past.
    const unsigned char* at = data + bytes - trailerBytes;
    const unsigned char* end = footer;
    const auto take = [&](uint32_t& out) {
        if (at + 4 > end) {
            return false;
        }
        out = readU32(at);
        at += 4;
        return true;
    };

    KernelInfo info;
    bool       matched = false;
    for (uint32_t e = 0; e < entryCount; ++e) {
        uint32_t nameLength = 0;
        if (!take(nameLength)) {
            return parsed;
        }
        const uint32_t padded = (nameLength + 3) & ~uint32_t{3};
        if (at + padded > end) {
            return parsed;
        }
        const std::string_view name(reinterpret_cast<const char*>(at), nameLength);
        at += padded;
        uint32_t gx = 0, gy = 0, gz = 0;
        if (!take(gx) || !take(gy) || !take(gz)) {
            return parsed;
        }
        if (name == entry) {
            info.threadGroup = {gx == 0 ? 1 : gx, gy == 0 ? 1 : gy, gz == 0 ? 1 : gz};
            matched = true;
        }
    }
    uint32_t bufferCount = 0;
    if (!take(bufferCount)) {
        return parsed;
    }
    info.elementBytes.resize(bufferCount);
    for (uint32_t& element : info.elementBytes) {
        if (!take(element)) {
            return parsed;
        }
    }
    if (!take(info.uniformBytes)) {
        return parsed;
    }

    // The payload ends where the trailer starts, matched entry or not: a
    // trailer for other entries is still not part of the library.
    parsed.payloadBytes = bytes - trailerBytes;
    if (matched) {
        parsed.info = std::move(info);
    }
    return parsed;
}

bool registerKernel(std::string_view name, std::string_view entry,
                    const void* blob, size_t bytes) {
    if (name.empty() || blob == nullptr || bytes == 0) {
        return false;
    }
    if (kernels::find(name) != nullptr) {
        std::fprintf(stderr,
                     "gpe: a kernel named '%.*s' is compiled into gpe; the one "
                     "being registered would never be found. Rename it.\n",
                     static_cast<int>(name.size()), name.data());
        return false;
    }
    const std::lock_guard<std::mutex> held(tableGuard());
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
    slot->raw = kernels::Blob{slot->name, slot->entry,
                              static_cast<const unsigned char*>(blob), bytes};
    ParsedKernelBlob parsed = parseKernelBlob(blob, bytes, entry);
    slot->resolved.blob = kernels::Blob{slot->name, slot->entry,
                                        static_cast<const unsigned char*>(blob),
                                        parsed.payloadBytes};
    slot->resolved.info = std::move(parsed.info);
    return true;
}

void unregisterKernel(std::string_view name) {
    const std::lock_guard<std::mutex> held(tableGuard());
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

std::optional<KernelInfo> kernelInfo(std::string_view name) {
    const kernels::Resolved* resolved = kernels::resolve(name);
    return resolved != nullptr ? resolved->info : std::nullopt;
}

namespace kernels {

const Blob* lookup(std::string_view name) {
    if (const Blob* compiled = find(name); compiled != nullptr) {
        return compiled;
    }
    const std::lock_guard<std::mutex> held(tableGuard());
    const Registered* registered = findRegistered(name);
    return registered == nullptr ? nullptr : &registered->raw;
}

const Resolved* resolve(std::string_view name) {
    // The build's own table first. A plugin must not be able to shadow
    // `display` or `scale` by naming a kernel after one of them.
    if (const Blob* compiled = find(name); compiled != nullptr) {
        const std::lock_guard<std::mutex> held(tableGuard());
        for (Registered* done : compiledResolved()) {
            if (done->name == name) {
                return &done->resolved;
            }
        }
        auto* made = new Registered();
        made->name = std::string(compiled->name);
        made->entry = std::string(compiled->entry);
        made->raw = *compiled;
        ParsedKernelBlob parsed = parseKernelBlob(compiled->data, compiled->size, compiled->entry);
        made->resolved.blob = Blob{made->name, made->entry, compiled->data, parsed.payloadBytes};
        made->resolved.info = std::move(parsed.info);
        compiledResolved().push_back(made);
        return &made->resolved;
    }
    const std::lock_guard<std::mutex> held(tableGuard());
    const Registered* registered = findRegistered(name);
    return registered == nullptr ? nullptr : &registered->resolved;
}

}   // namespace kernels
}   // namespace gpe
