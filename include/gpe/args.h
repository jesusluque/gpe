// Copyright (c) 2026 gpe contributors.
//
// What goes in the `args` blob of `dispatch`.
//
// The brief asks for one client code path on both backends, and that forces
// this file. Slang's CUDA output wants a struct of
//
//     {float4* data; size_t count;}   per buffer, plus a *pointer* to the
//                                     uniforms, which must be in device memory
//
// and its Metal output wants buffers bound to numbered slots with the uniforms
// as another slot. Neither is something a client can write down, and a client
// that wrote down either would compile on one machine and corrupt memory on
// the other.
//
// So the blob is ours, and it is the same shape everywhere:
//
//     uint32 bufferCount
//     uint32 uniformBytes
//     BufferId[bufferCount]      in the order the kernel declares them
//     uint32[bufferCount]        the size of one element of each
//     unsigned char[uniformBytes]
//
// The element sizes are there because Slang's CUDA output describes a buffer as
// {T* data; size_t count} and the count is in elements, not bytes. The host
// cannot infer T: a display target is RWStructuredBuffer<uint>, four bytes an
// element, while a plate is float4 at sixteen. Assuming float4 gives a count
// four times too small, and Slang's own bound check then silently drops three
// quarters of every write -- at some sizes, and not at others, which is the
// worst way for it to be wrong.
//
// Each backend reads that and builds whatever its own compiler asked for. The
// client writes buffers and a plain struct, and nothing it writes knows which
// machine it is on.
#pragma once

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <vector>

#include "gpe/types.h"

namespace gpe {

/// Builds a dispatch's argument blob.
///
/// Deliberately a builder and not a template: the layout has to be readable by
/// a backend that has never heard of the client's struct, and a template would
/// put the knowledge back in the header where only one side can see it.
class Args {
public:
    /// A buffer the kernel reads or writes, in declaration order. The order is
    /// the contract: kernels declare their buffers in the order the client adds
    /// them here, and nothing checks it because nothing can.
    Args& buffer(BufferId id, uint32_t elementBytes = kBytesPerPixel) {
        buffers_.push_back(id);
        elements_.push_back(elementBytes);
        return *this;
    }

    /// The kernel's uniform struct, copied. Whatever the client passes has to
    /// match the `struct` the .slang file declares, field for field -- which is
    /// why every one of them is four bytes wide in declaration order, so that
    /// no padding rule has to agree across two compilers.
    template <typename T>
    Args& uniforms(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "a uniform block is bytes on their way to a GPU");
        uniforms_.resize(sizeof(T));
        std::memcpy(uniforms_.data(), &value, sizeof(T));
        return *this;
    }

    /// The same, when the caller has bytes rather than a type.
    ///
    /// For a host relaying a uniform block from somewhere it cannot see the
    /// declaration -- a plugin, across a boundary. It is exactly as unchecked
    /// as the templated form, which is to say entirely: the bytes must match
    /// the kernel's struct and nothing here can know whether they do.
    Args& uniformBytes(const void* data, size_t bytes) {
        uniforms_.resize(bytes);
        if (data != nullptr && bytes > 0) {
            std::memcpy(uniforms_.data(), data, bytes);
        }
        return *this;
    }

    /// Serialises. The result stays alive as long as this object.
    [[nodiscard]] const std::vector<unsigned char>& blob() const {
        blob_.clear();
        const auto count = static_cast<uint32_t>(buffers_.size());
        const auto bytes = static_cast<uint32_t>(uniforms_.size());
        append(&count, sizeof(count));
        append(&bytes, sizeof(bytes));
        append(buffers_.data(), buffers_.size() * sizeof(BufferId));
        append(elements_.data(), elements_.size() * sizeof(uint32_t));
        append(uniforms_.data(), uniforms_.size());
        return blob_;
    }

    [[nodiscard]] const void* data() const { return blob().data(); }
    [[nodiscard]] size_t size() const { return blob().size(); }

    // --- the reading half, for a backend -----------------------------------

    struct View {
        const BufferId*      buffers = nullptr;
        /// Bytes per element of each buffer, parallel to `buffers`.
        const uint32_t*      elementBytes = nullptr;
        uint32_t             bufferCount = 0;
        const unsigned char* uniforms = nullptr;
        uint32_t             uniformBytes = 0;
        bool                 valid = false;
    };

    /// Reads a blob back. `valid` is false for anything that does not measure
    /// up -- a truncated blob is a caller bug, and a backend that trusted it
    /// would read past the end into whatever the stack had.
    [[nodiscard]] static View read(const void* data, size_t size) {
        View view;
        if (data == nullptr || size < 2 * sizeof(uint32_t)) {
            return view;
        }
        const auto* bytes = static_cast<const unsigned char*>(data);
        uint32_t count = 0;
        uint32_t uniformBytes = 0;
        std::memcpy(&count, bytes, sizeof(count));
        std::memcpy(&uniformBytes, bytes + sizeof(uint32_t), sizeof(uniformBytes));

        const size_t header = 2 * sizeof(uint32_t);
        const size_t handles = size_t{count} * sizeof(BufferId);
        const size_t elements = size_t{count} * sizeof(uint32_t);
        if (header + handles + elements + uniformBytes != size) {
            return view;
        }
        view.buffers = reinterpret_cast<const BufferId*>(bytes + header);
        view.elementBytes =
            reinterpret_cast<const uint32_t*>(bytes + header + handles);
        view.bufferCount = count;
        view.uniforms = bytes + header + handles + elements;
        view.uniformBytes = uniformBytes;
        view.valid = true;
        return view;
    }

private:
    void append(const void* from, size_t bytes) const {
        if (bytes == 0) {
            return;
        }
        const auto* p = static_cast<const unsigned char*>(from);
        blob_.insert(blob_.end(), p, p + bytes);
    }

    std::vector<BufferId>              buffers_;
    std::vector<uint32_t>              elements_;
    std::vector<unsigned char>         uniforms_;
    mutable std::vector<unsigned char> blob_;
};

/// The `scale` kernel's uniforms, matching `ScaleParams` in scale.slang field
/// for field. Four-byte members in declaration order, so there is no padding
/// for two compilers to disagree about.
struct ScaleUniforms {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;   // in pixels
    float    factor = 1.0f;
};
static_assert(sizeof(ScaleUniforms) == 16, "no padding, on any compiler");

}   // namespace gpe
