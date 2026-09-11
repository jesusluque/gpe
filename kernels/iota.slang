// Copyright (c) 2026 gpe contributors.
//
// out[i] = first + i, for i < count.
//
// The smallest kernel that is not a picture: one-dimensional, a uint buffer
// rather than float4, and a group of 64 rather than 16x16. It exists so the
// facts a kernel's trailer carries -- its group, its element size, its uniform
// block -- have something to be checked against that differs from `scale` in
// every one of them.
import common;

struct IotaParams {
    uint count;
    uint first;
};

RWStructuredBuffer<uint>   values;
ConstantBuffer<IotaParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void iotaMain(uint3 tid: SV_DispatchThreadID) {
    // Whole groups are launched, so the last one runs past `count`.
    if (tid.x >= params.count) {
        return;
    }
    values[tid.x] = params.first + tid.x;
}
