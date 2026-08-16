#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }
};

} // namespace ninfer::ops::detail
