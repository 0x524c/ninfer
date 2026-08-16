#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8AttnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8AttnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 attn_input_proj: unsupported policy");
    }
    return tokens >= 2 ? Fp8AttnInputRoute::A8 : Fp8AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    for (std::int32_t token = 0; token < x.ne[1]; ++token) {
        auto* input = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token) * kKvRows * sizeof(std::uint16_t);
        Tensor input_token(input, DType::BF16, {weight.k, 1});
        Tensor query_token(query, DType::BF16, {kQRows, 1});
        Tensor gate_token(output_gate, DType::BF16, {kQRows, 1});
        Tensor key_token(key, DType::BF16, {kKvRows, 1});
        Tensor value_token(value, DType::BF16, {kKvRows, 1});
        fp8_attn_input_decode_launch(input_token, weight, query_token, gate_token, key_token,
                                     value_token, stream);
    }
}

} // namespace

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8AttnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8AttnInputGeometry::kInputRows)
               : 0;
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    fp8_attn_input_a8_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
