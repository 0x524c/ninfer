#include "ops/linear/fp8/fp8_dispatch.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy,
                             std::int32_t tokens) {
    if (tokens <= 0 || !is_fp8_linear_problem(output_rows, input_rows)) {
        throw std::invalid_argument("fp8 linear: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear: unsupported policy");
    }

    switch (resolve_fp8_problem(output_rows, input_rows)) {
    case Fp8Problem::AttnInput:
        return tokens >= 2 ? Fp8LinearRoute::A8 : Fp8LinearRoute::A16;
    case Fp8Problem::GdnInput:
        return tokens >= 2 ? Fp8LinearRoute::A8 : Fp8LinearRoute::A16;
    }
    throw std::logic_error("unreachable FP8 linear problem");
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for (std::int32_t token = 0; token < x.ne[1]; ++token) {
        auto* input = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token) * weight.n * sizeof(std::uint16_t);
        Tensor input_token(input, DType::BF16, {weight.k, 1});
        Tensor output_token(output, DType::BF16, {weight.n, 1});
        launch_fp8_decode(input_token, weight, output_token, stream);
    }
}

} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                LinearPolicy policy, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Fp8LinearRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Fp8LinearRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    launch_fp8_a8(x, weight, out, scratch, stream);
}

} // namespace ninfer::ops::detail
