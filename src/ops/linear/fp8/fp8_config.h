#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Fp8Geometry {
    static_assert(OutputRows > 0 && (OutputRows % 16) == 0);
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <std::int32_t InputRows>
struct Fp8ActivationGeometry {
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kInputRows = InputRows;
};

enum class Fp8CodeCache : std::uint8_t {
    Default,
    Streaming,
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Fp8CodeCache CodeCache, int PhaseUnroll, int MinBlocksPerSm>
struct Fp8GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
};

using Fp8AttnInputGeometry      = Fp8Geometry<14336, 5120>;
using Fp8GdnInputGeometry       = Fp8Geometry<16384, 5120>;
using Fp8MlpGateUpGeometry      = Fp8Geometry<34816, 5120>;
using Fp8Activation5120Geometry = Fp8ActivationGeometry<5120>;

enum class Fp8Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
};

inline constexpr bool is_fp8_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return (output_rows == Fp8AttnInputGeometry::kOutputRows &&
            input_rows == Fp8AttnInputGeometry::kInputRows) ||
           (output_rows == Fp8GdnInputGeometry::kOutputRows &&
            input_rows == Fp8GdnInputGeometry::kInputRows) ||
           (output_rows == Fp8MlpGateUpGeometry::kOutputRows &&
            input_rows == Fp8MlpGateUpGeometry::kInputRows);
}

inline Fp8Problem resolve_fp8_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Fp8AttnInputGeometry::kOutputRows &&
        input_rows == Fp8AttnInputGeometry::kInputRows) {
        return Fp8Problem::AttnInput;
    }
    if (output_rows == Fp8GdnInputGeometry::kOutputRows &&
        input_rows == Fp8GdnInputGeometry::kInputRows) {
        return Fp8Problem::GdnInput;
    }
    if (output_rows == Fp8MlpGateUpGeometry::kOutputRows &&
        input_rows == Fp8MlpGateUpGeometry::kInputRows) {
        return Fp8Problem::MlpGateUp;
    }
    throw std::invalid_argument("unsupported FP8 problem");
}

template <class Geometry>
struct Fp8LinearDecodeProductionSchedule;

// RTX 5090 cold-cache winner for this exact problem. Each newly registered geometry supplies its
// own specialization so admission never silently inherits another problem's measured schedule.
template <>
struct Fp8LinearDecodeProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

} // namespace ninfer::ops::detail
