#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace spirula {
namespace lanczos_detail {

inline double kernel(double x) {
    x = std::abs(x);
    if (x >= 3.0) return 0.0;
    if (x < 1e-12) return 1.0;
    constexpr double pi = 3.14159265358979323846;
    const double p = pi * x;
    return (std::sin(p) / p) * (std::sin(p / 3.0) / (p / 3.0));
}

struct Span {
    int first, count;
    size_t offset;
};

struct Axis {
    int source = 0, destination = 0, max_taps = 0;
    std::vector<Span> spans;
    std::vector<double> weights;

    void prepare(int src, int dst) {
        if (source == src && destination == dst) return;
        source = src;
        destination = dst;
        spans.resize((size_t)dst);
        weights.clear();
        max_taps = 0;
        const double ratio = (double)src / dst;
        // Widen the low-pass footprint when shrinking; a fixed six-tap filter aliases.
        const double scale = std::max(1.0, ratio), support = 3.0 * scale;
        for (int i = 0; i < dst; ++i) {
            const double center = (i + 0.5) * ratio;
            const int first = (int)std::max(0.0, std::ceil(center - support - 0.5));
            const int end = (int)std::min((double)src, std::floor(center + support - 0.5) + 1.0);
            const size_t offset = weights.size();
            double sum = 0.0;
            for (int j = first; j < end; ++j) {
                const double w = kernel((j + 0.5 - center) / scale);
                weights.push_back(w);
                sum += w;
            }
            for (size_t j = offset; j < weights.size(); ++j) weights[j] /= sum;
            spans[(size_t)i] = {first, end - first, offset};
            max_taps = std::max(max_taps, end - first);
        }
    }
};

struct Workspace {
    Axis horizontal, vertical;
    std::vector<double> rows, accumulator;
    std::vector<int> row_ids;
};

}  // namespace lanczos_detail

// Interleaved channels, half-pixel centers; preserve floating HDR values and clamp integers only at output.
template <typename T, int C>
inline void resize_lanczos3(const T* src, int sh, int sw, T* dst, int dh, int dw) {
    static_assert(C > 0);
    if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0)
        throw std::invalid_argument("Lanczos resize dimensions must be positive");
    if (sh == dh && sw == dw) {
        std::copy_n(src, (size_t)sh * sw * C, dst);
        return;
    }

    static thread_local lanczos_detail::Workspace work;
    work.horizontal.prepare(sw, dw);
    work.vertical.prepare(sh, dh);
    // Bound scratch for extreme one-axis reductions, and reuse it between decoded images.
    const int cached_rows = std::min(64, work.vertical.max_taps);
    const size_t row_size = (size_t)dw * C;
    work.rows.resize((size_t)cached_rows * row_size);
    work.row_ids.assign((size_t)cached_rows, -1);
    work.accumulator.resize(row_size);

    for (int y = 0; y < dh; ++y) {
        std::fill(work.accumulator.begin(), work.accumulator.end(), 0.0);
        const auto& vertical = work.vertical.spans[(size_t)y];
        for (int k = 0; k < vertical.count; ++k) {
            const int source_y = vertical.first + k;
            const int slot = source_y % cached_rows;
            double* row = work.rows.data() + (size_t)slot * row_size;
            if (work.row_ids[(size_t)slot] != source_y) {
                const T* input = src + (size_t)source_y * sw * C;
                for (int x = 0; x < dw; ++x) {
                    const auto& horizontal = work.horizontal.spans[(size_t)x];
                    double sum[C] = {};
                    for (int j = 0; j < horizontal.count; ++j) {
                        const double weight = work.horizontal.weights[horizontal.offset + (size_t)j];
                        const T* pixel = input + (size_t)(horizontal.first + j) * C;
                        for (int c = 0; c < C; ++c) sum[c] += weight * pixel[c];
                    }
                    for (int c = 0; c < C; ++c) row[(size_t)x * C + c] = sum[c];
                }
                work.row_ids[(size_t)slot] = source_y;
            }
            const double weight = work.vertical.weights[vertical.offset + (size_t)k];
            for (size_t i = 0; i < row_size; ++i) work.accumulator[i] += weight * row[i];
        }
        for (size_t i = 0; i < row_size; ++i) {
            double value = work.accumulator[i];
            if constexpr (std::is_integral_v<T>)
                value = std::round(std::clamp(value, (double)std::numeric_limits<T>::lowest(),
                                             (double)std::numeric_limits<T>::max()));
            dst[(size_t)y * row_size + i] = (T)value;
        }
    }
}

}  // namespace spirula
