#include "core/LanczosResize.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <future>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* name) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    failures += !ok;
}

template <typename T>
void constants_and_identity() {
    bool identity = true, constant = true;
    const std::array<T, 3> color = {T(0), T(57), T(201)};
    for (const auto& shape : {std::array<int, 4>{31, 23, 21, 16}, {32, 24, 16, 12},
                              {32, 24, 4, 3}, {1, 1, 13, 7}, {1024, 3, 3, 2},
                              {3, 1024, 2, 3}, {13, 7, 1, 1}, {13, 7, 13, 7}}) {
        const auto [sh, sw, dh, dw] = shape;
        std::vector<T> input((size_t)sh * sw * 3), output((size_t)dh * dw * 3);
        for (size_t i = 0; i < input.size(); ++i) input[i] = color[i % 3];
        spirula::resize_lanczos3<T, 3>(input.data(), sh, sw, output.data(), dh, dw);
        for (size_t i = 0; i < output.size(); ++i)
            constant &= std::abs((double)output[i] - color[i % 3]) < 1e-5;
        for (size_t i = 0; i < input.size(); ++i) input[i] = T(i % 251);
        std::vector<T> same(input.size());
        spirula::resize_lanczos3<T, 3>(input.data(), sh, sw, same.data(), sh, sw);
        identity &= input == same;
    }
    check(identity, "identity is exact for every channel");
    check(constant, "constants survive edges, fractional scales, tiny images and cache eviction");
}

void reference_pixels() {
    // Pillow 12.3.0, mode F, Resampling.LANCZOS; includes negative lobes and clipped edge supports.
    const float impulse[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    const float expected[] = {0.016946694f, -0.089246094f, 0.407383174f,
                             0.407383174f, -0.089246094f, 0.016946694f};
    float output[6];
    spirula::resize_lanczos3<float, 1>(impulse, 1, 9, output, 1, 6);
    bool matches = true;
    for (int i = 0; i < 6; ++i) matches &= std::abs(output[i] - expected[i]) < 1e-6f;
    check(matches, "fractional impulse agrees with independent Lanczos3 reference");

    const float detail[] = {0, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    const float detail_expected[] = {-0.105135538f, 0.669048488f, 0.788083971f, 0.5f,
                                    0.211916029f, 0.330951512f, 1.10513556f};
    float resized[7];
    spirula::resize_lanczos3<float, 1>(detail, 1, 10, resized, 1, 7);
    matches = true;
    for (int i = 0; i < 7; ++i) matches &= std::abs(resized[i] - detail_expected[i]) < 1e-6f;
    check(matches, "0.7 scale agrees with independent Lanczos3 reference");

    std::vector<float> tall(193 * 3);
    for (int y = 0; y < 193; ++y)
        for (int x = 0; x < 3; ++x) tall[(size_t)y * 3 + x] = (float)((x * 13 + y * 7) % 31) - 7;
    const float tall_expected[] = {7.75173235f, 7.93673706f, 8.05213165f, 8.01585484f,
                                  7.99041605f, 7.99730682f, 7.99757385f, 8.00212288f,
                                  8.00413799f, 8.00904274f, 7.97632551f, 7.95197678f,
                                  8.10507870f, 8.22820473f};
    float reduced[14];
    spirula::resize_lanczos3<float, 1>(tall.data(), 193, 3, reduced, 7, 2);
    matches = true;
    for (int i = 0; i < 14; ++i) matches &= std::abs(reduced[i] - tall_expected[i]) < 2e-6f;
    check(matches, "filter supports larger than the row cache match the reference");
}

double frequency_gain(int destination, double frequency) {
    constexpr int source = 1024;
    constexpr double pi = 3.14159265358979323846;
    std::vector<float> input(source), output((size_t)destination);
    for (int x = 0; x < source; ++x) input[(size_t)x] = (float)std::sin(2 * pi * frequency * x);
    spirula::resize_lanczos3<float, 1>(input.data(), 1, source, output.data(), 1, destination);
    double correlation = 0.0, energy = 0.0;
    for (int x = 16; x < destination - 16; ++x) {
        const double source_x = (x + 0.5) * source / destination - 0.5;
        const double wave = std::sin(2 * pi * frequency * source_x);
        correlation += output[(size_t)x] * wave;
        energy += wave * wave;
    }
    return std::abs(correlation / energy);
}

void frequency_response() {
    check(frequency_gain(717, 0.20) > 0.95, "70% scale retains resolvable high-frequency contrast");
    check(frequency_gain(717, 0.49) < 0.04, "70% scale suppresses frequencies above output Nyquist");
    for (int divisor : {2, 4, 8}) {
        check(frequency_gain(1024 / divisor, 0.20 / divisor) > 0.98,
              "power-of-two reductions preserve passband detail");
        check(frequency_gain(1024 / divisor, 0.9 / divisor) < 0.02,
              "power-of-two reductions widen the antialiasing filter");
    }
}

template <typename T>
void integer_clipping() {
    std::vector<T> edge(64), output(45);
    for (int i = 32; i < 64; ++i) edge[(size_t)i] = std::numeric_limits<T>::max();
    spirula::resize_lanczos3<T, 1>(edge.data(), 1, 64, output.data(), 1, 45);
    check(output[21] == 0 && output[23] == std::numeric_limits<T>::max(),
          "integer negative lobes saturate without wrapping");
}

void hdr_and_symmetry() {
    constexpr int sw = 37, sh = 29, dw = 25, dh = 19;
    std::vector<float> input(sw * sh * 3), transposed(input.size());
    std::vector<float> output(dw * dh * 3), other(output.size());
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x)
            for (int c = 0; c < 3; ++c) {
                const float value = (float)((x * 13 + y * 7 + c * 3) % 31) - 7.0f;
                input[((size_t)y * sw + x) * 3 + c] = value;
                transposed[((size_t)x * sh + y) * 3 + c] = value;
            }
    spirula::resize_lanczos3<float, 3>(input.data(), sh, sw, output.data(), dh, dw);
    spirula::resize_lanczos3<float, 3>(transposed.data(), sw, sh, other.data(), dw, dh);
    bool symmetry = true, negative = false, hdr = false;
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x)
            for (int c = 0; c < 3; ++c) {
                const float value = output[((size_t)y * dw + x) * 3 + c];
                symmetry &= std::abs(value - other[((size_t)x * dh + y) * 3 + c]) < 1e-5f;
                negative |= value < 0;
                hdr |= value > 1;
            }
    check(symmetry, "two-dimensional filtering is transpose-symmetric across RGB channels");
    check(negative && hdr, "floating-point images retain signed and HDR values");
}

}  // namespace

int main() {
    constants_and_identity<uint8_t>();
    constants_and_identity<uint16_t>();
    constants_and_identity<float>();
    reference_pixels();
    frequency_response();
    integer_clipping<uint8_t>();
    integer_clipping<uint16_t>();
    hdr_and_symmetry();
    std::vector<std::future<double>> concurrent;
    for (int i = 0; i < 8; ++i)
        concurrent.push_back(std::async(std::launch::async, [i] { return frequency_gain(700 + i, 0.2); }));
    bool isolated = true;
    for (auto& result : concurrent) isolated &= result.get() > 0.95;
    check(isolated, "decode threads have independent filter workspaces");
    return failures ? 1 : 0;
}
