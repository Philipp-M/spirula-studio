#pragma once

#include "roma/Roma.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace spirula::align {

struct Image {
    int width = 0, height = 0;
    std::vector<float> rgb;
};

struct Settings {
    int spacing = 8, patch_radius = 3, refine_radius = 3;
    float min_overlap = 0.7f, max_cycle = 1.5f, max_displacement = 40.0f;
    float min_correlation = 0.75f, depth_jump = 0.04f;
};

struct MatchDiagnostics {
    size_t sampled = 0, finite = 0, overlap = 0, cycle = 0, bounded = 0;
    size_t old_opacity = 0, old_depth = 0, old_correlation = 0, old_accepted = 0;
    double mean_overlap = 0, median_cycle = 0, median_displacement = 0;
};

struct Triangle {
    std::array<std::array<float, 2>, 3> source, target;
    std::array<float, 3> depth;
};

struct Field {
    int width = 0, height = 0, columns = 0, rows = 0, spacing = 0;
    float depth_jump = .04f;
    std::vector<std::array<float, 2>> displacement;
    std::vector<float> depth, support;
    std::vector<uint8_t> valid;
    std::vector<std::array<float, 2>> audit;
    std::vector<Triangle> triangles;
    std::vector<std::vector<uint32_t>> bins;
    MatchDiagnostics diagnostics;
    size_t anchors = 0;
    float correction_scale = 1;
    bool fallback = false;
    bool source(float x, float y, float& sx, float& sy) const;
};

struct Audit {
    size_t count = 0, invalid = 0;
    double before = 0, after = 0;
    double gradient_before = 0, gradient_after = 0;
};

struct Coverage {
    size_t pixels = 0, covered = 0, masked = 0;
    std::vector<uint8_t> reason;
};

Field fit(const Image& photo, const Image& render, const std::vector<float>& depth,
          const std::vector<float>& opacity, const roma::PairPrediction& matches,
          const Settings& settings, const std::vector<uint8_t>& source_mask = {});
Audit audit(const Image& photo, const Image& corrected, const Image& render, const Field& field);
std::vector<Audit> audit_common(const Image& photo, const std::vector<Image>& corrected,
                                const Image& render, const std::vector<std::vector<uint8_t>>& masks);
void resample(const std::vector<uint8_t>& original, int width, int height,
              const std::vector<uint8_t>& original_mask, const Field& field,
              std::vector<uint8_t>& corrected, std::vector<uint8_t>& mask,
              Coverage* coverage = nullptr);
void index_field(Field& field);
void use_identity(Field& field);
void save_field(const Field& field, const std::filesystem::path& path);
Field load_field(const std::filesystem::path& path);
std::filesystem::path checked_output(const std::filesystem::path& dataset,
                                     const std::filesystem::path& output);

}  // namespace spirula::align
