#pragma once

#include "roma/Roma.h"
#include "data/CameraMath.h"
#include "core/Similarity.h"

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
    float min_overlap = 0.7f, max_cycle = 1.5f, max_displacement = 0.0f;
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

enum class CameraFitStatus { InsufficientSupport, InsufficientHoldout, NoImprovement, Accepted, ImageValidated, MatchValidated, ImageRejected };
const char* camera_fit_status(CameraFitStatus status);

struct CameraAlignment {
    Sim3 correction;
    Sim3 photometric_correction;
    Sim3 rotation_correction;
    bool accepted = false;
    bool candidate = false, photometric_candidate = false;
    bool rotation_candidate = false;
    size_t samples = 0, holdout = 0;
    double error_before = 0, error_after = 0;
    double photometric_error_after = 0;
    double rotation_error_after = 0;
    CameraFitStatus status = CameraFitStatus::InsufficientSupport;
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
    size_t anchors = 0, structure_anchors = 0;
    bool structure_fallback = false;
    float correction_scale = 1;
    bool fallback = false;
    bool source(float x, float y, float& sx, float& sy, bool allow_outside = false) const;
};

struct Audit {
    size_t count = 0, invalid = 0;
    double before = 0, after = 0;
    double gradient_before = 0, gradient_after = 0;
};

struct CameraAudit {
    bool accepted = false;
    size_t fine_count = 0, coarse_count = 0;
    double gain = 0, fine_before = 0, fine_after = 0, coarse_before = 0, coarse_after = 0;
};
CameraAudit audit_camera(const Image& photo, const Image& before, const Image& after,
                         const std::vector<uint8_t>& mask = {});

struct MatchAudit {
    bool accepted = false;
    size_t before_count = 0, after_count = 0, common_count = 0;
    double before = 0, after = 0;
    double supported_before = 0, supported_after = 0;
};
MatchAudit audit_camera_matches(const roma::PairPrediction& before, const roma::PairPrediction& after,
                                int width, int height, const Settings& settings,
                                const std::vector<uint8_t>& mask = {});

struct Coverage {
    size_t pixels = 0, covered = 0, masked = 0;
    std::vector<uint8_t> reason;
};

Field fit(const Image& photo, const Image& render, const std::vector<float>& depth,
          const std::vector<float>& opacity, const roma::PairPrediction& matches,
          const Settings& settings, const std::vector<uint8_t>& source_mask = {},
          bool refine_structure = true);
CameraAlignment refine_camera(const std::vector<float>& depth, const std::vector<float>& opacity,
                              const roma::PairPrediction& matches, const Settings& settings,
                              const camhost::Camera& camera, const std::vector<uint8_t>& source_mask = {},
                              const Image* photo = nullptr, const Image* render = nullptr);
Audit audit(const Image& photo, const Image& corrected, const Image& render, const Field& field);
std::vector<Audit> audit_common(const Image& photo, const std::vector<Image>& corrected,
                                const Image& render, const std::vector<std::vector<uint8_t>>& masks);
void resample(const std::vector<uint8_t>& original, int width, int height,
              const std::vector<uint8_t>& original_mask, const Field& field,
              std::vector<uint8_t>& corrected, std::vector<uint8_t>& mask,
              Coverage* coverage = nullptr);
void index_field(Field& field);
size_t transport_observations(const Field& field, int width, int height,
                              std::vector<std::array<double,2>>& observations);
void use_identity(Field& field);
void save_field(const Field& field, const std::filesystem::path& path);
Field load_field(const std::filesystem::path& path);
std::filesystem::path checked_output(const std::filesystem::path& dataset,
                                     const std::filesystem::path& output);

}  // namespace spirula::align
