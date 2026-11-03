#include "align/WarpMath.h"
#include "sfm/geometry/AbsolutePose.h"

#include <map>
#include <stdexcept>

namespace spirula::align {
namespace {
struct Observation {
    sfm::Vec3 point;
    std::array<double,2> source;
    double weight;
    int layer;
    bool holdout;
};

bool project(const camhost::Camera& camera, const sfm::Pose& pose, const sfm::Vec3& point, double* pixel) {
    const auto p = sfm::mul(pose.R,point)+pose.t;
    const double ray[3] = {p.x,p.y,p.z};
    return camhost::ray_pixel(camera,ray,pixel);
}

double error(const camhost::Camera& camera, const sfm::Pose& pose, const Observation& o) {
    double pixel[2],delta[2];
    if (!project(camera,pose,o.point,pixel)) return 1e3;
    camhost::pixel_difference(camera,pixel,o.source.data(),delta);
    return std::hypot(delta[0],delta[1]);
}
}

CameraAlignment refine_camera(const std::vector<float>& depth, const std::vector<float>& opacity,
                              const roma::PairPrediction& matches, const Settings& settings,
                              const camhost::Camera& camera, const std::vector<uint8_t>& mask) {
    CameraAlignment result;
    const int width=camera.width,height=camera.height;
    auto valid_prediction=[](const roma::Prediction& p){return p.width>0&&p.height>0&&p.warp.size()==(size_t)p.width*p.height*2&&p.overlap.size()==(size_t)p.width*p.height;};
    if (width<16 || height<16 || depth.size()!=(size_t)width*height || opacity.size()!=depth.size() ||
        (!mask.empty() && mask.size()!=depth.size()) || !(camera.fx>0) || !(camera.fy>0) ||
        !std::isfinite(camera.fx)||!std::isfinite(camera.fy)||!std::isfinite(camera.cx)||!std::isfinite(camera.cy)||
        !valid_prediction(matches.forward)||!valid_prediction(matches.backward)||
        !(settings.min_overlap>0&&settings.min_overlap<=1)||!(settings.max_cycle>0)||!std::isfinite(settings.max_cycle)||
        settings.max_displacement<0||!std::isfinite(settings.max_displacement))
        throw std::runtime_error("invalid camera alignment inputs");
    std::vector<Observation> observations;
    std::map<int,int> populations;
    const int step = std::max(16,std::max(width,height)/160);
    for (int y = step/2; y < height-2; y += step) for (int x = step/2; x < width-2; x += step) {
        const size_t k = (size_t)y*width+x;
        const float z = depth[k];
        if (!(z > 0) || !std::isfinite(z) || !std::isfinite(opacity[k]) || opacity[k] < .95f) continue;
        bool stable = true;
        for (int offset : {-2,2,-2*width,2*width})
            stable &= std::isfinite(depth[k+offset]) && std::abs(depth[k+offset]-z) < .04f*z;
        if (!stable) continue;
        float qx,qy,c,bx,by,bc;
        if (!detail::match(matches.forward,x,y,width,height,qx,qy,c) || c < settings.min_overlap ||
            qx < 0 || qy < 0 || qx > width-1 || qy > height-1 ||
            !detail::match(matches.backward,qx,qy,width,height,bx,by,bc) || bc < settings.min_overlap ||
            std::hypot(bx-x,by-y) > settings.max_cycle ||
            std::hypot(qx-x,qy-y) > detail::displacement_limit(settings,width,height)) continue;
        if (!mask.empty() && !mask[(size_t)std::lround(qy)*width+(int)std::lround(qx)]) continue;
        const double pixel[2] = {(double)x,(double)y}; double ray[3];
        if (!camhost::pixel_ray(camera,pixel,ray)) continue;
        const int layer = (int)std::floor(std::log2(z));
        const bool held = (x/step)%5 == 2 && (y/step)%5 == 2;
        observations.push_back({{z*ray[0],z*ray[1],z*ray[2]},{qx,qy},c*bc,layer,held});
        if (!held) ++populations[layer];
    }
    if (observations.size() < 64) return result;
    std::vector<int> training;
    for (size_t i = 0; i < observations.size(); ++i) if (!observations[i].holdout) training.push_back((int)i);
    for (auto& o : observations) o.weight *= std::min(16.0,(double)training.size()/std::max(1,populations[o.layer]));
    sfm::Pose pose; double focal_scale = 1;
    std::vector<double> weights(training.size());
    for (double scale : {20.0,5.0,2.0}) for (int iteration = 0; iteration < 3; ++iteration) {
        for (size_t j = 0; j < training.size(); ++j) {
            const auto& o = observations[training[j]];
            const double e = error(camera,pose,o)/scale;
            weights[j] = std::sqrt(o.weight/std::sqrt(1+e*e));
        }
        auto residual = [&](const sfm::Pose& p, double, int j, double* r) {
            const auto& o = observations[training[j]]; double pixel[2];
            if (!project(camera,p,o.point,pixel)) { r[0] = r[1] = 1e3; return; }
            camhost::pixel_difference(camera,pixel,o.source.data(),r);
            r[0] *= weights[j]; r[1] *= weights[j];
        };
        sfm::pose_detail::lmRefine((int)training.size(),residual,6,pose,focal_scale,20);
    }
    double before = 0, after = 0, total = 0;
    for (const auto& o : observations) if (o.holdout) {
        before += o.weight*std::min(20.0,error(camera,sfm::Pose{},o));
        after += o.weight*std::min(20.0,error(camera,pose,o));
        total += o.weight; ++result.holdout;
    }
    result.samples = training.size();
    if (total > 0) { result.error_before = before/total; result.error_after = after/total; }
    if (result.holdout < 10 || !(after < .9*before)) return result;
    std::copy(pose.R.begin(),pose.R.end(),result.correction.R);
    result.correction.t[0]=pose.t.x;result.correction.t[1]=pose.t.y;result.correction.t[2]=pose.t.z;
    result.accepted=true;return result;
}
}  // namespace spirula::align
