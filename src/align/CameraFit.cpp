#include "align/WarpMath.h"
#include "sfm/geometry/AbsolutePose.h"
#include "core/LanczosResize.h"

#include <map>
#include <stdexcept>

namespace spirula::align {
namespace {
struct Observation {
    sfm::Vec3 point;
    std::array<double,2> source, pixel;
    double weight;
    int layer;
    bool holdout;
    std::array<double,2> normal{};
    double image_weight = 1;
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
    return o.normal[0]||o.normal[1]?std::abs(delta[0]*o.normal[0]+delta[1]*o.normal[1]):std::hypot(delta[0],delta[1]);
}

sfm::Pose fit_pose(const camhost::Camera& camera,const std::vector<Observation>& observations,const std::vector<int>& training,int dof=6) {
    sfm::Pose pose;double focal_scale=1;std::vector<double> weights(training.size());
    for(double scale:{20.0,5.0,2.0})for(int iteration=0;iteration<3;++iteration) {
        for(size_t j=0;j<training.size();++j) {
            const auto& o=observations[training[j]];const double e=error(camera,pose,o)/scale;
            weights[j]=std::sqrt(o.weight/std::sqrt(1+e*e));
        }
        auto residual=[&](const sfm::Pose& p,double,int j,double* r) {
            const auto& o=observations[training[j]];double pixel[2];
            if(!project(camera,p,o.point,pixel)){r[0]=r[1]=1e3;return;}
            camhost::pixel_difference(camera,pixel,o.source.data(),r);
            if(o.normal[0]||o.normal[1]){r[0]=r[0]*o.normal[0]+r[1]*o.normal[1];r[1]=0;}
            r[0]*=weights[j];r[1]*=weights[j];
        };
        sfm::pose_detail::lmRefine((int)training.size(),residual,dof,pose,focal_scale,20);
    }
    return pose;
}

void set_correction(Sim3& correction,const sfm::Pose& pose) {
    std::copy(pose.R.begin(),pose.R.end(),correction.R);
    correction.t[0]=pose.t.x;correction.t[1]=pose.t.y;correction.t[2]=pose.t.z;
}

std::vector<Observation> image_matches(const std::vector<Observation>& observations,const Image& photo,const Image& render,
                                      const std::vector<uint8_t>& mask) {
    auto refined=observations;const auto pg=detail::gray(photo),rg=detail::gray(render);
    for(int scale:{4,2,1}) {
        const int w=std::max(16,photo.width/scale),h=std::max(16,photo.height/scale);
        std::vector<float> a((size_t)w*h),b(a.size());
        resize_lanczos3<float,1>(pg.data(),photo.height,photo.width,a.data(),h,w);
        resize_lanczos3<float,1>(rg.data(),render.height,render.width,b.data(),h,w);
        const double ax=(double)w/photo.width,ay=(double)h/photo.height;
        for(size_t k=0;k<refined.size();++k) {
            auto& o=refined[k];const auto& initial=observations[k];
            const float x=(float)((o.pixel[0]+.5)*ax-.5),y=(float)((o.pixel[1]+.5)*ay-.5);
            const float qx=(float)((o.source[0]+.5)*ax-.5),qy=(float)((o.source[1]+.5)*ay-.5);
            float best=detail::correlation(b,a,w,h,x,y,qx,qy,3),bx=qx,by=qy;
            if(best<-1)continue;
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
                const float score=detail::correlation(b,a,w,h,x,y,qx+dx,qy+dy,3);
                if(score>best+.001f){best=score;bx=qx+dx;by=qy+dy;}
            }
            auto subpixel=[&](float left,float right) {
                const float den=left-2*best+right;
                return den<-.00001f?std::clamp(.5f*(left-right)/den,-.5f,.5f):0.0f;
            };
            const float dx=subpixel(detail::correlation(b,a,w,h,x,y,bx-1,by,3),detail::correlation(b,a,w,h,x,y,bx+1,by,3));
            const float dy=subpixel(detail::correlation(b,a,w,h,x,y,bx,by-1,3),detail::correlation(b,a,w,h,x,y,bx,by+1,3));
            const std::array<double,2> q{(bx+dx+.5)/ax-.5,(by+dy+.5)/ay-.5};
            if(best>.6f&&std::hypot(q[0]-initial.source[0],q[1]-initial.source[1])<=4&&
                (mask.empty()||mask[(size_t)std::lround(q[1])*photo.width+(int)std::lround(q[0])]))o.source=q;
        }
    }
    return refined;
}

bool refine_image_pose(const camhost::Camera& camera,std::vector<Observation> observations,
                       const Image& photo,const Image& render,const std::vector<uint8_t>& mask,sfm::Pose& pose,int dof,sfm::Pose& polished) {
    std::vector<double> ranges;for(const auto& o:observations)ranges.push_back(o.point.norm());
    auto middle=ranges.begin()+ranges.size()/2;std::nth_element(ranges.begin(),middle,ranges.end());
    for(auto& o:observations)o.image_weight*=std::clamp(*middle/o.point.norm(),.25,16.);
    double scene_scale=1;
    if(dof==6) {
        scene_scale=*middle;
        // Finite differences must move subpixel image samples regardless of world units.
        for(auto& o:observations)o.point=o.point*(1/scene_scale);
        pose.t=pose.t*(1/scene_scale);
    }
    const auto pg=detail::gray(photo),rg=detail::gray(render);double gain=0;size_t held=0;bool coarse_gain=false;sfm::Pose coarse_pose=pose;
    for(int scale:{4,2,1}) {
        const int w=std::max(16,photo.width/scale),h=std::max(16,photo.height/scale);
        std::vector<float> a((size_t)w*h),b(a.size());
        resize_lanczos3<float,1>(pg.data(),photo.height,photo.width,a.data(),h,w);
        resize_lanczos3<float,1>(rg.data(),render.height,render.width,b.data(),h,w);
        const double ax=(double)w/photo.width,ay=(double)h/photo.height;
        auto correlation=[&](const sfm::Pose& p,int j) {
            const auto& o=observations[j];double pixel[2];
            if(!project(camera,p,o.point,pixel)||pixel[0]<0||pixel[1]<0||pixel[0]>photo.width-1||pixel[1]>photo.height-1)return -2.0f;
            if(!mask.empty()&&!mask[(size_t)std::lround(pixel[1])*photo.width+(int)std::lround(pixel[0])])return -2.0f;
            return detail::correlation(b,a,w,h,(float)((o.pixel[0]+.5)*ax-.5),(float)((o.pixel[1]+.5)*ay-.5),
                                        (float)((pixel[0]+.5)*ax-.5),(float)((pixel[1]+.5)*ay-.5),3);
        };
        std::vector<int> train,validation;
        for(size_t j=0;j<observations.size();++j)if(correlation(sfm::Pose{},(int)j)>-1) {
            if(observations[j].holdout)validation.push_back((int)j);
            else if(j%4==0)train.push_back((int)j);
        }
        if(train.size()<32||validation.size()<10){if(scale==1)break;pose.t=pose.t*scene_scale;return false;}
        auto weight=[&](int j){return scale==1?observations[j].image_weight:1.;};
        auto residual=[&](const sfm::Pose& p,double,int j,double* r) {
            const float score=correlation(p,train[j]);
            r[0]=(score<-1?1e3:std::sqrt(std::max(1e-6f,1-score)))*std::sqrt(weight(train[j]));r[1]=0;
        };
        auto cost=[&](const sfm::Pose& p){double sum=0,r[2];for(size_t j=0;j<train.size();++j){residual(p,1,(int)j,r);sum+=r[0]*r[0];}return sum;};
        if(scale==4&&cost(sfm::Pose{})<cost(pose))pose=sfm::Pose{};
        const auto previous=pose;double focal=1;
        sfm::pose_detail::lmRefine((int)train.size(),residual,dof,pose,focal,12,true);
        double improvement=0,unweighted=0,total_weight=0;
        for(int j:validation) {
            const double delta=std::max(-1.0f,correlation(pose,j))-std::max(-1.0f,correlation(previous,j));
            improvement+=weight(j)*delta;unweighted+=delta;total_weight+=weight(j);
        }
        if(improvement<(scale==1?.0005*total_weight:0)||unweighted<-.001*validation.size())pose=previous;
        gain=0;held=validation.size();
        for(int j:validation)gain+=std::max(-1.0f,correlation(pose,j))-correlation(sfm::Pose{},j);
        if(scale==2){coarse_gain=gain/held>.001;coarse_pose=pose;}
    }
    polished=pose;polished.t=polished.t*scene_scale;
    pose=coarse_pose;pose.t=pose.t*scene_scale;
    return coarse_gain;
}
}

const char* camera_fit_status(CameraFitStatus status) {
    switch(status) {
        case CameraFitStatus::InsufficientSupport: return "insufficient_geometry_matches";
        case CameraFitStatus::InsufficientHoldout: return "insufficient_validation_matches";
        case CameraFitStatus::NoImprovement: return "pose_improvement_below_threshold";
        case CameraFitStatus::Accepted: return "accepted";
        case CameraFitStatus::ImageValidated: return "image_validated";
        case CameraFitStatus::MatchValidated: return "fresh_matches_validated";
        case CameraFitStatus::StructureValidated: return "rendered_edges_validated";
        case CameraFitStatus::ImageRejected: return "render_did_not_improve";
    }
    return "unknown";
}

CameraAlignment refine_camera(const std::vector<float>& depth, const std::vector<float>& opacity,
                              const roma::PairPrediction& matches, const Settings& settings,
                              const camhost::Camera& camera, const std::vector<uint8_t>& mask,
                              const Image* photo,const Image* render,bool structure_only) {
    CameraAlignment result;
    const int width=camera.width,height=camera.height;
    auto valid_prediction=[](const roma::Prediction& p){return p.width>0&&p.height>0&&p.warp.size()==(size_t)p.width*p.height*2&&p.overlap.size()==(size_t)p.width*p.height;};
    if (width<16 || height<16 || depth.size()!=(size_t)width*height || opacity.size()!=depth.size() ||
        (!mask.empty() && mask.size()!=depth.size()) || !(camera.fx>0) || !(camera.fy>0) ||
        !std::isfinite(camera.fx)||!std::isfinite(camera.fy)||!std::isfinite(camera.cx)||!std::isfinite(camera.cy)||
        !valid_prediction(matches.forward)||!valid_prediction(matches.backward)||
        !(settings.min_overlap>0&&settings.min_overlap<=1)||!(settings.max_cycle>0)||!std::isfinite(settings.max_cycle)||
        settings.max_displacement<0||!std::isfinite(settings.max_displacement)||settings.spacing<4)
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
        observations.push_back({{z*ray[0],z*ray[1],z*ray[2]},{qx,qy},{(double)x,(double)y},c*bc,layer,held});
        observations.back().image_weight=c*bc;
        if (!held) ++populations[layer];
    }
    if (observations.size() < 64) {
        for(const auto& o:observations)if(o.holdout)++result.holdout;else ++result.samples;
        return result;
    }
    std::vector<int> training;
    for (size_t i = 0; i < observations.size(); ++i) if (!observations[i].holdout) training.push_back((int)i);
    for (auto& o : observations) o.weight *= std::min(16.0,(double)training.size()/std::max(1,populations[o.layer]));
    const auto pose=fit_pose(camera,observations,training);
    double before = 0, after = 0, total = 0;
    for (const auto& o : observations) if (o.holdout) {
        before += o.weight*std::min(20.0,error(camera,sfm::Pose{},o));
        after += o.weight*std::min(20.0,error(camera,pose,o));
        total += o.weight; ++result.holdout;
    }
    result.samples = training.size();
    if (total > 0) { result.error_before = before/total; result.error_after = after/total; }
    if (result.holdout < 10) {result.status=CameraFitStatus::InsufficientHoldout;return result;}
    set_correction(result.correction,pose);result.candidate=after<before;
    if(photo&&render) {
        if(photo->width!=width||photo->height!=height||render->width!=width||render->height!=height||
            photo->rgb.size()!=(size_t)width*height*3||render->rgb.size()!=photo->rgb.size())throw std::runtime_error("camera images differ from calibration");
        if(!structure_only) {
            const auto refined=image_matches(observations,*photo,*render,mask);
            const auto image_pose=fit_pose(camera,refined,training);
            double refined_before=0,refined_after=0,raw_after=0;
            for(size_t k=0;k<observations.size();++k)if(observations[k].holdout) {
                const auto& o=observations[k];
                refined_before+=o.weight*std::min(20.0,error(camera,sfm::Pose{},refined[k]));
                refined_after+=o.weight*std::min(20.0,error(camera,image_pose,refined[k]));
                raw_after+=o.weight*std::min(20.0,error(camera,image_pose,o));
            }
            result.photometric_error_after=raw_after/total;
            result.photometric_candidate=refined_after<.995*refined_before&&raw_after<before+.5*total;
            set_correction(result.photometric_correction,image_pose);
            auto rotation=fit_pose(camera,refined,training,3);sfm::Pose rotation_polish;
            const bool image_gain=refine_image_pose(camera,observations,*photo,*render,mask,rotation,3,rotation_polish);
            refined_after=0;raw_after=0;
            for(size_t k=0;k<observations.size();++k)if(observations[k].holdout) {
                const auto& o=observations[k];
                refined_after+=o.weight*std::min(20.0,error(camera,rotation,refined[k]));
                raw_after+=o.weight*std::min(20.0,error(camera,rotation,o));
            }
            result.rotation_error_after=raw_after/total;
            result.rotation_candidate=image_gain&&raw_after<before+total;
            set_correction(result.rotation_correction,rotation);
            auto direct=image_pose;sfm::Pose direct_polish;
            const bool direct_gain=refine_image_pose(camera,observations,*photo,*render,mask,direct,6,direct_polish);
            raw_after=0;
            for(const auto& o:observations)if(o.holdout)raw_after+=o.weight*std::min(20.0,error(camera,direct,o));
            result.direct_error_after=raw_after/total;
            result.direct_candidate=direct_gain&&raw_after<before+total;
            set_correction(result.direct_correction,direct);
            auto polish=[&](CameraPolish& out,const sfm::Pose& coarse,const sfm::Pose& fine,bool candidate) {
                double error_after=0;
                for(const auto& o:observations)if(o.holdout)error_after+=o.weight*std::min(20.,error(camera,fine,o));
                out.geometry_error=error_after/total;
                out.candidate=candidate&&error_after<before+total&&(coarse.R!=fine.R||
                    coarse.t.x!=fine.t.x||coarse.t.y!=fine.t.y||coarse.t.z!=fine.t.z);
                set_correction(out.correction,fine);
            };
            polish(result.rotation_polish,rotation,rotation_polish,result.rotation_candidate);
            polish(result.direct_polish,direct,direct_polish,result.direct_candidate);
        }
        Field grid;grid.width=width;grid.height=height;grid.spacing=settings.spacing;
        grid.columns=(width+grid.spacing-1)/grid.spacing+1;grid.rows=(height+grid.spacing-1)/grid.spacing+1;
        grid.displacement.resize((size_t)grid.columns*grid.rows);
        const auto anchors=detail::structure_matches(grid,detail::gray(*photo),detail::gray(*render),depth,opacity,matches,settings,mask);
        auto combined=observations;auto indices=training;size_t held=0;double edge_before=0;
        for(int gy=0;gy<grid.rows;++gy)for(int gx=0;gx<grid.columns;++gx) {
            const auto& a=anchors[(size_t)gy*grid.columns+gx];if(a[3]<=0)continue;
            const float x=std::min(gx*grid.spacing,width)-.5f,y=std::min(gy*grid.spacing,height)-.5f;
            float qx,qy,c;if(!detail::match(matches.backward,x,y,width,height,qx,qy,c))continue;
            const float offset=a[2]-a[0]*(qx-x)-a[1]*(qy-y);qx+=a[0]*offset;qy+=a[1]*offset;
            const int ix=(int)std::lround(qx),iy=(int)std::lround(qy);
            if(ix<8||iy<8||ix>=width-8||iy>=height-8)continue;
            float z=1e30f;
            for(int j=-4;j<=4;++j)for(int i=-4;i<=4;++i) {
                const size_t k=(size_t)(iy+j)*width+ix+i;const float v=depth[k];
                if(!(v>0)||!std::isfinite(v)||!(opacity[k]>=.95f))continue;
                bool stable=true;for(int d:{-1,1,-width,width})stable&=std::abs(depth[k+d]-v)<.04f*v;
                if(stable)z=std::min(z,v);
            }
            if(z==1e30f)continue;
            const double pixel[2]={qx,qy};double ray[3];if(!camhost::pixel_ray(camera,pixel,ray))continue;
            const bool holdout=(gx+3*gy)%7==3;
            if(!holdout)indices.push_back((int)combined.size());else{++held;edge_before+=std::abs(a[2]);}
            combined.push_back({{z*ray[0],z*ray[1],z*ray[2]},{x,y},{qx,qy},8,0,holdout,{a[0],a[1]}});
        }
        if(held>=12&&combined.size()>=observations.size()+64) {
            for(double weight:{8.,2.}) {
                for(size_t j=observations.size();j<combined.size();++j)combined[j].weight=weight;
                const auto structural=fit_pose(camera,combined,indices);
                double edge_after=0,raw_after=0;
                for(size_t j=0;j<combined.size();++j)if(combined[j].holdout) {
                    const double e=error(camera,structural,combined[j]);
                    if(j>=observations.size())edge_after+=e;else raw_after+=combined[j].weight*std::min(20.,e);
                }
                result.structure_error_before=edge_before/held;result.structure_error_after=edge_after/held;
                result.structure_geometry_error=raw_after/total;
                result.structure_candidate=edge_after<.8*edge_before&&edge_before-edge_after>.25*held&&raw_after<before+.5*total;
                set_correction(result.structure_correction,structural);
                if(result.structure_candidate)break;
            }
        }
    }
    if (!(after < .9*before)) {result.status=CameraFitStatus::NoImprovement;return result;}
    result.accepted=true;result.status=CameraFitStatus::Accepted;return result;
}
}  // namespace spirula::align
