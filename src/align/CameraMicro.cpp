#include "align/WarpMath.h"
#include "sfm/geometry/AbsolutePose.h"
#include <stdexcept>

namespace spirula::align {
CameraPolish polish_camera(const Image& photo,const Image& render,const std::vector<float>& depth,
                          const std::vector<float>& opacity,const camhost::Camera& camera,
                          const std::vector<uint8_t>& mask) {
    const int w=photo.width,h=photo.height;const size_t n=(size_t)w*h;
    if(w<16||h<16||render.width!=w||render.height!=h||camera.width!=w||camera.height!=h||
       photo.rgb.size()!=n*3||render.rgb.size()!=n*3||depth.size()!=n||opacity.size()!=n||
       (!mask.empty()&&mask.size()!=n))throw std::runtime_error("invalid camera micro-refinement dimensions");
    const auto pg=detail::gray(photo),rg=detail::gray(render);
    struct Patch {sfm::Vec3 point;int x,y;double weight;bool held;};
    std::vector<Patch> patches;std::vector<float> ranges;
    const int step=std::max(8,std::max(w,h)/128);
    for(int y=step/2;y<h-8;y+=step)for(int x=step/2;x<w-8;x+=step) {
        const size_t k=(size_t)y*w+x;const float z=depth[k];
        if(!(z>0)||!std::isfinite(z)||!(opacity[k]>=.95f))continue;
        bool valid=true;
        for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i) {
            const size_t p=(size_t)(y+j)*w+x+i;
            valid&=std::isfinite(depth[p])&&std::abs(depth[p]-z)<.04f*z&&(mask.empty()||mask[p]);
        }
        if(!valid||detail::correlation(rg,pg,w,h,x,y,x,y,3)<.4f)continue;
        const double pixel[2]={(double)x,(double)y};double ray[3];
        if(!camhost::pixel_ray(camera,pixel,ray))continue;
        patches.push_back({{ray[0]*z,ray[1]*z,ray[2]*z},x,y,1,(x/step+3*(y/step))%5==0});ranges.push_back(z);
    }
    CameraPolish result;result.samples=patches.size();if(patches.size()<80)return result;
    auto mid=ranges.begin()+ranges.size()/2;std::nth_element(ranges.begin(),mid,ranges.end());const double scale=*mid;
    std::vector<int> training,held;
    for(size_t j=0;j<patches.size();++j) {
        auto& p=patches[j];p.weight*=std::clamp(std::pow(scale/p.point.norm(),2),.25,64.);
        p.point=p.point*(1/scale);(p.held?held:training).push_back((int)j);
    }
    if(held.size()<20||training.size()<64)return result;
    auto score=[&](const sfm::Pose& pose,int j) {
        const auto& p=patches[j];const auto v=sfm::mul(pose.R,p.point)+pose.t;
        const double ray[3]={v.x,v.y,v.z};double q[2];
        if(!camhost::ray_pixel(camera,ray,q)||std::hypot(q[0]-p.x,q[1]-p.y)>3)return -1.f;
        if(!mask.empty())for(int yy=-3;yy<=3;++yy)for(int xx=-3;xx<=3;++xx) {
            const int x=(int)std::lround(q[0])+xx,y=(int)std::lround(q[1])+yy;
            if(x<0||y<0||x>=w||y>=h||!mask[(size_t)y*w+x])return -1.f;
        }
        return std::max(-1.f,detail::correlation(rg,pg,w,h,p.x,p.y,(float)q[0],(float)q[1],3));
    };
    sfm::Pose pose;double focal=1;
    auto residual=[&](const sfm::Pose& p,double,int j,double* r) {
        const int k=training[j];r[0]=std::sqrt(std::max(1e-8f,1-score(p,k))*patches[k].weight);r[1]=0;
    };
    sfm::pose_detail::lmRefine((int)training.size(),residual,6,pose,focal,10);
    double gain=0,total=0,movement=0;
    for(int j:held) {
        gain+=patches[j].weight*(score(pose,j)-score(sfm::Pose{},j));total+=patches[j].weight;
        const auto& p=patches[j];const auto v=sfm::mul(pose.R,p.point)+pose.t;
        const double ray[3]={v.x,v.y,v.z};double q[2];
        if(!camhost::ray_pixel(camera,ray,q))return result;
        movement=std::max(movement,std::hypot(q[0]-p.x,q[1]-p.y));
    }
    result.image_gain=gain/total;result.max_movement=movement;
    result.candidate=result.image_gain>.0002&&movement<=3;
    std::copy(pose.R.begin(),pose.R.end(),result.correction.R);
    result.correction.t[0]=pose.t.x*scale;result.correction.t[1]=pose.t.y*scale;result.correction.t[2]=pose.t.z*scale;
    return result;
}
}  // namespace spirula::align
