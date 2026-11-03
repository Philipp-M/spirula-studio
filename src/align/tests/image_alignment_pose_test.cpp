#include "align/Warp.h"
#include "sfm/geometry/AbsolutePose.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace A = spirula::align;

void foreground_edges() {
    const int w=512,h=512;
    camhost::Camera camera;camera.width=w;camera.height=h;camera.fx=camera.fy=500;camera.cx=camera.cy=256;
    A::Image photo{w,h,std::vector<float>((size_t)w*h*3)},render=photo;
    std::vector<float> depth((size_t)w*h),opacity(depth.size(),1);
    spirula::roma::PairPrediction matches;
    for(auto* p:{&matches.forward,&matches.backward}) {
        p->width=w;p->height=h;p->warp.resize(depth.size()*2);p->overlap.assign(depth.size(),1);
    }
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const size_t k=(size_t)y*w+x;float a=.2f,b=.2f,z=100;
        for(int edge:{104,240,376}) {
            a+=.6f*std::exp(-std::pow((x-edge-6.f)/2.f,2));
            b+=.6f*std::exp(-std::pow((x-edge)/2.f,2));
            if(std::abs(x-edge)<7)z=4;
            if(std::abs(x-edge-3)<16)for(auto* p:{&matches.forward,&matches.backward})p->overlap[k]=.1f;
        }
        for(int c=0;c<3;++c){photo.rgb[3*k+c]=a;render.rgb[3*k+c]=b;}
        const double pixel[2]={(double)x,(double)y};double ray[3];camhost::pixel_ray(camera,pixel,ray);
        depth[k]=(float)(z/ray[2]);
        for(auto* p:{&matches.forward,&matches.backward}){p->warp[2*k]=2*(x+.5f)/w-1;p->warp[2*k+1]=2*(y+.5f)/h-1;}
    }
    const auto fit=A::refine_camera(depth,opacity,matches,{},camera,{},&photo,&render);
    std::cout<<"foreground camera edges: "<<fit.structure_error_before<<" -> "<<fit.structure_error_after<<'\n';
    if(!fit.structure_candidate||fit.structure_error_before<4||fit.structure_error_after>1)
        throw std::runtime_error("camera fit ignored foreground edges absent from dense depth-interior matches");
    const auto incremental=A::refine_camera(depth,opacity,matches,{},camera,{},&photo,&render,true);
    for(int i=0;i<9;++i)if(fit.structure_correction.R[i]!=incremental.structure_correction.R[i])
        throw std::runtime_error("incremental camera proposal differs from the full candidate solve");
    for(int i=0;i<3;++i)if(fit.structure_correction.t[i]!=incremental.structure_correction.t[i])
        throw std::runtime_error("incremental camera translation differs from the full candidate solve");
    auto corrected=render;std::vector<float> corrected_depth(depth.size(),100);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const size_t k=(size_t)y*w+x;float value=.2f;
        for(int edge:{110,246,382}) {
            value+=.6f*std::exp(-std::pow((x-edge)/2.f,2));
            if(std::abs(x-edge)<7)corrected_depth[k]=4;
        }
        for(int c=0;c<3;++c)corrected.rgb[3*k+c]=value;
    }
    const auto audit=A::audit_camera_structure(photo,render,corrected,depth,opacity,corrected_depth,opacity,matches,{});
    if(!audit.accepted||audit.after>.5||audit.common_count<12)
        throw std::runtime_error("re-rendered foreground alignment did not validate on held-out edges");
    const auto unchanged=A::audit_camera_structure(photo,render,render,depth,opacity,depth,opacity,matches,{});
    if(unchanged.accepted)throw std::runtime_error("unchanged rendered edges validated a camera correction");
}

int main() {
    try {
        foreground_edges();
        sfm::Pose rotation;rotation.t={.1,.2,.3};double focal=1;
        const auto target=sfm::angleAxisToRotation({.01,-.02,.005});
        const std::array<sfm::Vec3,6> points{{{-1,-1,4},{1,-1,5},{-1,1,6},{1,1,7},{0,0,3},{.5,-.5,8}}};
        auto rotation_residual=[&](const sfm::Pose& pose,double,int i,double* r) {
            const auto a=sfm::mul(pose.R,points[i])+pose.t,b=sfm::mul(target,points[i])+sfm::Vec3{.1,.2,.3};
            r[0]=a.x/a.z-b.x/b.z;r[1]=a.y/a.z-b.y/b.z;
        };
        sfm::pose_detail::lmRefine((int)points.size(),rotation_residual,3,rotation,focal,20);
        if(rotation.t.x!=.1||rotation.t.y!=.2||rotation.t.z!=.3)throw std::runtime_error("rotation-only refinement changed translation");
        for(int i=0;i<9;++i)if(std::abs(rotation.R[i]-target[i])>1e-6)throw std::runtime_error("rotation-only refinement did not converge");
        const int w = 1024, h = 512;
        camhost::Camera camera; camera.width=w; camera.height=h;
        camera.fx=camera.fy=1000; camera.cx=w*.5; camera.cy=h*.5;
        std::vector<float> depth((size_t)w*h), opacity(depth.size(),1);
        spirula::roma::PairPrediction matches;
        for (auto* p : {&matches.forward,&matches.backward}) {
            p->width=w; p->height=h; p->warp.resize((size_t)w*h*2); p->overlap.assign((size_t)w*h,1);
        }
        for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;
            const float z=x<384?2:20, dx=20/z, dy=100/z;
            const double pixel[2]={(double)x,(double)y}; double ray[3];
            camhost::pixel_ray(camera,pixel,ray); depth[k]=(float)(z/ray[2]);
            matches.forward.warp[2*k]=2*(x+dx+.5f)/w-1;
            matches.forward.warp[2*k+1]=2*(y+dy+.5f)/h-1;
            const bool foreground=x>=10&&x<394&&y>=50;
            matches.backward.warp[2*k]=2*(x-(foreground?10:1)+.5f)/w-1;
            matches.backward.warp[2*k+1]=2*(y-(foreground?50:5)+.5f)/h-1;
        }
        A::Settings settings; settings.refine_radius=0;
        const auto alignment=A::refine_camera(depth,opacity,matches,settings,camera);
        std::cout << "pose: samples=" << alignment.samples << " held=" << alignment.holdout
                  << " before=" << alignment.error_before << " after=" << alignment.error_after << '\n';
        if (!alignment.accepted || alignment.status!=A::CameraFitStatus::Accepted || alignment.error_after>.05)
            throw std::runtime_error("depth-guided pose did not recover the camera translation");
        for(int x:{200,800}) {
            const double pixel[2]={(double)x,200};double ray[3],point[3],mapped[3],projected[2];
            camhost::pixel_ray(camera,pixel,ray);
            for(int c=0;c<3;++c)point[c]=depth[(size_t)200*w+x]*ray[c];
            alignment.correction.apply(point,mapped);
            const double dx=x<384?10:1,dy=x<384?50:5;
            if(!camhost::ray_pixel(camera,mapped,projected)||std::hypot(projected[0]-x-dx,projected[1]-200-dy)>.05)
                throw std::runtime_error("camera correction did not preserve depth-dependent parallax");
        }
        A::Image photo{w,h,std::vector<float>((size_t)w*h*3)},render=photo;
        auto signal=[](float x,float y){return .5f+.2f*std::sin(.04f*x+.07f*y)+.2f*std::cos(.08f*x-.03f*y);};
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;
            const float px=(matches.backward.warp[2*k]+1)*w*.5f-.5f,py=(matches.backward.warp[2*k+1]+1)*h*.5f-.5f;
            for(int c=0;c<3;++c){render.rgb[3*k+c]=signal((float)x,(float)y);photo.rgb[3*k+c]=signal(px,py);}
        }
        const auto image_alignment=A::refine_camera(depth,opacity,matches,settings,camera,{},&photo,&render);
        if(!image_alignment.photometric_candidate||image_alignment.photometric_error_after>.1)
            throw std::runtime_error("image-pyramid matching failed to propose the coherent camera correction");
        auto scaled_depth=depth;for(auto& z:scaled_depth)z*=1000;
        const auto scaled=A::refine_camera(scaled_depth,opacity,matches,settings,camera,{},&photo,&render);
        for(int x:{200,800}) {
            const double pixel[2]={(double)x,200};double ray[3],a[3],b[3],pa[2],pb[2];
            camhost::pixel_ray(camera,pixel,ray);
            for(int c=0;c<3;++c){a[c]=depth[(size_t)200*w+x]*ray[c];b[c]=1000*a[c];}
            for(bool fine:{false,true}) {
                const auto& first=fine?image_alignment.direct_polish.correction:image_alignment.direct_correction;
                const auto& second=fine?scaled.direct_polish.correction:scaled.direct_correction;
                double ma[3],mb[3];first.apply(a,ma);second.apply(b,mb);
                if(!camhost::ray_pixel(camera,ma,pa)||!camhost::ray_pixel(camera,mb,pb)||std::hypot(pa[0]-pb[0],pa[1]-pb[1])>.1)
                    throw std::runtime_error("direct camera refinement depends on world-coordinate units");
            }
        }
        settings.max_displacement=40;
        const auto limited=A::refine_camera(depth,opacity,matches,settings,camera);
        if (limited.samples>=alignment.samples)
            throw std::runtime_error("explicit displacement limit was ignored");
        if(A::refine_camera(depth,opacity,matches,settings,camera,std::vector<uint8_t>(depth.size(),0)).accepted)
            throw std::runtime_error("masked observations determined the camera pose");
        const auto missing=A::refine_camera(std::vector<float>(depth.size(),0),opacity,matches,settings,camera);
        if(missing.accepted||missing.status!=A::CameraFitStatus::InsufficientSupport)
            throw std::runtime_error("missing geometry determined the camera pose");
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;const float dx=(x/16+y/16)%2?.5f:-.5f;
            matches.forward.warp[2*k]=2*(x+dx+.5f)/w-1;
            matches.backward.warp[2*k]=2*(x-dx+.5f)/w-1;
            matches.forward.warp[2*k+1]=matches.backward.warp[2*k+1]=2*(y+.5f)/h-1;
        }
        const auto residual=A::refine_camera(depth,opacity,matches,settings,camera);
        if(residual.accepted||residual.status!=A::CameraFitStatus::NoImprovement||residual.samples<64||residual.holdout<10)
            throw std::runtime_error("local residuals were confused with absent matches or a validated pose change");
        std::cout << "PASS: pose correction, held-out geometry, large foreground parallax, distant background and explicit limit\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
