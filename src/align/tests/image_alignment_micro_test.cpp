#include "align/Warp.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace A=spirula::align;
int main() {
    try {
        const int w=512,h=384;
        camhost::Camera camera;camera.width=w;camera.height=h;camera.fx=camera.fy=500;camera.cx=w*.5;camera.cy=h*.5;
        A::Image photo{w,h,std::vector<float>((size_t)w*h*3)},render=photo;
        std::vector<float> depth((size_t)w*h),opacity(depth.size(),1);
        auto texture=[](double x,double y){return .5+.16*std::sin(.21*x+.11*y)+.16*std::cos(.13*x-.19*y)+.1*std::sin(.08*x+.31*y);};
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;const double z=x>200&&x<300?6:100;
            const double pixel[2]={(double)x,(double)y};double ray[3];camhost::pixel_ray(camera,pixel,ray);depth[k]=z/ray[2];
            for(int c=0;c<3;++c){render.rgb[3*k+c]=texture(x,y);photo.rgb[3*k+c]=texture(x-4/z,y+3/z);}
        }
        const auto fit=A::polish_camera(photo,render,depth,opacity,camera);
        if(!fit.candidate)throw std::runtime_error("subpixel pose was not proposed");
        auto scaled=depth;for(auto& z:scaled)z*=1000;
        const auto scaled_fit=A::polish_camera(photo,render,scaled,opacity,camera);
        if(!scaled_fit.candidate)throw std::runtime_error("world scaling changed proposal acceptance");
        for(int y:{80,300})for(int x:{100,250,400}) {
            const double pixel[2]={(double)x,(double)y};double ray[3],p[3],q[3],uv[2];camhost::pixel_ray(camera,pixel,ray);
            for(int c=0;c<3;++c)p[c]=ray[c]*depth[(size_t)y*w+x];fit.correction.apply(p,q);camhost::ray_pixel(camera,q,uv);
            const double z=x==250?6:100,error=std::hypot(uv[0]-x-4/z,uv[1]-y+3/z);
            std::cout<<"micro error "<<error<<'\n';if(error>.15)throw std::runtime_error("rigid micro fit lost foreground parallax");
            for(int c=0;c<3;++c)p[c]*=1000;scaled_fit.correction.apply(p,q);double scaled_uv[2];camhost::ray_pixel(camera,q,scaled_uv);
            std::cout<<"scale delta "<<std::hypot(uv[0]-scaled_uv[0],uv[1]-scaled_uv[1])<<'\n';
            if(std::hypot(uv[0]-scaled_uv[0],uv[1]-scaled_uv[1])>.02)throw std::runtime_error("micro fit depends on world units");
        }
        if(A::polish_camera(render,render,depth,opacity,camera).candidate)throw std::runtime_error("aligned image produced a micro correction");
        if(A::polish_camera(photo,render,depth,opacity,camera,std::vector<uint8_t>(depth.size(),0)).candidate)
            throw std::runtime_error("masked pixels determined camera pose");
        auto before=render,after=render;
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c) {
            const bool foreground=x>200&&x<300;const size_t k=((size_t)y*w+x)*3+c;
            before.rgb[k]=texture(x+(foreground?0:.3),y);after.rgb[k]=texture(x+(foreground?.5:0),y);
        }
        const auto audit=A::audit_foreground(render,before,after,depth,opacity);
        std::cout<<"foreground NCC "<<audit.before<<" -> "<<audit.after<<'\n';
        if(audit.count<32||audit.after>=audit.before-.001)throw std::runtime_error("background improvement concealed foreground regression");
        auto lines=render,shifted=render;
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            double a=.2,b=.2;
            for(int edge:{104,240,376}) {
                a+=.6*std::exp(-std::pow((x-edge)/2.,2));b+=.6*std::exp(-std::pow((x-edge-.8)/2.,2));
            }
            for(int c=0;c<3;++c){lines.rgb[((size_t)y*w+x)*3+c]=a;shifted.rgb[((size_t)y*w+x)*3+c]=b;}
        }
        const auto points=A::structure_points(lines);
        const auto structural=A::audit_structure(lines,lines,shifted,points);
        std::cout<<"structural NCC "<<structural.before<<" -> "<<structural.after<<" count "<<structural.count<<'\n';
        if(structural.count<12||structural.after>=structural.before-.001)throw std::runtime_error("source-line regression was not detected");
        if(!A::structure_points(lines,std::vector<uint8_t>(depth.size(),0)).empty())throw std::runtime_error("masked lines entered validation");
        A::CameraAudit camera_audit;camera_audit.fine_count=camera_audit.coarse_count=100;
        camera_audit.fine_before=camera_audit.coarse_before=.5;
        camera_audit.fine_after=.51;camera_audit.coarse_after=.496;
        A::Audit foreground_audit,structure_audit;foreground_audit.count=100;structure_audit.count=20;
        foreground_audit.before=structure_audit.before=.5;foreground_audit.after=structure_audit.after=.52;
        if(!A::accept_camera_micro(camera_audit,foreground_audit,structure_audit))
            throw std::runtime_error("small low-pass loss vetoed consistent fine and foreground gains");
        camera_audit.coarse_after=.494;
        if(A::accept_camera_micro(camera_audit,foreground_audit,structure_audit))throw std::runtime_error("excessive background loss accepted");
        camera_audit.coarse_after=.496;structure_audit.after=.498;
        if(A::accept_camera_micro(camera_audit,foreground_audit,structure_audit))throw std::runtime_error("structural regression accepted");
        structure_audit.after=.52;foreground_audit.after=.498;
        if(A::accept_camera_micro(camera_audit,foreground_audit,structure_audit))throw std::runtime_error("foreground regression accepted");
        foreground_audit.after=.52;camera_audit.fine_after=.4995;
        if(A::accept_camera_micro(camera_audit,foreground_audit,structure_audit))throw std::runtime_error("low-pass relaxation lacked fine improvement");
        auto bad=depth;bad.pop_back();bool rejected=false;
        try{A::audit_foreground(render,before,after,bad,opacity);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid audit geometry accepted");
        std::cout<<"PASS: micro pose, scale invariance, aligned and masked inputs, foreground regression\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
