#include "align/Warp.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace A = spirula::align;

int main() {
    try {
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
        if (!alignment.accepted || alignment.error_after>.05) throw std::runtime_error("depth-guided pose did not recover the camera translation");
        for(int x:{200,800}) {
            const double pixel[2]={(double)x,200};double ray[3],point[3],mapped[3],projected[2];
            camhost::pixel_ray(camera,pixel,ray);
            for(int c=0;c<3;++c)point[c]=depth[(size_t)200*w+x]*ray[c];
            alignment.correction.apply(point,mapped);
            const double dx=x<384?10:1,dy=x<384?50:5;
            if(!camhost::ray_pixel(camera,mapped,projected)||std::hypot(projected[0]-x-dx,projected[1]-200-dy)>.05)
                throw std::runtime_error("camera correction did not preserve depth-dependent parallax");
        }
        settings.max_displacement=40;
        const auto limited=A::refine_camera(depth,opacity,matches,settings,camera);
        if (limited.samples>=alignment.samples)
            throw std::runtime_error("explicit displacement limit was ignored");
        if(A::refine_camera(depth,opacity,matches,settings,camera,std::vector<uint8_t>(depth.size(),0)).accepted)
            throw std::runtime_error("masked observations determined the camera pose");
        if(A::refine_camera(std::vector<float>(depth.size(),0),opacity,matches,settings,camera).accepted)
            throw std::runtime_error("missing geometry determined the camera pose");
        std::cout << "PASS: pose correction, held-out geometry, large foreground parallax, distant background and explicit limit\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
