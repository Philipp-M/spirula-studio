#include "align/WarpMath.h"

#include <iostream>
#include <stdexcept>

namespace A=spirula::align;

int main() {
    try {
        const int w=512,h=384;
        A::Field grid;grid.width=w;grid.height=h;grid.spacing=8;grid.columns=w/8+1;grid.rows=h/8+1;
        const size_t n=(size_t)grid.columns*grid.rows;
        grid.displacement.resize(n);grid.depth.assign(n,10);
        std::vector<std::array<float,2>> lens(n),noisy(n);
        std::vector<float> weights(n,1);
        std::vector<std::array<float,4>> structure(n);
        for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.columns;++x) {
            const size_t k=(size_t)y*grid.columns+x;
            grid.depth[k]=x<grid.columns/2?10.f:100.f;
            const float u=(std::min(x*8,w)-.5f-w*.5f)/(w*.5f);
            const float v=(std::min(y*8,h)-.5f-h*.5f)/(w*.5f),r=u*u+v*v;
            lens[k]={1+2*u-.5f*v+4*u*r-1.5f*u*r*r+.3f*2*u*v-.2f*(r+2*u*u),
                     -2+.5f*u+v+4*v*r-1.5f*v*r*r+.3f*(r+2*v*v)-.2f*2*u*v};
            noisy[k]=lens[k];
            if((x+3*y)%19==0){noisy[k][0]+=20;noisy[k][1]-=15;}
        }
        const auto recovered=A::detail::smooth_residual(grid,noisy,weights,structure);
        double worst=0;
        for(size_t k=0;k<n;++k)worst=std::max(worst,(double)std::hypot(recovered[k][0]-lens[k][0],recovered[k][1]-lens[k][1]));
        std::cout<<"smooth lens with outliers: maximum error="<<worst<<'\n';
        if(worst>.3)throw std::runtime_error("coherent lens field followed local outliers");
        auto scaled=grid;for(auto& z:scaled.depth)z*=1000;
        const auto rescaled=A::detail::smooth_residual(scaled,noisy,weights,structure);
        for(size_t k=0;k<n;++k)for(int c=0;c<2;++c)
            if(std::abs(rescaled[k][c]-recovered[k][c])>1e-4f)throw std::runtime_error("residual prior depends on world units");

        A::Image photo{w,h,std::vector<float>((size_t)w*h*3)},render=photo;
        std::vector<float> depth((size_t)w*h,10),opacity(depth.size(),1);
        spirula::roma::PairPrediction matches;
        for(auto* p:{&matches.forward,&matches.backward}) {
            p->width=w;p->height=h;p->overlap.assign(depth.size(),1);p->warp.resize(depth.size()*2);
        }
        auto signal=[](float x,float y){return .5f+.2f*std::sin(.73f*x+.019f*y*y)+.17f*std::cos(.021f*x*x+.47f*y);};
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;
            const float dx=2.5f*std::sin(y*6.2831853f/48);
            for(int c=0;c<3;++c){photo.rgb[3*k+c]=signal(x,y);render.rgb[3*k+c]=signal(x-dx,y);}
            matches.forward.warp[2*k]=2*(x-dx+.5f)/w-1;
            matches.backward.warp[2*k]=2*(x+dx+.5f)/w-1;
            matches.forward.warp[2*k+1]=matches.backward.warp[2*k+1]=2*(y+.5f)/h-1;
        }
        A::Settings settings;settings.refine_radius=0;settings.camera_residual=true;
        for(bool edges:{false,true}) {
            const auto field=A::fit(photo,render,depth,opacity,matches,settings,{},edges);
            float maximum=0;
            for(int y=144;y<=240;++y) {
                float sx,sy;
                if(!field.source(256,(float)y,sx,sy))throw std::runtime_error("bounded residual lost interior coverage");
                maximum=std::max(maximum,std::hypot(sx-256,sy-y));
            }
            std::cout<<"interior contradictory warp: structure="<<edges<<" max="<<maximum<<'\n';
            if(field.fallback||maximum>.7f)throw std::runtime_error("local interior warp escaped camera-residual policy");
        }
        std::cout<<"PASS: smooth lens recovery, scale invariance and bounded interior correction including fallback fitter\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
