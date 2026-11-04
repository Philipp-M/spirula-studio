#include "align/WarpMath.h"
#include <iostream>
#include <stdexcept>

namespace A=spirula::align;
int main() {
    try {
        const int w=256,h=192;
        auto make_field=[&](float dx) {
            A::Field f;f.width=w;f.height=h;f.spacing=8;f.columns=w/8+1;f.rows=h/8+1;
            f.displacement.assign((size_t)f.columns*f.rows,{dx,0});f.support.assign(f.displacement.size(),1);
            f.depth.assign(f.displacement.size(),10);f.valid.assign((size_t)w*h,255);
            for(int y=0;y<h;y+=8)for(int x=0;x<w;x+=8) {
                const std::array<float,2> p[4]={{{x-.5f,y-.5f}},{{x+7.5f,y-.5f}},{{x+7.5f,y+7.5f}},{{x-.5f,y+7.5f}}};
                for(auto ids:{std::array<int,3>{0,1,2},std::array<int,3>{0,2,3}}) {
                    A::Triangle t;
                    for(int j=0;j<3;++j){t.source[j]=p[ids[j]];t.target[j]={p[ids[j]][0]+dx,p[ids[j]][1]};t.depth[j]=10;}
                    f.triangles.push_back(t);
                }
            }
            for(int y=16;y<h-16;y+=40)for(int x=16;x<w-16;x+=40)f.audit.push_back({(float)x,(float)y});
            A::index_field(f);return f;
        };
        A::Image source{w,h,std::vector<float>((size_t)w*h*3)},render=source;
        auto signal=[](float x,float y){return .5f+.2f*std::sin(.27f*x+.013f*y*y)+.17f*std::cos(.011f*x*x+.23f*y);};
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c) {
            source.rgb[((size_t)y*w+x)*3+c]=signal(x,y);
            render.rgb[((size_t)y*w+x)*3+c]=signal(x-1,y);
        }
        std::vector<float> depth((size_t)w*h,10),opacity(depth.size(),1);
        auto shifted=make_field(6);
        A::select_residual(shifted,source,render,depth,opacity,{},{});
        float sx,sy;
        if(shifted.fallback||!shifted.source(128,96,sx,sy)||std::abs(sx-127)>.6f)
            throw std::runtime_error("overcorrection was not reduced toward the measured shift");
        auto unsupported=make_field(0);unsupported.fallback=true;
        A::select_residual(unsupported,source,render,depth,opacity,{},{});
        if(unsupported.fallback||unsupported.residual_model!=3||!unsupported.source(128,96,sx,sy)||std::abs(sx-127)>.3f)
            throw std::runtime_error("direct image support did not recover a missing dense match correction");
        auto aligned=make_field(0);aligned.fallback=true;
        A::select_residual(aligned,source,source,depth,opacity,{},{});
        if(!aligned.fallback)throw std::runtime_error("selector invented correction for an aligned photo");
        auto masked=make_field(1);
        A::select_residual(masked,source,render,depth,opacity,std::vector<uint8_t>(depth.size(),0),{});
        if(!masked.fallback)throw std::runtime_error("masked evidence authorized a correction");
        std::cout<<"PASS: reduced overcorrection, direct recovery, aligned and masked inputs\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
