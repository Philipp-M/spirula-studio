#include "align/Warp.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace A = spirula::align;

int main() {
    try {
        const int w = 512, h = 384;
        for (bool diagonal : {false, true}) for (int period : {32,64,128}) {
            A::Image photo{w, h, std::vector<float>((size_t)w*h*3, .2f)}, render = photo;
            std::vector<float> depth((size_t)w*h, 10), opacity(depth.size(), 1);
            spirula::roma::PairPrediction matches;
            for (auto* p : {&matches.forward, &matches.backward}) {
                p->width = w; p->height = h;
                p->warp.resize((size_t)w*h*2); p->overlap.assign((size_t)w*h, 1);
            }
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                const size_t k = (size_t)y*w+x;
                const float dx = 3 + 2.5f*std::sin(6.2831853f*y/period);
                const int edge = diagonal ? y+32 : 96;
                for (int c = 0; c < 3; ++c) {
                    photo.rgb[3*k+c] = x >= edge && x < edge+24 ? .8f : .2f;
                    render.rgb[3*k+c] = x >= edge+3 && x < edge+27 ? .8f : .2f;
                }
                matches.backward.warp[2*k] = 2*(x+dx+.5f)/w-1;
                matches.forward.warp[2*k] = 2*(x-dx+.5f)/w-1;
                matches.backward.warp[2*k+1] = matches.forward.warp[2*k+1] = 2*(y+.5f)/h-1;
            }
            A::Settings settings; settings.refine_radius = 0;
            for (bool noisy_depth : {false, true}) {
                if (noisy_depth) for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
                    depth[(size_t)y*w+x] = (x/8)%7 == 3 && (y/8)%5 == 2 ? 2 : 10;
                const auto field = A::fit(photo, render, depth, opacity, matches, settings);
                if (field.fallback) throw std::runtime_error("noisy beam matches discarded the coherent translation");
                double mean = 0, ys = 0, yy = 0, yd = 0; int count = 0;
                std::vector<float> displacement;
                for (int y = 24; y < h-24; ++y) {
                    const float x = diagonal ? y+35 : 99;
                    float sx, sy;
                    if (!field.source(x, (float)y, sx, sy)) throw std::runtime_error("straight beam acquired an interior hole");
                    const float dx = x-sx-(diagonal ? y-sy : 0);
                    displacement.push_back(dx); mean += dx; ys += y; yy += y*y; yd += y*dx; ++count;
                }
                const double slope = (yd-ys*mean/count)/(yy-ys*ys/count);
                float lo = 1e6f, hi = -1e6f;
                for (int i = 0; i < count; ++i) {
                    const float error = (float)(displacement[i]-mean/count-slope*(i+24-ys/count));
                    lo = std::min(lo,error); hi = std::max(hi,error);
                }
                std::cout << "beam: diagonal=" << diagonal << " period=" << period << " noisy_depth=" << noisy_depth
                          << " peak-to-peak=" << hi-lo << " mean=" << mean/count << '\n';
                if (hi-lo > .5f) throw std::runtime_error("correspondence noise bent a straight beam");
                if (std::abs(mean/count-3) > .25) throw std::runtime_error("smoothing lost the beam translation");
                for (int y = 16; y < h-16; ++y) for (int x = 16; x < w-16; ++x) {
                    float sx, sy;
                    if (!field.source((float)x, (float)y, sx, sy)) throw std::runtime_error("isolated depth noise tore a continuous surface");
                }
            }
        }
        A::Image photo{w,h,std::vector<float>((size_t)w*h*3)},render=photo;
        std::vector<float> depth((size_t)w*h,10),opacity(depth.size(),1);
        spirula::roma::PairPrediction radial;
        for(auto* p:{&radial.forward,&radial.backward}) {
            p->width=w;p->height=h;p->warp.resize(depth.size()*2);p->overlap.assign(depth.size(),1);
        }
        auto distort=[&](float x,float y) {
            const float u=(x-w*.5f)/(w*.5f),v=(y-h*.5f)/(w*.5f),k=.04f*(u*u+v*v);
            return std::array<float,2>{x+k*(x-w*.5f),y+k*(y-h*.5f)};
        };
        auto signal=[](float x,float y) {
            const float edge=std::min({std::abs(x-80),std::abs(x-430),std::abs(y-60),std::abs(y-320)});
            return .2f+.6f*std::exp(-edge*edge/8);
        };
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t i=(size_t)y*w+x;const auto q=distort((float)x,(float)y);
            float px=(float)x,py=(float)y;
            for(int j=0;j<12;++j){const auto d=distort(px,py);px+=x-d[0];py+=y-d[1];}
            radial.forward.warp[2*i]=2*(q[0]+.5f)/w-1;radial.forward.warp[2*i+1]=2*(q[1]+.5f)/h-1;
            radial.backward.warp[2*i]=2*(px+.5f)/w-1;radial.backward.warp[2*i+1]=2*(py+.5f)/h-1;
            for(int c=0;c<3;++c){photo.rgb[3*i+c]=signal(px,py);render.rgb[3*i+c]=signal((float)x,(float)y);}
        }
        A::Settings settings;settings.refine_radius=0;
        const auto rectified=A::fit(photo,render,depth,opacity,radial,settings);
        double sum=0,normal_sum=0,initial_sum=0;float maximum=0,normal_max=0;int count=0;
        for(float x:{80.0f,430.0f})for(int y=30;y<h-30;++y) {
            float sx,sy;if(!rectified.source(x,(float)y,sx,sy))throw std::runtime_error("radial correction lost a structural edge");
            const auto q=distort(x,(float)y);const float e=std::hypot(sx-q[0],sy-q[1]);sum+=e;maximum=std::max(maximum,e);++count;
            const float normal=std::abs(sx-q[0]);normal_sum+=normal;normal_max=std::max(normal_max,normal);
            initial_sum+=std::abs(x-q[0]);
        }
        std::cout<<"radial correction: mean="<<sum/count<<" max="<<maximum<<" normal mean="<<normal_sum/count<<" normal max="<<normal_max<<'\n';
        if(rectified.fallback||normal_sum/count>.5||normal_max>1||normal_sum>.2*initial_sum)
            throw std::runtime_error("structure regularization suppressed smooth lens correction");
        std::cout << "PASS: straight beams, smooth radial correction and continuous coverage with noisy matches and depth\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
