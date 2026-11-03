#include "align/Warp.h"
#include "align/WarpMath.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace A = spirula::align;

void thin_foreground() {
    const int w=256,h=192;
    A::Image photo{w,h,std::vector<float>((size_t)w*h*3)},render=photo;
    std::vector<float> depth((size_t)w*h,100),opacity(depth.size(),1);
    spirula::roma::PairPrediction matches;
    for(auto* p:{&matches.forward,&matches.backward}) {
        p->width=w;p->height=h;p->warp.resize(depth.size()*2);p->overlap.assign(depth.size(),.1f);
    }
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const size_t k=(size_t)y*w+x;
        const float texture=.06f*std::sin(.73f*x+.51f*y)+.04f*std::cos(.49f*x-.81f*y);
        for(int c=0;c<3;++c) {
            photo.rgb[3*k+c]=.2f+texture+.6f*std::exp(-std::pow((x-96.f)/1.5f,2));
            render.rgb[3*k+c]=.2f+.6f*std::exp(-std::pow((x-90.f)/1.5f,2));
        }
        if(std::abs(x-90)<5)depth[k]=10;
        for(auto* p:{&matches.forward,&matches.backward}) {
            p->warp[2*k]=2*(x+.5f)/w-1;p->warp[2*k+1]=2*(y+.5f)/h-1;
        }
    }
    A::Field grid;grid.width=w;grid.height=h;grid.spacing=8;grid.columns=w/8+1;grid.rows=h/8+1;
    grid.displacement.resize((size_t)grid.columns*grid.rows);
    const auto pg=A::detail::gray(photo),rg=A::detail::gray(render);
    auto anchors=A::detail::structure_matches(grid,pg,rg,depth,opacity,matches,{},{});
    int count=0;double error=0;
    for(int y=4;y<grid.rows-4;++y) {
        const auto& a=anchors[(size_t)y*grid.columns+12];
        if(a[3]<=0)continue;
        error+=std::abs(a[2]+6*a[0]);++count;
        if(std::abs(a[1])>.1f)throw std::runtime_error("thin edge constrained its ambiguous tangential motion");
    }
    std::cout<<"thin foreground normal anchors: "<<count<<" error="<<error/std::max(1,count)<<'\n';
    if(count<8||error/count>.5)throw std::runtime_error("thin foreground edge did not recover its missing correspondence");
    std::fill(opacity.begin(),opacity.end(),.6f);
    anchors=A::detail::structure_matches(grid,pg,rg,depth,opacity,matches,{},{});
    count=0;for(const auto& a:anchors)count+=a[3]>0;
    if(count<8)throw std::runtime_error("partially covered foreground edges lost their normal matches");
    std::fill(opacity.begin(),opacity.end(),.4f);
    anchors=A::detail::structure_matches(grid,pg,rg,depth,opacity,matches,{},{});
    for(const auto& a:anchors)if(a[3]>0)throw std::runtime_error("transparent geometry supplied a foreground constraint");
    std::fill(opacity.begin(),opacity.end(),1.f);
    auto incomplete=render;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const size_t k=(size_t)y*w+x;
        if(y<64)for(int c=0;c<3;++c)incomplete.rgb[3*k+c]=.2f+.005f*std::exp(-std::pow((x-90.f)/1.5f,2));
        for(auto* p:{&matches.forward,&matches.backward})p->overlap[k]=std::abs(x-96)<24?.1f:1.f;
    }
    const auto extended=A::fit(photo,incomplete,depth,opacity,matches,{});
    float worst=0;
    for(int y:{24,32,48,80,120}) {
        float sx,sy;
        if(!extended.source(90,(float)y,sx,sy))throw std::runtime_error("weakly rendered rod acquired a coverage hole");
        worst=std::max(worst,std::abs(sx-96));
    }
    std::cout<<"unsupported thin tip: normal error="<<worst<<'\n';
    if(extended.fallback||worst>1.5f)throw std::runtime_error("foreground correction did not reach its weakly rendered tip");
    for(auto& z:depth)if(z==100)z=0;
    anchors=A::detail::structure_matches(grid,pg,rg,depth,opacity,matches,{},{});
    count=0;for(const auto& a:anchors)count+=a[3]>0;
    if(count<8)throw std::runtime_error("missing background depth disabled a measured foreground edge");
    std::fill(depth.begin(),depth.end(),100);
    anchors=A::detail::structure_matches(grid,pg,rg,depth,opacity,matches,{},{});
    for(const auto& a:anchors)if(a[3]>0)throw std::runtime_error("flat depth acquired foreground constraints");
    for(int y=0;y<h;++y)for(int x=86;x<95;++x)depth[(size_t)y*w+x]=10;
    std::vector<uint8_t> mask(depth.size(),255);
    for(int y=0;y<h;++y)for(int x=76;x<116;++x)mask[(size_t)y*w+x]=0;
    anchors=A::detail::structure_matches(grid,pg,rg,depth,opacity,matches,{},mask);
    for(const auto& a:anchors)if(a[3]>0)throw std::runtime_error("masked photograph contributed a structure constraint");
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c) {
        photo.rgb[3*((size_t)y*w+x)+c]=.5f+.4f*std::cos(x*6.2831853f/12);
        render.rgb[3*((size_t)y*w+x)+c]=.5f+.4f*std::cos((x+6)*6.2831853f/12);
    }
    anchors=A::detail::structure_matches(grid,A::detail::gray(photo),A::detail::gray(render),depth,opacity,matches,{},{});
    for(int y=4;y<grid.rows-4;++y)if(anchors[(size_t)y*grid.columns+12][3]>0)
        throw std::runtime_error("repeated edges supplied an ambiguous normal correspondence");
    const size_t n=grid.displacement.size();
    grid.depth.assign(n,10);grid.displacement.assign(n,{-3.2f,2.4f});
    const auto data=grid.displacement;
    std::vector<float> weights(n,.01f),support(n,1),flat((size_t)w*h,.5f);
    anchors.assign(n,{.6f,.8f,6,16});
    A::detail::regularize(grid,data,weights,support,flat,{},anchors);
    for(const auto& v:grid.displacement) {
        if(std::abs(.6f*v[0]+.8f*v[1]-6)>.02f||std::abs(-.8f*v[0]+.6f*v[1]-4)>.01f)
            throw std::runtime_error("normal constraint changed the unconstrained tangential movement");
    }
}

void oblique_photo_line() {
    const int w=256,h=192;
    A::Field f;f.width=w;f.height=h;f.spacing=8;f.columns=w/8+1;f.rows=h/8+1;
    const size_t n=(size_t)f.columns*f.rows;
    f.displacement.resize(n);f.depth.assign(n,20);
    std::vector<float> photo((size_t)w*h),weights(n,1),support(n,1);
    std::vector<std::array<float,2>> data(n);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)
        photo[(size_t)y*w+x]=.2f+.7f*std::exp(-std::pow((y-60-.35f*x)/2.f,2));
    for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x)
        data[(size_t)y*f.columns+x]={0,2*std::sin(x*8.f*.04908739f)};
    const auto lines=A::detail::straight_lines(f,photo,{});
    if(lines.size()<8)throw std::runtime_error("oblique photographic line was not detected");
    if(!A::detail::straight_lines(f,photo,std::vector<uint8_t>(photo.size(),0)).empty())
        throw std::runtime_error("masked photograph supplied a line constraint");
    A::detail::regularize(f,data,weights,support,photo,{},std::vector<std::array<float,4>>(n));
    std::vector<float> vertical(n);for(size_t k=0;k<n;++k)vertical[k]=f.displacement[k][1];
    float low=1e9,high=-1e9;
    for(int x=64;x<=192;x+=4) {
        float displacement=0;
        A::detail::sample(vertical,f.columns,f.rows,1,(x+.5f)/8,(60+.35f*x+.5f)/8,&displacement);
        low=std::min(low,displacement);high=std::max(high,displacement);
    }
    std::cout<<"oblique source line under contradictory matches: peak-to-peak="<<high-low<<'\n';
    if(high-low>.7f)throw std::runtime_error("reference mismatch bent a straight photographic line");
}

void short_weak_tip() {
    const int w=256,h=192;
    A::Field f;f.width=w;f.height=h;f.spacing=8;f.columns=w/8+1;f.rows=h/8+1;
    const size_t n=(size_t)f.columns*f.rows;
    f.displacement.resize(n);f.depth.assign(n,20);
    std::vector<float> photo((size_t)w*h,.2f),weights(n,1),support(n,1);
    std::vector<std::array<float,2>> data(n);
    std::vector<std::array<float,4>> anchors(n);
    for(int y=80;y<148;++y)for(int x=0;x<w;++x)
        photo[(size_t)y*w+x]+=(y<112?.18f:.6f)*std::exp(-std::pow((x-128.f)/(y<112?1.5f:3.f),2));
    for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x) {
        const size_t k=(size_t)y*f.columns+x;
        data[k]={(float)(std::clamp(.16f*(112-y*8),0.f,6.f)*std::exp(-std::pow((x*8-128.f)/32,2))),0};
        if(x==16&&y*8>=112&&y*8<148)anchors[k]={1,0,0,16};
    }
    A::detail::regularize(f,data,weights,support,photo,{},anchors);
    std::vector<float> horizontal(n);for(size_t k=0;k<n;++k)horizontal[k]=f.displacement[k][0];
    float top=0,base=0;
    A::detail::sample(horizontal,f.columns,f.rows,1,128.5f/8,84.5f/8,&top);
    A::detail::sample(horizontal,f.columns,f.rows,1,128.5f/8,108.5f/8,&base);
    std::cout<<"short weak tip: shear="<<std::abs(top-base)<<'\n';
    if(std::abs(top-base)>1.5f)throw std::runtime_error("weak short tip followed the background shear");
    for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x)data[(size_t)y*f.columns+x]={.1f*y*8,.05f*x*8};
    f.displacement=data;
    A::detail::regularize(f,data,weights,support,photo,{},std::vector<std::array<float,4>>(n));
    for(size_t k=0;k<n;++k)for(int c=0;c<2;++c)if(std::abs(f.displacement[k][c]-data[k][c])>1e-4f)
        throw std::runtime_error("short-line protection suppressed a coherent affine correction");
}

void local_line_rotation() {
    const int w=512,h=384;
    A::Field f;f.width=w;f.height=h;f.spacing=8;f.columns=w/8+1;f.rows=h/8+1;
    const size_t n=(size_t)f.columns*f.rows;
    f.displacement.resize(n);f.depth.assign(n,10);
    std::vector<float> photo((size_t)w*h),weights(n,1),support(n,1);
    std::vector<std::array<float,2>> data(n);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)
        photo[(size_t)y*w+x]=.2f+.6f*std::exp(-std::pow((x-256.f)/3,2.f));
    for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x)
        data[(size_t)y*f.columns+x]={.06f*(y*8-192)*std::exp(-std::pow((x*8-256.f)/48,2.f)),0};
    A::detail::regularize(f,data,weights,support,photo,{},std::vector<std::array<float,4>>(n));
    const float shear=std::abs(f.displacement[32*f.columns+32][0]-f.displacement[16*f.columns+32][0]);
    std::cout<<"central straight rod: local shear="<<shear<<'\n';
    if(shear>1.5f)throw std::runtime_error("local matches rotated a central photographic structure");
}

int main() {
    try {
        local_line_rotation();
        short_weak_tip();
        oblique_photo_line();
        thin_foreground();
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
        for(bool foreground:{false,true}) for(bool bounded:{false,true}) {
        settings.camera_residual=bounded;
        if(foreground)for(int y=0;y<h;++y)for(int x=0;x<w;++x)
            depth[(size_t)y*w+x]=signal((float)x,(float)y)>.21f?10:100;
        const auto rectified=A::fit(photo,render,depth,opacity,radial,settings);
        double sum=0,normal_sum=0,initial_sum=0;float maximum=0,normal_max=0;int count=0;
        for(float x:{80.0f,430.0f})for(int y=30;y<h-30;++y) {
            float sx,sy;if(!rectified.source(x,(float)y,sx,sy))throw std::runtime_error("radial correction lost a structural edge");
            const auto q=distort(x,(float)y);const float e=std::hypot(sx-q[0],sy-q[1]);sum+=e;maximum=std::max(maximum,e);++count;
            const float normal=std::abs(sx-q[0]);normal_sum+=normal;normal_max=std::max(normal_max,normal);
            initial_sum+=std::abs(x-q[0]);
        }
        std::cout<<"radial correction: foreground="<<foreground<<" bounded="<<bounded<<" mean="<<sum/count<<" max="<<maximum<<" normal mean="<<normal_sum/count<<" normal max="<<normal_max<<'\n';
        if(rectified.fallback||normal_sum/count>.5||normal_max>1||normal_sum>.2*initial_sum)
            throw std::runtime_error("structure regularization suppressed smooth lens correction");
        }
        std::cout << "PASS: straight beams, smooth radial correction and continuous coverage with noisy matches and depth\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
