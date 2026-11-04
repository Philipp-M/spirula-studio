#pragma once

#include "align/Warp.h"

#include <algorithm>
#include <cmath>

namespace spirula::align::detail {
inline float displacement_limit(const Settings& settings, int width, int height) {
    return settings.max_displacement == 0 ? .05f*std::hypot((float)width,(float)height) : settings.max_displacement;
}

std::vector<float> depth_support(const Field& field, const std::vector<float>& weights);
std::vector<std::array<float,2>> smooth_residual(const Field& field,
    const std::vector<std::array<float,2>>& data,const std::vector<float>& weights,
    const std::vector<std::array<float,4>>& structure);
void limit_residual(Field& field,const std::vector<std::array<float,2>>& baseline);
struct LineConstraint {
    std::array<float,2> center{};
    std::array<size_t,12> nodes{};
    std::array<float,12> coefficients{};
    std::array<float,2> normal{};
    int count = 0;
    float allowance = .25f;
    bool relative = false;
};
std::vector<LineConstraint> straight_lines(const Field& field,const std::vector<float>& photo,
                                         const std::vector<uint8_t>& mask);
// Entries hold unit normal x/y, normal displacement in working pixels, and weight.
std::vector<std::array<float,4>> structure_matches(const Field& field,const std::vector<float>& photo,
    const std::vector<float>& render,const std::vector<float>& depth,const std::vector<float>& opacity,
    const roma::PairPrediction& matches,const Settings& settings,const std::vector<uint8_t>& mask);
void regularize(Field& field, const std::vector<std::array<float,2>>& data,
                const std::vector<float>& weights, const std::vector<float>& support,
                const std::vector<float>& photo, const std::vector<uint8_t>& mask,
                const std::vector<std::array<float,4>>& structure,bool preserve_lines = true);

inline bool sample(const std::vector<float>& v, int w, int h, int c, float x, float y, float* out) {
    if (!std::isfinite(x) || !std::isfinite(y) || x<0 || y<0 || x>w-1 || y>h-1 ||
        v.size()!=(size_t)w*h*c) return false;
    const int x0=(int)x,y0=(int)y,x1=std::min(x0+1,w-1),y1=std::min(y0+1,h-1);
    const float a=x-x0,b=y-y0;
    for(int k=0;k<c;++k)out[k]=(1-b)*((1-a)*v[((size_t)y0*w+x0)*c+k]+a*v[((size_t)y0*w+x1)*c+k])+
        b*((1-a)*v[((size_t)y1*w+x0)*c+k]+a*v[((size_t)y1*w+x1)*c+k]);
    return true;
}
inline std::vector<float> gray(const Image& a) {
    std::vector<float> g((size_t)a.width*a.height);
    for(size_t i=0;i<g.size();++i)g[i]=.2126f*a.rgb[3*i]+.7152f*a.rgb[3*i+1]+.0722f*a.rgb[3*i+2];
    return g;
}
inline float correlation(const std::vector<float>& a,const std::vector<float>& b,int w,int h,
                         float x,float y,float qx,float qy,int r) {
    double sa=0,sb=0,aa=0,bb=0,ab=0;
    for(int j=-r;j<=r;++j)for(int i=-r;i<=r;++i) {
        float u,v;
        if(!sample(a,w,h,1,x+i,y+j,&u)||!sample(b,w,h,1,qx+i,qy+j,&v))return -2;
        sa+=u;sb+=v;aa+=u*u;bb+=v*v;ab+=u*v;
    }
    const double n=(2*r+1)*(2*r+1),va=aa-sa*sa/n,vb=bb-sb*sb/n;
    if(va/n<.0001||vb/n<.0001)return -2;
    return (float)((ab-sa*sb/n)/std::sqrt(std::max(1e-20,va*vb)));
}
inline bool match(const roma::Prediction& p,float x,float y,int w,int h,float& qx,float& qy,float& confidence) {
    float uv[2];
    const float gx=std::clamp((x+.5f)*p.width/w-.5f,0.0f,(float)p.width-1);
    const float gy=std::clamp((y+.5f)*p.height/h-.5f,0.0f,(float)p.height-1);
    if(!sample(p.warp,p.width,p.height,2,gx,gy,uv)||!sample(p.overlap,p.width,p.height,1,gx,gy,&confidence))return false;
    qx=(uv[0]+1)*w*.5f-.5f;qy=(uv[1]+1)*h*.5f-.5f;
    return std::isfinite(qx)&&std::isfinite(qy)&&std::isfinite(confidence);
}
inline float cross(const std::array<float,2>& a,const std::array<float,2>& b,const std::array<float,2>& c) {
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
}
inline bool barycentric(const Triangle& t,float x,float y,std::array<float,3>& b) {
    const std::array<float,2> p{x,y};const float det=cross(t.target[0],t.target[1],t.target[2]);
    if(!(det>1e-8f))return false;
    b[0]=cross(t.target[1],t.target[2],p)/det;b[1]=cross(t.target[2],t.target[0],p)/det;b[2]=1-b[0]-b[1];
    return b[0]>=-1e-5f&&b[1]>=-1e-5f&&b[2]>=-1e-5f;
}
inline float triangle_depth(const Triangle& t,const std::array<float,3>& b) {
    float inverse=0;
    for(int i=0;i<3;++i)if(t.depth[i]>0&&std::isfinite(t.depth[i]))inverse+=std::max(0.0f,b[i])/t.depth[i];
    return inverse>0?1/inverse:1e30f;
}
}  // namespace spirula::align::detail
