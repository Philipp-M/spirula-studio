#include "align/WarpMath.h"
#include <stdexcept>

namespace spirula::align {
Audit audit_foreground(const Image& reference,const Image& before,const Image& after,
                       const std::vector<float>& depth,const std::vector<float>& opacity,
                       const std::vector<uint8_t>& mask) {
    const int w=reference.width,h=reference.height;const size_t n=(size_t)w*h;
    if(w<16||h<16||before.width!=w||before.height!=h||after.width!=w||after.height!=h||
       reference.rgb.size()!=n*3||before.rgb.size()!=n*3||after.rgb.size()!=n*3||depth.size()!=n||opacity.size()!=n||
       (!mask.empty()&&mask.size()!=n))throw std::runtime_error("invalid foreground audit dimensions");
    const auto r=detail::gray(reference),a=detail::gray(before),b=detail::gray(after);
    struct Patch {float before,after,range;};std::vector<Patch> patches;std::vector<float> ranges;
    const int step=std::max(8,std::max(w,h)/160);
    for(int y=7;y<h-7;y+=step)for(int x=7;x<w-7;x+=step) {
        const size_t k=(size_t)y*w+x;const float z=depth[k];
        if(!(z>0)||!std::isfinite(z)||!(opacity[k]>=.95f))continue;
        bool valid=true;
        for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i)
            valid&=mask.empty()||mask[(size_t)(y+j)*w+x+i]!=0;
        if(!valid)continue;
        const float old=detail::correlation(r,a,w,h,x,y,x,y,3);
        if(old<-1)continue;
        const float next=detail::correlation(r,b,w,h,x,y,x,y,3);
        patches.push_back({old,next<-1?old-.05f:next,z});ranges.push_back(z);
    }
    Audit result;if(ranges.empty())return result;
    auto mid=ranges.begin()+ranges.size()/2;std::nth_element(ranges.begin(),mid,ranges.end());
    double total=0;
    for(const auto& p:patches) {
        const double weight=std::clamp(std::pow(*mid/p.range,2),.25,64.);
        result.before+=weight*p.before;result.after+=weight*p.after;total+=weight;++result.count;
    }
    result.before/=total;result.after/=total;return result;
}
std::vector<std::array<float,2>> structure_points(const Image& source,const std::vector<uint8_t>& mask) {
    const size_t n=(size_t)source.width*source.height;
    if(source.width<16||source.height<16||source.rgb.size()!=n*3||(!mask.empty()&&mask.size()!=n))
        throw std::runtime_error("invalid structural support dimensions");
    Field grid;grid.width=source.width;grid.height=source.height;grid.spacing=8;
    grid.columns=(grid.width+7)/8+1;grid.rows=(grid.height+7)/8+1;
    const auto lines=detail::straight_lines(grid,detail::gray(source),mask);
    std::vector<std::array<float,2>> points;std::vector<uint8_t> used(n);
    for(const auto& line:lines)if(!line.relative) {
        const int x=(int)std::lround(line.center[0]),y=(int)std::lround(line.center[1]);
        if(x<8||y<8||x>=grid.width-8||y>=grid.height-8||used[(size_t)y*grid.width+x])continue;
        used[(size_t)y*grid.width+x]=1;points.push_back({(float)x,(float)y});
    }
    return points;
}

Audit audit_structure(const Image& reference,const Image& before,const Image& after,
                      const std::vector<std::array<float,2>>& points,const std::vector<uint8_t>& mask) {
    const int w=reference.width,h=reference.height;const size_t n=(size_t)w*h;
    if(w<16||h<16||before.width!=w||before.height!=h||after.width!=w||after.height!=h||
       reference.rgb.size()!=n*3||before.rgb.size()!=n*3||after.rgb.size()!=n*3||(!mask.empty()&&mask.size()!=n))
        throw std::runtime_error("invalid structural audit dimensions");
    const auto r=detail::gray(reference),a=detail::gray(before),b=detail::gray(after);Audit result;
    for(const auto& p:points) {
        if(!std::isfinite(p[0])||!std::isfinite(p[1])||p[0]<4||p[1]<4||p[0]>=w-4||p[1]>=h-4)continue;
        const int x=(int)p[0],y=(int)p[1];if((x/8+3*(y/8))%5!=0)continue;
        bool valid=true;for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i)valid&=mask.empty()||mask[(size_t)(y+j)*w+x+i]!=0;
        if(!valid)continue;
        const float old=detail::correlation(r,a,w,h,x,y,x,y,3);if(old<-1)continue;
        const float next=detail::correlation(r,b,w,h,x,y,x,y,3);
        result.before+=old;result.after+=next<-1?old-.05f:next;++result.count;
    }
    if(result.count){result.before/=result.count;result.after/=result.count;}return result;
}
bool accept_camera_micro(const CameraAudit& image,const Audit& foreground,const Audit& structure) {
    const double fine=image.fine_after-image.fine_before,coarse=image.coarse_after-image.coarse_before;
    const double close=foreground.after-foreground.before,lines=structure.after-structure.before;
    if(image.fine_count<50||image.coarse_count<50||foreground.count<32||close<-.001||
       (structure.count>=12&&lines<-.001)||!(close>.0005||(structure.count>=12&&lines>.0005)))return false;
    const bool detail_gain=fine>.001&&close>0&&(structure.count<12||lines>0)&&
        std::max(close,structure.count>=12?lines:0.)>.005;
    return fine>=-.001&&coarse>= (detail_gain?-.005:-.001);
}
}  // namespace spirula::align
