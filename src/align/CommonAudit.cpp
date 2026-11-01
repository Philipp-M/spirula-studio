#include "align/WarpMath.h"

#include <stdexcept>

namespace spirula::align {
std::vector<Audit> audit_common(const Image& photo,const std::vector<Image>& corrected,
                                const Image& render,const std::vector<std::vector<uint8_t>>& masks) {
    const int w=render.width,h=render.height;const size_t pixels=(size_t)w*h;
    if(w<16||h<16||photo.width!=w||photo.height!=h||photo.rgb.size()!=pixels*3||render.rgb.size()!=pixels*3||
       corrected.empty()||masks.size()!=corrected.size())throw std::runtime_error("invalid common-audit inputs");
    std::vector<std::vector<float>> gray;
    for(size_t i=0;i<corrected.size();++i) {
        if(corrected[i].width!=w||corrected[i].height!=h||corrected[i].rgb.size()!=pixels*3||masks[i].size()!=pixels)
            throw std::runtime_error("comparison images and masks differ in dimensions");
        gray.push_back(detail::gray(corrected[i]));
    }
    Field field;field.width=w;field.height=h;field.valid.assign(pixels,255);
    for(const auto& mask:masks)for(size_t i=0;i<pixels;++i)if(mask[i]!=255)field.valid[i]=0;
    const auto shared=field.valid;
    // This margin protects scoring from resize-filter footprints; it never changes exported masks.
    for(int y=2;y<h-2;++y)for(int x=2;x<w-2;++x)
        for(int j=-2;j<=2;++j)for(int i=-2;i<=2;++i)if(!shared[(size_t)(y+j)*w+x+i])field.valid[(size_t)y*w+x]=0;
    const auto pg=detail::gray(photo),rg=detail::gray(render);
    const float scale=std::max(w,h)/1280.0f;
    for(float y=16*scale;y<h-16*scale;y+=40*scale)for(float x=16*scale;x<w-16*scale;x+=40*scale) {
        const int px=(int)std::lround(x),py=(int)std::lround(y);if(px<4||py<4||px>=w-4||py>=h-4)continue;
        bool usable=detail::correlation(rg,pg,w,h,px,py,px,py,3)>-1;
        for(const auto& g:gray)usable&=detail::correlation(rg,g,w,h,px,py,px,py,3)>-1;
        field.audit.push_back({(float)px,(float)py});if(!usable)field.valid[(size_t)py*w+px]=0;
    }
    std::vector<Audit> scores;
    for(const auto& image:corrected)scores.push_back(audit(photo,image,render,field));
    for(const auto& score:scores)if(score.count!=scores.front().count||score.invalid!=scores.front().invalid)
        throw std::runtime_error("common audit scored different patches for different versions");
    return scores;
}
}  // namespace spirula::align
