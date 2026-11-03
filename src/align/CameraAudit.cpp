#include "align/WarpMath.h"
#include "core/LanczosResize.h"
#include <stdexcept>

namespace spirula::align {
MatchAudit audit_camera_structure(const Image& photo,const Image& before,const Image& after,
    const std::vector<float>& before_depth,const std::vector<float>& before_opacity,
    const std::vector<float>& after_depth,const std::vector<float>& after_opacity,
    const roma::PairPrediction& matches,const Settings& settings,const std::vector<uint8_t>& mask) {
    const size_t pixels=(size_t)photo.width*photo.height;
    if(photo.width<16||photo.height<16||settings.spacing<4||before.width!=photo.width||after.width!=photo.width||
       before.height!=photo.height||after.height!=photo.height||photo.rgb.size()!=pixels*3||before.rgb.size()!=pixels*3||after.rgb.size()!=pixels*3||
       before_depth.size()!=pixels||after_depth.size()!=pixels||before_opacity.size()!=pixels||after_opacity.size()!=pixels||(!mask.empty()&&mask.size()!=pixels))
        throw std::runtime_error("invalid camera structure audit dimensions");
    Field grid;grid.width=photo.width;grid.height=photo.height;grid.spacing=settings.spacing;
    grid.columns=(grid.width+grid.spacing-1)/grid.spacing+1;grid.rows=(grid.height+grid.spacing-1)/grid.spacing+1;
    grid.displacement.resize((size_t)grid.columns*grid.rows);
    const auto gray=detail::gray(photo);
    const auto a=detail::structure_matches(grid,gray,detail::gray(before),before_depth,before_opacity,matches,settings,mask);
    const auto b=detail::structure_matches(grid,gray,detail::gray(after),after_depth,after_opacity,matches,settings,mask);
    MatchAudit result;
    for(int y=0;y<grid.rows;++y)for(int x=0;x<grid.columns;++x)if((x+3*y)%7==3) {
        const size_t k=(size_t)y*grid.columns+x;const bool va=a[k][3]>0,vb=b[k][3]>0;
        result.before_count+=va;result.after_count+=vb;
        if(va){result.supported_before+=std::abs(a[k][2]);result.supported_after+=vb?std::abs(b[k][2]):std::abs(a[k][2]);}
        if(va&&vb){++result.common_count;result.before+=std::abs(a[k][2]);result.after+=std::abs(b[k][2]);}
    }
    if(result.common_count){result.before/=result.common_count;result.after/=result.common_count;}
    if(result.before_count){result.supported_before/=result.before_count;result.supported_after/=result.before_count;}
    result.accepted=result.common_count>=12&&result.common_count>=.7*result.before_count&&
        result.after_count>=.8*result.before_count&&result.after<.8*result.before&&result.before-result.after>.25&&
        result.supported_after<.8*result.supported_before;
    return result;
}

MatchAudit audit_camera_matches(const roma::PairPrediction& before,const roma::PairPrediction& after,
                                int width,int height,const Settings& settings,const std::vector<uint8_t>& mask) {
    if(width<16||height<16||(!mask.empty()&&mask.size()!=(size_t)width*height))
        throw std::runtime_error("invalid camera matching audit dimensions");
    MatchAudit result;
    auto displacement=[&](const roma::PairPrediction& matches,int x,int y,double& error) {
        float rx,ry,c,px,py,bc;
        if(!detail::match(matches.backward,x,y,width,height,rx,ry,c)||c<settings.min_overlap||
           rx<0||ry<0||rx>width-1||ry>height-1||
           !detail::match(matches.forward,rx,ry,width,height,px,py,bc)||bc<settings.min_overlap||
           std::hypot(px-x,py-y)>settings.max_cycle)return false;
        error=std::min(20.0,(double)std::hypot(rx-x,ry-y));return true;
    };
    const int step=std::max(8,std::max(width,height)/160);
    for(int y=step/2;y<height;y+=step)for(int x=step/2;x<width;x+=step) {
        if(!mask.empty()&&!mask[(size_t)y*width+x])continue;
        double a=0,b=0;const bool va=displacement(before,x,y,a),vb=displacement(after,x,y,b);
        result.before_count+=va;result.after_count+=vb;
        // Missing matches contribute no improvement; changing visibility is not a measured 20-pixel error.
        if(va){result.supported_before+=a;result.supported_after+=vb?b:a;}
        if(va&&vb){++result.common_count;result.before+=a;result.after+=b;}
    }
    if(result.common_count){result.before/=result.common_count;result.after/=result.common_count;}
    if(result.before_count){result.supported_before/=result.before_count;result.supported_after/=result.before_count;}
    result.accepted=result.common_count>=128&&result.common_count>=.5*result.before_count&&
        result.after_count>=.8*result.before_count&&result.after<.9*result.before&&result.before-result.after>.1&&
        result.supported_after<.9*result.supported_before;
    return result;
}

CameraAudit audit_camera(const Image& photo,const Image& before,const Image& after,const std::vector<uint8_t>& mask) {
    const auto valid=mask.empty()?std::vector<uint8_t>((size_t)photo.width*photo.height,255):mask;
    const auto fine=audit_common(photo,{before,after},photo,{valid,valid});
    const int w=std::max(16,photo.width/4),h=std::max(16,photo.height/4);
    auto low_pass=[&](const Image& image) {
        Image low{w,h,std::vector<float>((size_t)w*h*3)};
        resize_lanczos3<float,3>(image.rgb.data(),image.height,image.width,low.rgb.data(),h,w);
        return low;
    };
    std::vector<uint8_t> low_mask((size_t)w*h,255);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const int x0=x*photo.width/w,x1=((x+1)*photo.width+w-1)/w;
        const int y0=y*photo.height/h,y1=((y+1)*photo.height+h-1)/h;
        for(int yy=y0;yy<y1;++yy)for(int xx=x0;xx<x1;++xx)
            if(!valid[(size_t)yy*photo.width+xx])low_mask[(size_t)y*w+x]=0;
    }
    const auto low_photo=low_pass(photo);
    const auto coarse=audit_common(low_photo,{low_pass(before),low_pass(after)},low_photo,{low_mask,low_mask});
    CameraAudit result;
    result.fine_count=fine[0].count;result.coarse_count=coarse[0].count;
    result.fine_before=fine[0].after;result.fine_after=fine[1].after;
    result.coarse_before=coarse[0].after;result.coarse_after=coarse[1].after;
    const double fine_gain=result.fine_after-result.fine_before,coarse_gain=result.coarse_after-result.coarse_before;
    result.gain=.5*(fine_gain+coarse_gain);
    result.accepted=fine[0].count>=50&&coarse[0].count>=50&&fine_gain>=-.001&&coarse_gain>=-.001&&result.gain>.001;
    return result;
}
}  // namespace spirula::align
