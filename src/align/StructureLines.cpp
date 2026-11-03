#include "align/WarpMath.h"

namespace spirula::align::detail {

std::vector<LineConstraint> straight_lines(const Field& f,const std::vector<float>& photo,
                                         const std::vector<uint8_t>& mask) {
    std::vector<LineConstraint> lines;
    auto filtered=photo;std::vector<float> tmp(photo.size());
    for(int pass=0;pass<2;++pass) {
#pragma omp parallel for schedule(static)
        for(int y=0;y<f.height;++y)for(int x=0;x<f.width;++x) {
            const size_t k=(size_t)y*f.width+x;
            const size_t a=pass?(size_t)std::max(0,y-1)*f.width+x:(size_t)y*f.width+std::max(0,x-1);
            const size_t b=pass?(size_t)std::min(f.height-1,y+1)*f.width+x:(size_t)y*f.width+std::min(f.width-1,x+1);
            tmp[k]=.25f*filtered[a]+.5f*filtered[k]+.25f*filtered[b];
        }
        filtered.swap(tmp);
    }
    auto gradient=[&](float x,float y,float& dx,float& dy) {
        float a,b,c,d;
        if(!sample(filtered,f.width,f.height,1,x-1,y,&a)||!sample(filtered,f.width,f.height,1,x+1,y,&b)||
           !sample(filtered,f.width,f.height,1,x,y-1,&c)||!sample(filtered,f.width,f.height,1,x,y+1,&d))return false;
        if(!mask.empty())for(int j=-2;j<=2;++j)for(int i=-2;i<=2;++i)
            if(!mask[(size_t)((int)y+j)*f.width+(int)x+i])return false;
        dx=.5f*(b-a);dy=.5f*(d-c);return true;
    };
    for(int y=40;y<f.height-40;y+=f.spacing)for(int x=40;x<f.width-40;x+=f.spacing) {
        float best=0,px=0,py=0,nx=0,ny=0;
        for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i) {
            float dx,dy;if(!gradient(x+i,y+j,dx,dy))continue;
            const float length=std::hypot(dx,dy);
            if(length>best){best=length;px=x+i;py=y+j;nx=dx;ny=dy;}
        }
        if(best<.08f)continue;
        nx/=best;ny/=best;
        for(float span:{16.f,32.f}) {
            float sum=0,slope=0;std::array<float,9> offsets{};bool valid=true;
            for(int j=-4;j<=4;++j) {
                const float t=j*span/4;float strength=0,offset=0;
                for(int i=-3;i<=3;++i) {
                    const float n=.5f*i;float dx,dy;
                    if(!gradient(px-ny*t+nx*n,py+nx*t+ny*n,dx,dy))continue;
                    const float along=nx*dx+ny*dy,length=std::hypot(dx,dy);
                    if(along<.97f*length||along<=strength)continue;
                    strength=along;offset=n;
                }
                if(strength<std::max(.04f,.2f*best)){valid=false;break;}
                offsets[j+4]=offset;sum+=offset;slope+=t*offset;
            }
            if(!valid)continue;
            const float intercept=sum/9;slope/=60*span*span/16;
            for(int j=-4;j<=4;++j)if(std::abs(offsets[j+4]-intercept-slope*j*span/4)>.4f)valid=false;
            if(!valid)continue;
            LineConstraint line;
            const float tx=-ny+nx*slope,ty=nx+ny*slope,length=std::hypot(tx,ty);
            line.normal={ty/length,-tx/length};
            for(int j=-1;j<=1;++j) {
                const float xx=px+nx*intercept+j*span*tx,yy=py+ny*intercept+j*span*ty;
                const float gx=(xx+.5f)/f.spacing,gy=(yy+.5f)/f.spacing;
                const int ix=(int)gx,iy=(int)gy;const float ax=gx-ix,ay=gy-iy;
                for(int dy=0;dy<=1;++dy)for(int dx=0;dx<=1;++dx) {
                    const size_t node=(size_t)(iy+dy)*f.columns+ix+dx;
                    const float coefficient=(j==0?-2.f:1.f)*(dx?ax:1-ax)*(dy?ay:1-ay);
                    int k=0;while(k<line.count&&line.nodes[k]!=node)++k;
                    if(k==line.count)line.nodes[line.count++]=node;
                    line.coefficients[k]+=coefficient;
                }
            }
            lines.push_back(line);
        }
    }
    return lines;
}
}  // namespace spirula::align::detail
