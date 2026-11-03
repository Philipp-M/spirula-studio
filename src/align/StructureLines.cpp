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
    auto trace=[&](const std::array<float,5>& seed,float span,bool supplemental) {
        const float best=seed[0],px=seed[1],py=seed[2],nx=seed[3],ny=seed[4];
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
            if(strength<std::max(supplemental?.02f:.04f,.2f*best)){valid=false;break;}
            offsets[j+4]=offset;sum+=offset;slope+=t*offset;
        }
        if(!valid)return;
        const float intercept=sum/9;slope/=60*span*span/16;
        for(int j=-4;j<=4;++j)if(std::abs(offsets[j+4]-intercept-slope*j*span/4)>.4f)valid=false;
        if(!valid)return;
        const float tx=-ny+nx*slope,ty=nx+ny*slope,length=std::hypot(tx,ty);
        const float extent=span+(supplemental?std::min(f.spacing,8):0);
        for(bool derivative:{false,true}) {
            LineConstraint line;line.relative=derivative;
            if(derivative) {
                const float x=2*px/f.width-1,y=2*py/f.height-1;
                const float radius=.5f*(x*x+y*y);
                line.allowance=(supplemental?.04f:.005f+.035f*radius)*2*extent;
            }
            line.normal={ty/length,-tx/length};
            for(int j=-1;j<=1;++j) {
                if(derivative&&j==0)continue;
                const float xx=px+nx*intercept+j*extent*tx,yy=py+ny*intercept+j*extent*ty;
                const float gx=(xx+.5f)/f.spacing,gy=(yy+.5f)/f.spacing;
                const int ix=(int)gx,iy=(int)gy;const float ax=gx-ix,ay=gy-iy;
                for(int dy=0;dy<=1;++dy)for(int dx=0;dx<=1;++dx) {
                    const size_t node=(size_t)(iy+dy)*f.columns+ix+dx;
                    const float coefficient=(derivative?(float)j:j==0?-2.f:1.f)*(dx?ax:1-ax)*(dy?ay:1-ay);
                    int k=0;while(k<line.count&&line.nodes[k]!=node)++k;
                    if(k==line.count)line.nodes[line.count++]=node;
                    line.coefficients[k]+=coefficient;
                }
            }
            lines.push_back(line);
        }
    };
    std::vector<std::array<float,2>> supported((size_t)f.columns*f.rows);
    for(int pass=0;pass<2;++pass)for(int y=40;y<f.height-40;y+=f.spacing)for(int x=40;x<f.width-40;x+=f.spacing) {
        const int gx=x/f.spacing,gy=y/f.spacing;
        if(pass&&supported[(size_t)gy*f.columns+gx]!=std::array<float,2>{})continue;
        std::vector<std::array<float,5>> seeds;std::array<float,5> strongest{};
        for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i) {
            float dx,dy;if(!gradient(x+i,y+j,dx,dy))continue;
            const float length=std::hypot(dx,dy);
            if(length<.04f)continue;
            const std::array<float,5> seed{length,(float)(x+i),(float)(y+j),dx/length,dy/length};
            seeds.push_back(seed);if(length>strongest[0])strongest=seed;
        }
        const size_t existing=lines.size();
        if(!pass) {
            if(strongest[0]>=.08f)for(float span:{16.f,32.f})trace(strongest,span,false);
            if(lines.size()>existing)supported[(size_t)gy*f.columns+gx]=lines.back().normal;
            continue;
        }
        std::stable_sort(seeds.begin(),seeds.end(),[](const auto& a,const auto& b){return a[0]>b[0];});
        std::vector<std::array<float,2>> attempted;
        for(const auto& seed:seeds) {
            bool covered=false;
            for(int j=std::max(0,gy-2);j<=std::min(f.rows-1,gy+2);++j)
                for(int i=std::max(0,gx-2);i<=std::min(f.columns-1,gx+2);++i) {
                    const auto& n=supported[(size_t)j*f.columns+i];
                    covered|=std::abs(n[0]*seed[3]+n[1]*seed[4])>.95f;
                }
            if(covered)continue;
            bool nearby=false;for(const auto& p:attempted)if(std::hypot(seed[1]-p[0],seed[2]-p[1])<2.5f)nearby=true;
            if(nearby)continue;
            if(attempted.size()==4)break;
            attempted.push_back({seed[1],seed[2]});
            for(float span:{8.f,12.f,16.f})trace(seed,span,true);
            if(lines.size()>existing)break;
        }
    }
    return lines;
}
}  // namespace spirula::align::detail
