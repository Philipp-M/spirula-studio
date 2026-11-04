#include "align/WarpMath.h"
#include "core/LanczosResize.h"
#include <stdexcept>

namespace spirula::align {
namespace {
using Vec=std::array<float,2>;
struct Patch {int x,y;float before,range;bool structure;};

float sample_patch(const std::vector<float>& source,int w,int h,float x,float y) {
    const int ix=(int)std::floor(x)-2,iy=(int)std::floor(y)-2;
    double wx[6],wy[6],sum=0,total=0;
    for(int i=0;i<6;++i){wx[i]=lanczos_detail::kernel(x-ix-i);wy[i]=lanczos_detail::kernel(y-iy-i);}
    for(int j=0;j<6;++j)for(int i=0;i<6;++i) {
        const int xx=ix+i,yy=iy+j;if(xx<0||yy<0||xx>=w||yy>=h)continue;
        const double weight=wx[i]*wy[j];sum+=weight*source[(size_t)yy*w+xx];total+=weight;
    }
    return (float)(sum/total);
}

float warped_correlation(const Field& f,const std::vector<float>& source,
                         const std::vector<float>& reference,const std::vector<uint8_t>& mask,int x,int y) {
    double sa=0,sb=0,aa=0,bb=0,ab=0;
    for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i) {
        float sx,sy,v;
        if(!f.source((float)x+i,(float)y+j,sx,sy)||sx<0||sy<0||sx>f.width-1||sy>f.height-1)return -2;
        v=sample_patch(source,f.width,f.height,sx,sy);
        if(!mask.empty()&&!mask[(size_t)std::lround(sy)*f.width+(int)std::lround(sx)])return -2;
        const float u=reference[(size_t)(y+j)*f.width+x+i];
        sa+=u;sb+=v;aa+=u*u;bb+=v*v;ab+=u*v;
    }
    const double va=aa-sa*sa/49,vb=bb-sb*sb/49;
    return va>49*.0001&&vb>49*.0001?(float)((ab-sa*sb/49)/std::sqrt(va*vb)):-2;
}

Vec interpolate(const Field& f,const std::vector<Vec>& values,const Vec& source) {
    const float gx=(source[0]+.5f)/f.spacing,gy=(source[1]+.5f)/f.spacing;
    const int x=std::clamp((int)gx,0,f.columns-2),y=std::clamp((int)gy,0,f.rows-2);
    const float dx=std::max(1,std::min((x+1)*f.spacing,f.width)-x*f.spacing);
    const float dy=std::max(1,std::min((y+1)*f.spacing,f.height)-y*f.spacing);
    const float ax=std::clamp((source[0]+.5f-x*f.spacing)/dx,0.f,1.f);
    const float ay=std::clamp((source[1]+.5f-y*f.spacing)/dy,0.f,1.f);
    const size_t k=(size_t)y*f.columns+x;Vec value{};
    for(int c=0;c<2;++c)value[c]=(1-ay)*((1-ax)*values[k][c]+ax*values[k+1][c])+
        ay*((1-ax)*values[k+f.columns][c]+ax*values[k+f.columns+1][c]);
    return value;
}

std::vector<Vec> direct_field(const Field& f,const std::vector<float>& photo,const std::vector<float>& render,
                             const std::vector<uint8_t>& mask,const std::vector<float>& depth) {
    const size_t n=f.displacement.size();std::vector<Vec> data(n);std::vector<float> weights(n);
#pragma omp parallel for schedule(dynamic)
    for(int gy=1;gy<f.rows-1;gy+=2)for(int gx=1;gx<f.columns-1;gx+=2) {
        if(gx%5==2&&gy%5==2)continue;
        const float x=gx*f.spacing-.5f,y=gy*f.spacing-.5f;
        if(x<7||y<7||x>=f.width-7||y>=f.height-7)continue;
        bool valid=true;
        if(!mask.empty())for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i)
            valid&=mask[(size_t)std::lround(y+j)*f.width+(int)std::lround(x+i)]!=0;
        if(!valid)continue;
        float best=-2,qx=x,qy=y;
        for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i) {
            const float score=detail::correlation(photo,render,f.width,f.height,x,y,x+i,y+j,3);
            if(score>best){best=score;qx=x+i;qy=y+j;}
        }
        if(best<.75f||std::abs(qx-x)>=3||std::abs(qy-y)>=3)continue;
        auto subpixel=[&](float a,float b) {
            const float den=a-2*best+b;
            return den<-.00001f?std::clamp(.5f*(a-b)/den,-.5f,.5f):0.f;
        };
        const float dx=subpixel(detail::correlation(photo,render,f.width,f.height,x,y,qx-1,qy,3),
                                detail::correlation(photo,render,f.width,f.height,x,y,qx+1,qy,3));
        const float dy=subpixel(detail::correlation(photo,render,f.width,f.height,x,y,qx,qy-1,3),
                                detail::correlation(photo,render,f.width,f.height,x,y,qx,qy+1,3));
        const size_t k=(size_t)gy*f.columns+gx;data[k]={qx+dx-x,qy+dy-y};weights[k]=best*best;
    }
    if(std::count_if(weights.begin(),weights.end(),[](float w){return w>0;})<32)return {};
    Field grid;grid.width=f.width;grid.height=f.height;grid.spacing=f.spacing;grid.columns=f.columns;
    grid.depth.resize(n);
    for(size_t k=0;k<n;++k) {
        const int x=std::min((int)(k%f.columns)*f.spacing,f.width-1),y=std::min((int)(k/f.columns)*f.spacing,f.height-1);
        grid.depth[k]=depth[(size_t)y*f.width+x];
    }
    auto fitted=detail::smooth_residual(grid,data,weights,std::vector<std::array<float,4>>(n));
    float maximum=0;for(const auto& v:fitted)maximum=std::max(maximum,std::hypot(v[0],v[1]));
    if(maximum>3)for(auto& v:fitted)for(auto& c:v)c*=3/maximum;
    return fitted;
}
}

void select_residual(Field& f,const Image& photo,const Image& render,
                     const std::vector<float>& depth,const std::vector<float>& opacity,
                     const std::vector<uint8_t>& mask,const std::vector<Vec>& structures) {
    const size_t pixels=(size_t)f.width*f.height,nodes=(size_t)f.columns*f.rows;
    if(f.width<16||f.height<16||f.columns<2||f.rows<2||f.spacing<=0||
       photo.width!=f.width||photo.height!=f.height||render.width!=f.width||render.height!=f.height||
       photo.rgb.size()!=3*pixels||render.rgb.size()!=3*pixels||depth.size()!=pixels||opacity.size()!=pixels||
       (!mask.empty()&&mask.size()!=pixels)||f.depth.size()!=nodes||f.support.size()!=nodes||
       f.displacement.size()!=nodes||f.valid.size()!=pixels||f.triangles.empty())
        throw std::runtime_error("invalid residual selection inputs");
    f.residual_gain=0;
    const auto pg=detail::gray(photo),rg=detail::gray(render);std::vector<Patch> patches;
    std::vector<float> ranges;
    auto add=[&](int x,int y,bool structure) {
        if(x<7||y<7||x>=f.width-7||y>=f.height-7)return;
        if(!mask.empty())for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i)
            if(!mask[(size_t)(y+j)*f.width+x+i])return;
        const float before=detail::correlation(rg,pg,f.width,f.height,x,y,x,y,3);
        if(before<-1)return;
        const size_t k=(size_t)y*f.width+x;
        const float z=std::isfinite(depth[k])&&depth[k]>0&&opacity[k]>=.95f?depth[k]:0;
        patches.push_back({x,y,before,z,structure});if(z>0)ranges.push_back(z);
    };
    for(const auto& p:f.audit)add((int)p[0],(int)p[1],false);
    for(const auto& p:structures)if(((int)p[0]/8+3*((int)p[1]/8))%5==0)add((int)p[0],(int)p[1],true);
    if(patches.size()<10){use_identity(f);return;}
    float median=1;
    if(!ranges.empty()){auto mid=ranges.begin()+ranges.size()/2;std::nth_element(ranges.begin(),mid,ranges.end());median=*mid;}
    auto quality=[&] {
        double total=0,foreground=0,lines=0,weight=0;size_t count=0,close=0,structural=0;
        for(const auto& p:patches) {
            float after=warped_correlation(f,pg,rg,mask,p.x,p.y);
            const double gain=(after<-1?p.before-.05f:after)-p.before;
            if(p.structure){lines+=gain;++structural;continue;}
            total+=gain;++count;
            if(p.range>0) {
                const double w=std::clamp(std::pow(median/p.range,2),.25,64.);
                foreground+=w*gain;weight+=w;++close;
            }
        }
        total/=std::max(size_t(1),count);foreground/=std::max(1e-12,weight);lines/=std::max(size_t(1),structural);
        if(count<10||total<-.001||(close>=32&&foreground<-.001)||(structural>=12&&lines<-.001))return -1.;
        return total+(close>=32?foreground:0)+(structural>=12?lines:0);
    };
    const auto dense=f.triangles;const auto displacement=f.displacement;
    const bool available=!f.fallback;double best=available?quality():-1;
    bool selected=best>1e-5;float strength=1;int model=selected?1:0;
    auto best_triangles=selected?f.triangles:std::vector<Triangle>{};auto best_displacement=displacement;
    auto test=[&](const std::vector<Vec>* values,float scale,int kind) {
        f.triangles=dense;bool safe=true;
        for(auto& t:f.triangles) {
            for(int j=0;j<3;++j) {
                const auto delta=values?interpolate(f,*values,t.source[j]):Vec{t.target[j][0]-t.source[j][0],t.target[j][1]-t.source[j][1]};
                for(int c=0;c<2;++c)t.target[j][c]=t.source[j][c]+scale*delta[c];
            }
            const float det=detail::cross(t.target[0],t.target[1],t.target[2])/detail::cross(t.source[0],t.source[1],t.source[2]);
            safe&=std::isfinite(det)&&det>=.25f&&det<=4;
            for(int i=0;i<3;++i) {
                const int j=(i+1)%3;
                const float a=std::hypot(t.source[i][0]-t.source[j][0],t.source[i][1]-t.source[j][1]);
                const float b=std::hypot(t.target[i][0]-t.target[j][0],t.target[i][1]-t.target[j][1]);
                safe&=b>=.25f*a&&b<=3*a;
            }
        }
        if(!safe)return;
        index_field(f);const double score=quality();
        if(score<=std::max(1e-5,best)+1e-5)return;
        best=score;selected=true;strength=scale;model=kind;best_triangles=f.triangles;
        best_displacement=values?*values:displacement;for(auto& v:best_displacement)for(auto& c:v)c*=scale;
    };
    if(!selected&&available) {
        for(float scale:{.5f,.25f})test(nullptr,scale,1);
        const auto smooth=f.smooth_displacement.empty()?
            detail::smooth_residual(f,displacement,f.support,std::vector<std::array<float,4>>(displacement.size())):f.smooth_displacement;
        for(float scale:{1.f,.5f,.25f})test(&smooth,scale,2);
    }
    if(!selected) {
        const auto direct=direct_field(f,pg,rg,mask,depth);
        if(!direct.empty())for(float scale:{1.f,.5f,.25f})test(&direct,scale,3);
    }
    if(!selected){use_identity(f);return;}
    f.triangles=std::move(best_triangles);f.displacement=std::move(best_displacement);index_field(f);
    f.fallback=false;f.correction_scale*=strength;f.residual_model=model;
    f.residual_gain=best;
    if(f.correction_scale==0)f.correction_scale=strength;
#pragma omp parallel for schedule(static)
    for(int y=0;y<f.height;++y)for(int x=0;x<f.width;++x) {
        float sx,sy;f.valid[(size_t)y*f.width+x]=f.source((float)x,(float)y,sx,sy)?255:0;
    }
}
}
