#include "align/WarpMath.h"
#include "core/LanczosResize.h"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        namespace A=spirula::align;
        const int w=256,h=192;
        A::Field field;field.width=w;field.height=h;field.columns=field.rows=2;field.spacing=w;
        const std::array<float,2> corners[4]={{{-.5f,-.5f}},{{w-.5f,-.5f}},{{w-.5f,h-.5f}},{{-.5f,h-.5f}}};
        for(auto ids:{std::array<int,3>{0,1,2},std::array<int,3>{0,2,3}}) {
            A::Triangle t;
            for(int j=0;j<3;++j) {
                const auto p=corners[ids[j]];t.source[j]=p;
                t.target[j]={1.0031f*p[0]+.0137f*p[1]+.31f,1.0047f*p[1]-.009f*p[0]+.51f};t.depth[j]=10;
            }
            field.triangles.push_back(t);
        }
        A::index_field(field);
        std::vector<uint8_t> source((size_t)w*h*3),mask((size_t)w*h,255),output,valid;
        uint32_t state=7;
        for(auto& c:source){state=1664525*state+1013904223;c=state>>24;}
        for(int y=80;y<90;++y)for(int x=100;x<110;++x)mask[(size_t)y*w+x]=0;
        A::resample(source,w,h,mask,field,output,valid);
        double error=0;int maximum=0;size_t count=0;
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;if(!valid[k])continue;
            float sx,sy;if(!field.source(x,y,sx,sy))throw std::runtime_error("coverage mismatch");
            const double qx=std::clamp((double)sx,0.,w-1.),qy=std::clamp((double)sy,0.,h-1.);
            const int ix=(int)std::floor(qx)-2,iy=(int)std::floor(qy)-2;
            double sum[3]={},total=0;
            for(int j=0;j<6;++j)for(int i=0;i<6;++i) {
                const int xx=ix+i,yy=iy+j;if(xx<0||yy<0||xx>=w||yy>=h||!mask[(size_t)yy*w+xx])continue;
                const double weight=spirula::lanczos_detail::kernel(qx-xx)*spirula::lanczos_detail::kernel(qy-yy);
                for(int c=0;c<3;++c)sum[c]+=weight*source[((size_t)yy*w+xx)*3+c];total+=weight;
            }
            for(int c=0;c<3;++c) {
                const int reference=(int)std::clamp(std::lround(sum[c]/total),0l,255l);
                const int delta=std::abs((int)output[3*k+c]-reference);error+=delta;maximum=std::max(maximum,delta);++count;
            }
        }
        std::cout<<"Lanczos table: maximum byte error="<<maximum<<", mean="<<error/count<<'\n';
        if(maximum>1||error/count>.0001)throw std::runtime_error("tabulated Lanczos differs from analytic sampling");
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
