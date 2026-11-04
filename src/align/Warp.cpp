#include "align/WarpMath.h"
#include "core/LanczosResize.h"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace spirula::align {
namespace {
void lanczos_weights(double fraction,double* weights) {
    constexpr int resolution=4096;
    static const auto table=[] {
        std::array<std::array<double,6>,resolution+1> values{};
        for(int j=0;j<=resolution;++j)for(int i=0;i<6;++i)
            values[j][i]=lanczos_detail::kernel((double)j/resolution+2-i);
        return values;
    }();
    const double coordinate=fraction*resolution;
    const int index=std::clamp((int)coordinate,0,resolution-1);
    const double alpha=coordinate-index;
    for(int i=0;i<6;++i)weights[i]=(1-alpha)*table[index][i]+alpha*table[index+1][i];
}
}

void use_identity(Field& f) {
    for(auto& t:f.triangles)t.target=t.source;
    std::fill(f.displacement.begin(),f.displacement.end(),std::array<float,2>{});
    std::fill(f.valid.begin(),f.valid.end(),255);f.correction_scale=0;f.fallback=true;index_field(f);
}

void index_field(Field& f) {
    f.bins.clear();f.bins.resize((size_t)f.columns*f.rows);
    for(uint32_t i=0;i<f.triangles.size();++i) {
        const auto& t=f.triangles[i];
        float x0=t.target[0][0],x1=x0,y0=t.target[0][1],y1=y0;
        for(int j=1;j<3;++j){x0=std::min(x0,t.target[j][0]);x1=std::max(x1,t.target[j][0]);y0=std::min(y0,t.target[j][1]);y1=std::max(y1,t.target[j][1]);}
        if(x1<-.5f||y1<-.5f||x0>f.width-.5f||y0>f.height-.5f)continue;
        const int ax=std::clamp((int)std::floor((x0+.5f)/f.spacing),0,f.columns-1),bx=std::clamp((int)std::floor((x1+.5f)/f.spacing),0,f.columns-1);
        const int ay=std::clamp((int)std::floor((y0+.5f)/f.spacing),0,f.rows-1),by=std::clamp((int)std::floor((y1+.5f)/f.spacing),0,f.rows-1);
        for(int y=ay;y<=by;++y)for(int x=ax;x<=bx;++x)f.bins[(size_t)y*f.columns+x].push_back(i);
    }
}

bool Field::source(float x,float y,float& sx,float& sy,bool allow_outside) const {
    if(!std::isfinite(x)||!std::isfinite(y)||x<-.5f||y<-.5f||x>width-.5f||y>height-.5f||bins.empty())return false;
    const int bx=std::clamp((int)((x+.5f)/spacing),0,columns-1),by=std::clamp((int)((y+.5f)/spacing),0,rows-1);
    float nearest=1e31f;bool covered=false;
    for(uint32_t id:bins[(size_t)by*columns+bx]) {
        const auto& t=triangles[id];std::array<float,3> b;
        if(!detail::barycentric(t,x,y,b))continue;
        const float z=detail::triangle_depth(t,b);if(covered&&z>=nearest)continue;
        nearest=z;sx=sy=0;for(int i=0;i<3;++i){sx+=b[i]*t.source[i][0];sy+=b[i]*t.source[i][1];}covered=true;
    }
    return covered&&(allow_outside||(sx>=-.501f&&sy>=-.501f&&sx<=width-.499f&&sy<=height-.499f));
}

size_t transport_observations(const Field& field,int width,int height,std::vector<std::array<double,2>>& observations) {
    if(width<=0||height<=0||field.width<=0||field.height<=0)throw std::runtime_error("invalid observation image dimensions");
    Field inverse;inverse.width=field.width;inverse.height=field.height;
    inverse.columns=field.columns;inverse.rows=field.rows;inverse.spacing=field.spacing;
    inverse.triangles=field.triangles;
    for(auto& t:inverse.triangles)std::swap(t.source,t.target);
    index_field(inverse);size_t outside=0;
    for(auto& point:observations) {
        if(!std::isfinite(point[0])||!std::isfinite(point[1]))throw std::runtime_error("nonfinite COLMAP observation");
        // Cropped/undistorted COLMAP models can retain observations outside the delivered image.
        if(point[0]<0||point[1]<0||point[0]>width||point[1]>height){++outside;continue;}
        float tx,ty;
        // COLMAP uses corner-based coordinates; fields use integer pixel centers.
        if(!inverse.source((float)(point[0]*field.width/width-.5),(float)(point[1]*field.height/height-.5),tx,ty,true))
            throw std::runtime_error("source warp does not cover an in-image COLMAP observation");
        point={(tx+.5)*width/field.width,(ty+.5)*height/field.height};
    }
    return outside;
}

void resample(const std::vector<uint8_t>& original,int w,int h,const std::vector<uint8_t>& original_mask,
              const Field& f,std::vector<uint8_t>& corrected,std::vector<uint8_t>& mask,Coverage* coverage) {
    if(w<=0||h<=0||original.size()!=(size_t)w*h*3||(!original_mask.empty()&&original_mask.size()!=(size_t)w*h))
        throw std::runtime_error("invalid native image or mask");
    corrected.assign(original.size(),0);mask.assign((size_t)w*h,0);
    if(coverage){*coverage={};coverage->pixels=(size_t)w*h;coverage->reason.resize((size_t)w*h,1);}
    if(std::all_of(f.triangles.begin(),f.triangles.end(),[](const auto& t){return t.source==t.target;})&&!f.triangles.empty()) {
        corrected=original;
        size_t masked=0;
        for(size_t k=0;k<mask.size();++k) {
            mask[k]=original_mask.empty()?255:original_mask[k];masked+=!mask[k];
            if(!mask[k])for(int c=0;c<3;++c)corrected[3*k+c]=0;
            if(coverage)coverage->reason[k]=mask[k]?0:2;
        }
        if(coverage){coverage->covered=mask.size();coverage->masked=masked;}
        return;
    }
    const double ax=(double)w/f.width,ay=(double)h/f.height;
    size_t covered=0,masked=0;
#pragma omp parallel for schedule(dynamic) reduction(+:covered,masked)
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const size_t p=(size_t)y*w+x;float sx,sy;
        if(!f.source((float)((x+.5)/ax-.5),(float)((y+.5)/ay-.5),sx,sy))continue;
        ++covered;
        const double qx=std::clamp((sx+.5)*ax-.5,0.0,(double)w-1),qy=std::clamp((sy+.5)*ay-.5,0.0,(double)h-1);
        const int center_x=(int)std::lround(qx),center_y=(int)std::lround(qy);
        if(!original_mask.empty()&&!original_mask[(size_t)center_y*w+center_x]) {
            ++masked;if(coverage)coverage->reason[p]=2;continue;
        }
        if(coverage)coverage->reason[p]=0;
        const int ix=(int)std::floor(qx)-2,iy=(int)std::floor(qy)-2;double wx[6],wy[6],sum[3]={},total=0;
        lanczos_weights(qx-std::floor(qx),wx);lanczos_weights(qy-std::floor(qy),wy);
        for(int j=0;j<6;++j)for(int i=0;i<6;++i) {
            const int xx=ix+i,yy=iy+j;const double v=wx[i]*wy[j];
            if(xx<0||yy<0||xx>=w||yy>=h||(!original_mask.empty()&&!original_mask[(size_t)yy*w+xx]))continue;
            for(int c=0;c<3;++c)sum[c]+=v*original[((size_t)yy*w+xx)*3+c];total+=v;
        }
        for(int c=0;c<3;++c)corrected[3*p+c]=std::abs(total)>1e-8?(uint8_t)std::clamp(std::lround(sum[c]/total),0l,255l):original[((size_t)center_y*w+center_x)*3+c];
        mask[p]=255;
    }
    if(coverage){coverage->covered=covered;coverage->masked=masked;}
}

namespace {
template<class T> void write_vector(std::ostream& out,const std::vector<T>& v) {
    const uint64_t size=v.size();out.write((const char*)&size,sizeof(size));
    out.write((const char*)v.data(),(std::streamsize)(size*sizeof(T)));
}
template<class T> void read_vector(std::istream& in,std::vector<T>& v,size_t maximum) {
    uint64_t size=0;in.read((char*)&size,sizeof(size));
    if(!in||size>maximum)throw std::runtime_error("invalid alignment field length");
    v.resize((size_t)size);in.read((char*)v.data(),(std::streamsize)(size*sizeof(T)));
    if(!in)throw std::runtime_error("truncated alignment field");
}
}

void save_field(const Field& f,const std::filesystem::path& path) {
    std::ofstream out(path,std::ios::binary);out.write("SSALIGN2",8);
    for(int v:{f.width,f.height,f.columns,f.rows,f.spacing})out.write((const char*)&v,sizeof(v));
    out.write((const char*)&f.depth_jump,sizeof(f.depth_jump));
    write_vector(out,f.displacement);write_vector(out,f.depth);write_vector(out,f.support);write_vector(out,f.valid);
    write_vector(out,f.audit);write_vector(out,f.triangles);
    if(!out)throw std::runtime_error("cannot save alignment field");
}
Field load_field(const std::filesystem::path& path) {
    Field f;std::ifstream in(path,std::ios::binary);char magic[8]{};in.read(magic,8);
    if(std::string(magic,8)!="SSALIGN2")throw std::runtime_error("unknown alignment field");
    for(int* p:{&f.width,&f.height,&f.columns,&f.rows,&f.spacing})in.read((char*)p,sizeof(int));
    in.read((char*)&f.depth_jump,sizeof(f.depth_jump));
    const uint64_t pixel_count=(uint64_t)std::max(0,f.width)*std::max(0,f.height);
    if(!in||f.width<16||f.height<16||pixel_count>std::numeric_limits<int>::max()||f.spacing<4||f.spacing>8192||
       f.columns!=(f.width+f.spacing-1)/f.spacing+1||f.rows!=(f.height+f.spacing-1)/f.spacing+1||
       !std::isfinite(f.depth_jump)||f.depth_jump<=0)throw std::runtime_error("invalid alignment field dimensions");
    const size_t pixels=(size_t)f.width*f.height,n=(size_t)f.columns*f.rows;
    read_vector(in,f.displacement,n);read_vector(in,f.depth,n);read_vector(in,f.support,n);
    read_vector(in,f.valid,pixels);read_vector(in,f.audit,n);read_vector(in,f.triangles,n*12);
    if(f.displacement.size()!=n||f.depth.size()!=n||f.support.size()!=n||f.valid.size()!=pixels||f.triangles.empty())
        throw std::runtime_error("incomplete alignment field");
    for(const auto& t:f.triangles) {
        for(const auto* coords:{&t.source,&t.target})for(const auto& p:*coords)for(float v:p)
            if(!std::isfinite(v)||std::abs(v)>std::numeric_limits<int>::max()/4.0)throw std::runtime_error("nonfinite or unbounded alignment triangle");
        for(float z:t.depth)if(!std::isfinite(z)||z<=0)throw std::runtime_error("invalid alignment triangle depth");
    }
    index_field(f);return f;
}

std::filesystem::path checked_output(const std::filesystem::path& dataset,const std::filesystem::path& output) {
    namespace fs=std::filesystem;
    if(output.empty())throw std::runtime_error("an output directory is required");
    const auto source=fs::canonical(dataset),dest=fs::weakly_canonical(fs::absolute(output));
    const auto rel=dest.lexically_relative(source);
    if(rel.empty()||(!rel.is_absolute()&&*rel.begin()!=".."))throw std::runtime_error("output must be outside the read-only dataset");
    if(fs::exists(dest))throw std::runtime_error("output already exists; use a fresh directory");return dest;
}
}  // namespace spirula::align
