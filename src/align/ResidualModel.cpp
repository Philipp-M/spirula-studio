#include "align/WarpMath.h"
#include "sfm/geometry/LinAlg.h"

namespace spirula::align::detail {
namespace {
constexpr int parameters=10;
using Row=std::array<double,parameters>;
using Vec=std::array<float,2>;

std::array<Row,2> basis(const Field& f,size_t k) {
    const double scale=.5*std::max(f.width,f.height);
    const double x=(std::min((int)(k%f.columns)*f.spacing,f.width)-.5-.5*f.width)/scale;
    const double y=(std::min((int)(k/f.columns)*f.spacing,f.height)-.5-.5*f.height)/scale;
    const double r=x*x+y*y;
    return {{{1,x,y,0,0,0,x*r,x*r*r,2*x*y,r+2*x*x},
             {0,0,0,1,x,y,y*r,y*r*r,r+2*y*y,2*x*y}}};
}
}

std::vector<Vec> smooth_residual(const Field& f,const std::vector<Vec>& data,
    const std::vector<float>& weights,const std::vector<std::array<float,4>>& structure) {
    const size_t n=data.size();
    std::vector<Vec> result(n);
    std::vector<float> ranges;
    for(size_t k=0;k<n;++k)if(weights[k]>.05f&&f.depth[k]>0&&std::isfinite(f.depth[k]))ranges.push_back(f.depth[k]);
    float median=1;
    if(!ranges.empty()) {
        auto middle=ranges.begin()+ranges.size()/2;std::nth_element(ranges.begin(),middle,ranges.end());median=*middle;
    }
    for(int iteration=0;iteration<6;++iteration) {
        std::array<double,parameters*parameters> matrix{},vectors{};
        Row rhs{},eigen{},solution{};
        auto add=[&](const Row& row,double target,double weight) {
            for(int i=0;i<parameters;++i) {
                rhs[i]+=weight*row[i]*target;
                for(int j=0;j<parameters;++j)matrix[i*parameters+j]+=weight*row[i]*row[j];
            }
        };
        for(size_t k=0;k<n;++k) {
            const auto rows=basis(f,k);
            const float error=std::hypot(result[k][0]-data[k][0],result[k][1]-data[k][1]);
            const float range_weight=f.depth[k]>0?std::clamp(median/f.depth[k],.25f,16.f):.25f;
            const float weight=weights[k]*range_weight*(iteration?std::min(1.f,1.5f/std::max(.001f,error)):1.f);
            for(int c=0;c<2;++c)add(rows[c],data[k][c],weight);
            const auto& a=structure[k];
            if(a[3]>0) {
                Row row{};for(int i=0;i<parameters;++i)row[i]=a[0]*rows[0][i]+a[1]*rows[1][i];
                const float residual=std::abs(a[0]*result[k][0]+a[1]*result[k][1]-a[2]);
                add(row,a[2],a[3]*(iteration?std::min(1.f,1.5f/std::max(.001f,residual)):1.f));
            }
        }
        double scale=0;for(int i=0;i<parameters;++i)scale=std::max(scale,matrix[i*parameters+i]);
        for(int i=0;i<parameters;++i)matrix[i*parameters+i]+=std::max(1e-8,scale*1e-6);
        sfm::jacobiEigenSymmetric(matrix.data(),parameters,eigen.data(),vectors.data());
        for(int j=0;j<parameters;++j)if(eigen[j]>std::max(1e-8,scale*1e-8)) {
            double projection=0;for(int i=0;i<parameters;++i)projection+=vectors[i*parameters+j]*rhs[i];
            for(int i=0;i<parameters;++i)solution[i]+=vectors[i*parameters+j]*projection/eigen[j];
        }
        for(size_t k=0;k<n;++k) {
            const auto rows=basis(f,k);
            for(int c=0;c<2;++c) {
                double value=0;for(int i=0;i<parameters;++i)value+=rows[c][i]*solution[i];
                result[k][c]=(float)value;
            }
        }
    }
    return result;
}

void limit_residual(Field& f,const std::vector<Vec>& baseline) {
    for(size_t k=0;k<f.displacement.size();++k) {
        const float x=2.f*std::min((int)(k%f.columns)*f.spacing,f.width)/f.width-1;
        const float y=2.f*std::min((int)(k/f.columns)*f.spacing,f.height)/f.height-1;
        const float radius=.5f*(x*x+y*y),limit=.25f+3.75f*radius*radius*radius;
        const float dx=f.displacement[k][0]-baseline[k][0],dy=f.displacement[k][1]-baseline[k][1];
        const float ratio=std::hypot(dx,dy)/limit;
        const float strength=ratio>1e-6f?std::tanh(ratio)/ratio:1.f;
        f.displacement[k]={baseline[k][0]+strength*dx,baseline[k][1]+strength*dy};
    }
}
}  // namespace spirula::align::detail
