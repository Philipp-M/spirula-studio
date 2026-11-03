#include "align/WarpMath.h"

#include <numeric>
#include <stdexcept>

namespace spirula::align {
namespace {
using namespace detail;
using Vec=std::array<float,2>;

float median(std::vector<float> values) {
    if(values.empty())return 0;
    const size_t m=values.size()/2;std::nth_element(values.begin(),values.begin()+m,values.end());return values[m];
}

bool compatible(float a,float b,float jump) {
    return std::isfinite(a)&&std::isfinite(b)&&a>0&&b>0&&std::abs(a-b)<=jump*std::min(a,b);
}

MatchDiagnostics diagnose(const Image& photo,const Image& render,const std::vector<float>& depth,
                          const std::vector<float>& opacity,const roma::PairPrediction& p,const Settings& s) {
    MatchDiagnostics d;const int w=render.width,h=render.height;
    const auto pg=gray(photo),rg=gray(render);std::vector<float> cycles,movement;
    for(int y=s.patch_radius;y<h-s.patch_radius;y+=s.spacing)for(int x=s.patch_radius;x<w-s.patch_radius;x+=s.spacing) {
        ++d.sampled;float qx,qy,c,bx,by,bc;
        if(!match(p.forward,x,y,w,h,qx,qy,c))continue;
        ++d.finite;d.mean_overlap+=c;movement.push_back(std::hypot(qx-x,qy-y));
        const bool reverse=match(p.backward,qx,qy,w,h,bx,by,bc);
        const float cycle=reverse?std::hypot(bx-x,by-y):1e6f;if(reverse)cycles.push_back(cycle);
        const bool overlap=c>=s.min_overlap&&reverse&&bc>=s.min_overlap;
        const bool consistent=overlap&&cycle<=s.max_cycle;
        const bool bounded=consistent&&std::hypot(qx-x,qy-y)<=detail::displacement_limit(s,w,h);
        d.overlap+=overlap;d.cycle+=consistent;d.bounded+=bounded;
        if(!bounded||opacity[(size_t)y*w+x]<.95f)continue;
        ++d.old_opacity;bool interior=true;
        for(int j=-s.patch_radius;j<=s.patch_radius;++j)for(int i=-s.patch_radius;i<=s.patch_radius;++i)
            interior&=compatible(depth[(size_t)y*w+x],depth[(size_t)(y+j)*w+x+i],s.depth_jump);
        if(!interior)continue;
        ++d.old_depth;float best=-2;
        for(int j=-s.refine_radius;j<=s.refine_radius;++j)for(int i=-s.refine_radius;i<=s.refine_radius;++i)
            best=std::max(best,correlation(rg,pg,w,h,x,y,qx+i,qy+j,s.patch_radius));
        if(best<s.min_correlation)continue;
        ++d.old_correlation;++d.old_accepted;
    }
    if(d.finite)d.mean_overlap/=d.finite;
    d.median_cycle=median(cycles);d.median_displacement=median(movement);return d;
}

std::vector<Vec> broad_field(const std::vector<Vec>& data,const std::vector<float>& weight,const Field& f,
                             const std::vector<uint32_t>& labels={}) {
    using Matrix=std::array<std::array<double,5>,3>;
    std::vector<Vec> out(data.size());std::vector<float> robust=weight;
    std::vector<Matrix> matrices(labels.empty()?1:data.size());
    auto coordinates=[&](int x,int y){return std::array<double,3>{1,2.0*std::min(x*f.spacing,f.width)/f.width-1,2.0*std::min(y*f.spacing,f.height)/f.height-1};};
    for(int iteration=0;iteration<6;++iteration) {
        std::fill(matrices.begin(),matrices.end(),Matrix{});
        for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x) {
            const size_t k=(size_t)y*f.columns+x;const auto a=coordinates(x,y);
            auto& matrix=matrices[labels.empty()?0:labels[k]];
            for(int i=0;i<3;++i){for(int j=0;j<3;++j)matrix[i][j]+=robust[k]*a[i]*a[j];
                for(int c=0;c<2;++c)matrix[i][3+c]+=robust[k]*a[i]*data[k][c];}
        }
        for(auto& matrix:matrices) {
            if(matrix[0][0]==0)continue;
            const double cx=matrix[0][1]/matrix[0][0],cy=matrix[0][2]/matrix[0][0];
            for(int i=0;i<3;++i)matrix[i][i]+=.01;
            for(int i=0;i<3;++i) {
                int pivot=i;for(int j=i+1;j<3;++j)if(std::abs(matrix[j][i])>std::abs(matrix[pivot][i]))pivot=j;
                std::swap(matrix[i],matrix[pivot]);
                const double v=matrix[i][i];for(int j=i;j<5;++j)matrix[i][j]/=v;
                for(int k=0;k<3;++k)if(k!=i){const double t=matrix[k][i];for(int j=i;j<5;++j)matrix[k][j]-=t*matrix[i][j];}
            }
            if(!labels.empty()) {
                double gradient=0;
                for(int c=0;c<2;++c)gradient+=std::pow(2*matrix[1][3+c]/f.width,2)+std::pow(2*matrix[2][3+c]/f.height,2);
                const double scale=std::min(1.0,.1/std::max(1e-12,std::sqrt(gradient)));
                for(int c=0;c<2;++c) {
                    matrix[0][3+c]+=(1-scale)*(cx*matrix[1][3+c]+cy*matrix[2][3+c]);
                    matrix[1][3+c]*=scale;matrix[2][3+c]*=scale;
                }
            }
        }
        for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x) {
            const size_t k=(size_t)y*f.columns+x;const auto a=coordinates(x,y);
            const auto& matrix=matrices[labels.empty()?0:labels[k]];
            for(int c=0;c<2;++c)out[k][c]=(float)(matrix[0][3+c]+matrix[1][3+c]*a[1]+matrix[2][3+c]*a[2]);
            const float residual=std::hypot(out[k][0]-data[k][0],out[k][1]-data[k][1]);
            robust[k]=weight[k]*std::min(1.0f,3/std::max(.01f,residual));
        }
    }
    return out;
}

std::vector<uint32_t> surfaces(const Field& f,const std::vector<float>& support) {
    std::vector<uint32_t> parent(f.displacement.size());std::iota(parent.begin(),parent.end(),0);
    auto root=[&](uint32_t k){while(parent[k]!=k){parent[k]=parent[parent[k]];k=parent[k];}return k;};
    for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x) {
        const size_t k=(size_t)y*f.columns+x;
        for(int j:{x+1<f.columns?(int)k+1:-1,y+1<f.rows?(int)k+f.columns:-1}) {
            if(j<0)continue;
            const float a=f.depth[k],b=f.depth[j];
            const float motion=std::hypot(f.displacement[k][0]-f.displacement[j][0],f.displacement[k][1]-f.displacement[j][1]);
            const bool cut=support[k]>.12f&&support[j]>.12f&&a>0&&b>0&&std::isfinite(a)&&std::isfinite(b)&&
                std::abs(a-b)>.12f*std::min(a,b)&&motion>1.5f;
            if(!cut)parent[root((uint32_t)k)]=root((uint32_t)j);
        }
    }
    for(auto& k:parent)k=root(k);
    return parent;
}

void mesh(Field& f,const std::vector<uint32_t>& labels) {
    f.triangles.clear();
    auto point=[&](int k){return Vec{std::min((k%f.columns)*f.spacing,f.width)-.5f,std::min((k/f.columns)*f.spacing,f.height)-.5f};};
    for(int y=0;y<f.rows-1;++y)for(int x=0;x<f.columns-1;++x) {
        const int a=y*f.columns+x,b=a+1,c=a+f.columns,d=c+1;
        for(const auto ids:{std::array<int,3>{a,b,d},std::array<int,3>{a,d,c}}) {
            std::array<Vec,3> uv{point(ids[0]),point(ids[1]),point(ids[2])};
            auto add=[&](const std::vector<std::array<float,3>>& polygon,uint32_t label) {
                auto vertex=[&](const std::array<float,3>& bary,Vec& source,Vec& target,float& depth) {
                    source={0,0};Vec displacement{};float weight=0,inverse=0;
                    for(int i=0;i<3;++i) {
                        for(int c=0;c<2;++c)source[c]+=bary[i]*uv[i][c];
                        if(labels[ids[i]]!=label)continue;
                        weight+=bary[i];for(int c=0;c<2;++c)displacement[c]+=bary[i]*f.displacement[ids[i]][c];
                        if(f.depth[ids[i]]>0&&std::isfinite(f.depth[ids[i]]))inverse+=bary[i]/f.depth[ids[i]];
                    }
                    for(int c=0;c<2;++c)target[c]=source[c]+displacement[c]/std::max(1e-8f,weight);
                    depth=inverse>0?weight/inverse:1e30f;
                };
                for(size_t i=1;i+1<polygon.size();++i) {
                    Triangle t;const std::array<float,3> bary[3]={polygon[0],polygon[i],polygon[i+1]};
                    for(int j=0;j<3;++j)vertex(bary[j],t.source[j],t.target[j],t.depth[j]);
                    f.triangles.push_back(t);
                }
            };
            const uint32_t l0=labels[ids[0]],l1=labels[ids[1]],l2=labels[ids[2]];
            if(l0==l1&&l1==l2){add({{1,0,0},{0,1,0},{0,0,1}},l0);continue;}
            // Barycentric ownership partitions the source triangle without removing source area.
            for(int i=0;i<3;++i) {
                const int j=(i+1)%3,k=(i+2)%3;
                std::array<float,3> v{},m1{},m2{},center{1.0f/3,1.0f/3,1.0f/3};
                v[i]=1;m1[i]=m1[j]=.5f;m2[i]=m2[k]=.5f;
                add({v,m1,center,m2},labels[ids[i]]);
            }
        }
    }
}

double photometric_score(const Field& f,const std::vector<float>& photo,const std::vector<float>& render,
                         const std::vector<uint8_t>& source_mask) {
    double score=0;size_t count=0;
    for(int y=12;y<f.height-12;y+=24)for(int x=12;x<f.width-12;x+=24) {
        const int ax=(x-16+4000)%40,ay=(y-16+4000)%40;
        if(std::min(ax,40-ax)<=7&&std::min(ay,40-ay)<=7)continue;
        const float before=correlation(render,photo,f.width,f.height,x,y,x,y,3);
        if(before<-1)continue;
        double sa=0,sb=0,aa=0,bb=0,ab=0;bool usable=true;
        for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i) {
            float sx,sy,v;
            if(!f.source((float)x+i,(float)y+j,sx,sy)||!sample(photo,f.width,f.height,1,sx,sy,&v)) {usable=false;continue;}
            if(!source_mask.empty()&&!source_mask[(size_t)std::clamp((int)std::lround(sy),0,f.height-1)*f.width+std::clamp((int)std::lround(sx),0,f.width-1)]){usable=false;continue;}
            const float u=render[(size_t)(y+j)*f.width+x+i];sa+=u;sb+=v;aa+=u*u;bb+=v*v;ab+=u*v;
        }
        const double va=aa-sa*sa/49,vb=bb-sb*sb/49;
        score+=usable&&va>1e-6&&vb>1e-6?(ab-sa*sb/49)/std::sqrt(va*vb):before-.05;
        ++count;
    }
    return count?score/count:0;
}

void choose_strength(Field& f,const std::vector<float>& photo,const std::vector<float>& render,
                     const std::vector<uint8_t>& source_mask,const std::vector<Vec>& base,
                     const std::vector<uint32_t>& labels) {
    const auto triangles=f.triangles;const auto displacement=f.displacement;
    f.displacement=base;mesh(f,labels);const auto baseline=f.triangles;
    f.displacement=displacement;f.triangles=triangles;index_field(f);
    double best=photometric_score(f,photo,render,source_mask);float scale=1;
    for(float candidate:{0.0f,.25f,.5f,.75f}) {
        f.triangles=triangles;bool safe=true;
        for(size_t k=0;k<f.triangles.size();++k) {
            auto& t=f.triangles[k];
            for(int i=0;i<3;++i)for(int c=0;c<2;++c)t.target[i][c]=baseline[k].target[i][c]+candidate*(t.target[i][c]-baseline[k].target[i][c]);
            if(cross(t.target[0],t.target[1],t.target[2])<.25f*cross(t.source[0],t.source[1],t.source[2]))safe=false;
        }
        if(!safe)continue;
        index_field(f);const double score=photometric_score(f,photo,render,source_mask);
        if(score>best+.0005){best=score;scale=candidate;}
    }
    f.triangles=triangles;
    for(size_t k=0;k<f.triangles.size();++k)for(int i=0;i<3;++i)for(int c=0;c<2;++c)
        f.triangles[k].target[i][c]=baseline[k].target[i][c]+scale*(f.triangles[k].target[i][c]-baseline[k].target[i][c]);
    for(size_t k=0;k<f.displacement.size();++k)for(int c=0;c<2;++c)f.displacement[k][c]=base[k][c]+scale*(displacement[k][c]-base[k][c]);
    f.correction_scale*=scale;
}
}  // namespace

Field fit(const Image& photo,const Image& render,const std::vector<float>& depth,
          const std::vector<float>& opacity,const roma::PairPrediction& matches,const Settings& s,const std::vector<uint8_t>& source_mask,bool refine_structure) {
    const int w=render.width,h=render.height;
    if(w<16||h<16||photo.width!=w||photo.height!=h||depth.size()!=(size_t)w*h||opacity.size()!=depth.size()||
       photo.rgb.size()!=depth.size()*3||render.rgb.size()!=photo.rgb.size()||s.spacing<4||s.patch_radius<1||
       s.refine_radius<0||!(s.min_overlap>0&&s.min_overlap<=1)||!(s.min_correlation>0&&s.min_correlation<=1)||!(s.depth_jump>0)||!(s.max_cycle>0)||!std::isfinite(s.max_displacement)||s.max_displacement<0||matches.backward.warp.empty()||
       (!source_mask.empty()&&source_mask.size()!=(size_t)w*h))
        throw std::runtime_error("invalid image-alignment inputs");
    Field f;f.width=w;f.height=h;f.spacing=s.spacing;f.depth_jump=s.depth_jump;
    const float max_displacement=detail::displacement_limit(s,w,h);
    f.columns=(w+s.spacing-1)/s.spacing+1;f.rows=(h+s.spacing-1)/s.spacing+1;
    const size_t n=(size_t)f.columns*f.rows;f.displacement.resize(n);f.depth.resize(n);f.support.resize(n);
    f.diagnostics=diagnose(photo,render,depth,opacity,matches,s);
    std::vector<float> weights(n),precision(n,1);std::vector<Vec> data(n);const auto rg=detail::gray(render),pg=detail::gray(photo);
    if(!matches.backward.precision.empty()) {
        std::vector<float> measured;
        for(int gy=0;gy<f.rows;++gy)for(int gx=0;gx<f.columns;++gx) {
            const float x=std::clamp(std::min(gx*s.spacing,w)*(float)matches.backward.width/w-.5f,0.0f,(float)matches.backward.width-1);
            const float y=std::clamp(std::min(gy*s.spacing,h)*(float)matches.backward.height/h-.5f,0.0f,(float)matches.backward.height-1);float v[3];
            if(detail::sample(matches.backward.precision,matches.backward.width,matches.backward.height,3,x,y,v)) {
                const float eigen=.5f*(v[0]+v[2]-std::hypot(v[0]-v[2],2*v[1]));
                precision[(size_t)gy*f.columns+gx]=std::isfinite(eigen)?std::max(0.0f,eigen):0;
                if(eigen>0&&std::isfinite(eigen))measured.push_back(eigen);
            }
        }
        const float scale=std::max(1e-8f,median(measured));
        for(float& v:precision)v=std::clamp(std::sqrt(v/scale),.1f,1.0f);
    }
#pragma omp parallel for schedule(dynamic)
    for(int gy=0;gy<f.rows;++gy)for(int gx=0;gx<f.columns;++gx) {
        const size_t k=(size_t)gy*f.columns+gx;
        const float x=std::clamp(std::min(gx*s.spacing,w)-.5f,0.0f,(float)w-1),y=std::clamp(std::min(gy*s.spacing,h)-.5f,0.0f,(float)h-1);
        if(!source_mask.empty()&&!source_mask[(size_t)std::lround(y)*w+(int)std::lround(x)])continue;
        float qx,qy,c,bx,by,bc;
        if(!detail::match(matches.backward,x,y,w,h,qx,qy,c)||std::hypot(qx-x,qy-y)>max_displacement||
           !detail::match(matches.forward,qx,qy,w,h,bx,by,bc))continue;
        const float cycle=std::hypot(bx-x,by-y);
        const float overlap=std::clamp(c*bc,0.0f,1.0f);
        float weight=overlap/(1+std::exp(12*(s.min_overlap-std::min(c,bc))))*precision[k]/(1+std::pow(cycle/s.max_cycle,4.0f));
        if(weight<1e-5f)continue;
        const float ox=qx,oy=qy,initial=detail::correlation(pg,rg,w,h,x,y,qx,qy,s.patch_radius);
        float best=initial;
        for(int j=-s.refine_radius;j<=s.refine_radius;++j)for(int i=-s.refine_radius;i<=s.refine_radius;++i) {
            const float score=detail::correlation(pg,rg,w,h,x,y,ox+i,oy+j,s.patch_radius);
            if(score>best+.001f){best=score;qx=ox+i;qy=oy+j;}
        }
        if(best>-1) {
            auto offset=[&](float a,float b){const float den=a-2*best+b;return den<-.00001f?std::clamp(.5f*(a-b)/den,-.5f,.5f):0.0f;};
            qx+=offset(detail::correlation(pg,rg,w,h,x,y,qx-1,qy,s.patch_radius),detail::correlation(pg,rg,w,h,x,y,qx+1,qy,s.patch_radius));
            qy+=offset(detail::correlation(pg,rg,w,h,x,y,qx,qy-1,s.patch_radius),detail::correlation(pg,rg,w,h,x,y,qx,qy+1,s.patch_radius));
        }
        if(std::hypot(qx-x,qy-y)>max_displacement)continue;
        if(!detail::match(matches.forward,qx,qy,w,h,bx,by,bc))continue;
        const float refined_cycle=std::hypot(bx-x,by-y);
        weight=std::clamp(c*bc,0.0f,1.0f)/(1+std::exp(12*(s.min_overlap-std::min(c,bc))))*precision[k]/(1+std::pow(refined_cycle/s.max_cycle,4.0f));
        const float refined=detail::correlation(pg,rg,w,h,x,y,qx,qy,s.patch_radius);
        if(refined>-1){const float texture=std::clamp(refined/s.min_correlation,0.0f,1.0f);weight*=.25f+.75f*texture*texture;}
        float z=0;if(detail::sample(depth,w,h,1,qx,qy,&z)&&std::isfinite(z)&&z>0)f.depth[k]=z;
        if(gx%5==2&&gy%5==2)continue;
        weights[k]=weight;data[k]={qx-x,qy-y};f.support[k]=weight;
    }
    for(float weight:weights)f.anchors+=weight>.05f;
    f.fallback=f.anchors<12;
    const auto broad=broad_field(data,weights,f);f.displacement=broad;
    const auto support=detail::depth_support(f,weights);
    const auto structure=refine_structure?detail::structure_matches(f,pg,rg,depth,opacity,matches,s,source_mask):
        std::vector<std::array<float,4>>(n);
    f.structure_anchors=std::count_if(structure.begin(),structure.end(),[](const auto& a){return a[3]>0;});
    detail::regularize(f,data,weights,support,pg,source_mask,structure);
    auto labels=surfaces(f,support);
    const auto fitted=f.displacement;std::vector<float> strength(n,1);
    auto safety_weights=weights;for(float& v:safety_weights)v+=.001f;
    const auto safe_base=broad_field(fitted,safety_weights,f,labels);
    for(int attempt=0;attempt<32;++attempt) {
        mesh(f,labels);bool safe=true;
        for(const auto& t:f.triangles) {
            const float area=detail::cross(t.source[0],t.source[1],t.source[2]);
            const float det=detail::cross(t.target[0],t.target[1],t.target[2])/area;
            bool unsafe=!std::isfinite(det)||det<.25f||det>4;
            for(int i=0;i<3;++i) {
                const int j=(i+1)%3;const float a=std::hypot(t.source[i][0]-t.source[j][0],t.source[i][1]-t.source[j][1]);
                const float b=std::hypot(t.target[i][0]-t.target[j][0],t.target[i][1]-t.target[j][1]);
                if(b>3*a||b<.25f*a)unsafe=true;
            }
            if(!unsafe)continue;
            safe=false;
            for(const auto& point:t.source) {
                const int gx=std::clamp((int)((point[0]+.5f)/f.spacing),0,f.columns-2),gy=std::clamp((int)((point[1]+.5f)/f.spacing),0,f.rows-2);
                for(int y=gy;y<=gy+1;++y)for(int x=gx;x<=gx+1;++x) {
                    const size_t k=(size_t)y*f.columns+x;
                    const float current=std::hypot(f.displacement[k][0]-safe_base[k][0],f.displacement[k][1]-safe_base[k][1]);
                    const float original=std::hypot(fitted[k][0]-safe_base[k][0],fitted[k][1]-safe_base[k][1]);
                    strength[k]=std::min(strength[k],.75f*current/std::max(.001f,original));
                }
            }
        }
        if(safe)break;
        auto smooth=strength;
        for(int y=0;y<f.rows;++y)for(int x=0;x<f.columns;++x) {
            const size_t k=(size_t)y*f.columns+x;
            for(int q:{x?(int)k-1:-1,x+1<f.columns?(int)k+1:-1,y?(int)k-f.columns:-1,y+1<f.rows?(int)k+f.columns:-1})
                if(q>=0&&labels[q]==labels[k])smooth[k]=std::min(smooth[k],.5f+.5f*strength[q]);
        }
        strength.swap(smooth);f.correction_scale=*std::min_element(strength.begin(),strength.end());
        for(size_t k=0;k<n;++k)for(int c=0;c<2;++c)f.displacement[k][c]=safe_base[k][c]+(fitted[k][c]-safe_base[k][c])*strength[k];
        if(attempt==31)f.fallback=true;
    }
    if(f.fallback&&f.structure_anchors>0) {
        auto conservative=fit(photo,render,depth,opacity,matches,s,source_mask,false);
        conservative.structure_fallback=true;return conservative;
    }
    if(f.fallback) {
        f.correction_scale=0;std::fill(f.displacement.begin(),f.displacement.end(),Vec{});std::fill(labels.begin(),labels.end(),0);mesh(f,labels);
    }
    if(!f.fallback)choose_strength(f,pg,rg,source_mask,safe_base,labels);
    index_field(f);f.valid.resize((size_t)w*h);
#pragma omp parallel for schedule(static)
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){float sx,sy;f.valid[(size_t)y*w+x]=f.source(x,y,sx,sy)?255:0;}
    for(int y=16;y<h-16;y+=40)for(int x=16;x<w-16;x+=40)
        if(detail::correlation(rg,pg,w,h,x,y,x,y,3)>-1)f.audit.push_back({(float)x,(float)y});
    return f;
}

Audit audit(const Image& photo,const Image& corrected,const Image& render,const Field& f) {
    Audit out;const auto pg=detail::gray(photo),cg=detail::gray(corrected),rg=detail::gray(render);
    for(auto p:f.audit) {
        bool valid=true;
        for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i)valid&=f.valid[(size_t)((int)p[1]+j)*f.width+(int)p[0]+i]!=0;
        if(!valid){++out.invalid;continue;}
        const float a=detail::correlation(rg,pg,f.width,f.height,p[0],p[1],p[0],p[1],3);
        const float b=detail::correlation(rg,cg,f.width,f.height,p[0],p[1],p[0],p[1],3);
        if(a<-1||b<-1){++out.invalid;continue;}
        out.before+=a;out.after+=b;++out.count;
        const int k=(int)p[1]*f.width+(int)p[0];
        for(int offset:{1,f.width}) {
            out.gradient_before+=std::abs((pg[k+offset]-pg[k-offset])-(rg[k+offset]-rg[k-offset]));
            out.gradient_after+=std::abs((cg[k+offset]-cg[k-offset])-(rg[k+offset]-rg[k-offset]));
        }
    }
    if(out.count){out.before/=out.count;out.after/=out.count;out.gradient_before/=out.count*2;out.gradient_after/=out.count*2;}return out;
}
}  // namespace spirula::align
