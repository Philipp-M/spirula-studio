#include "app/Tools.h"
#include "i18n/catalog/Align.h"

#include <cstdio>
#include <exception>
#include <stdexcept>

#ifdef SS_HAVE_ROMA
#include "align/Report.h"
#include "app/cli/AlignCompare.h"
#include "align/Warp.h"
#include "align/WarpMath.h"
#include "checkpoint/SplatPly.h"
#include "core/Camera.h"
#include "core/LanczosResize.h"
#include "core/Sha256.h"
#include "data/DatasetParser.h"
#include "data/ImageProbe.h"
#include "data/Json.h"
#include "data/JsonWrite.h"
#include "data/SparseEdit.h"
#include "engine/Engine.h"
#include "external/stb_image.h"
#include "nn/io/Fetch.h"
#include "nn/io/Image.h"
#include "roma/model/Fetch.h"
#include "roma/PredictionFile.h"
#include "sfm/core/Pose.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
namespace fs=std::filesystem;
namespace A=spirula::align;
namespace M=spirula::i18n::msg::align;
using spirula::i18n::format;

struct Options {
    fs::path data,reference,output,checkpoint,reuse_renders,metrics;
    std::vector<fs::path> comparisons;
    size_t sample=10;
    uint32_t seed=20261031;
    int resolution=0;
    int jpeg_quality=95;
    bool all=false,resume=false,render_only=false,keep_alignment_data=false,review=false,pipeline=true,refine_cameras=true,residual_warp=true;
    std::string primitive="3dgs";
    std::string precision="float32";
    std::string image_format="png";
    A::Settings warp;
};

void help() {
    std::printf("%s\n\n%s --data DATASET --reference MODEL --output DIRECTORY\n\n%s\n\n",
        M::title.get(),app::program_name().c_str(),M::help.get());
    std::printf("%s\n\n",M::review_help.get());
    std::puts("--sample 10 | --all\n--seed 20261031\n--max-resolution PIXELS\n--checkpoint PATH\n--primitive 3dgs|mip|3dgut\n--precision float32|mixed|auto\n--image-format png|jpeg\n--jpeg-quality 95\n--reuse-renders PREVIOUS_OUTPUT\n--max-displacement 0\n--min-overlap 0.7\n--max-cycle 1.5\n--min-correlation 0.75\n--review\n--keep-alignment-data\n--camera-only | --warp-only\n--refine-cameras\n--residual-warp\n--metrics PATH\n--no-pipeline\n--resume\n--render-only\n--compare RUN --compare RUN --output COMPARISON");
}

double number(const std::string& text) {
    size_t n=0; const double v=std::stod(text,&n);
    if (n!=text.size() || !std::isfinite(v)) throw std::runtime_error("invalid numeric argument: "+text);
    return v;
}

Options parse(int argc,char** argv) {
    Options o;bool sampled=false,conversion_option=false;
    for (int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key!="--compare"&&key!="--output")conversion_option=true;
        if (key=="--review") {o.review=true;continue;}
        if (key=="--keep-alignment-data") {o.keep_alignment_data=true;continue;}
        if (key=="--refine-cameras") {o.refine_cameras=true;continue;}
        if (key=="--residual-warp") {o.residual_warp=true;continue;}
        if (key=="--camera-only") {o.refine_cameras=true;o.residual_warp=false;continue;}
        if (key=="--warp-only") {o.refine_cameras=false;o.residual_warp=false;continue;}
        if (key=="--no-pipeline") {o.pipeline=false;continue;}
        if (key=="--all") {o.all=true;continue;}
        if (key=="--resume") {o.resume=true;continue;}
        if (key=="--render-only") {o.render_only=true;continue;}
        if (i+1==argc) throw std::runtime_error("missing value for "+key);
        const std::string v=argv[++i];
        if (key=="--data") o.data=v;
        else if (key=="--reference") o.reference=v;
        else if (key=="--output") o.output=v;
        else if (key=="--metrics") o.metrics=fs::weakly_canonical(fs::absolute(v));
        else if (key=="--checkpoint") o.checkpoint=v;
        else if (key=="--reuse-renders") o.reuse_renders=fs::canonical(v);
        else if (key=="--compare") o.comparisons.push_back(fs::canonical(v));
        else if (key=="--precision") o.precision=v;
        else if (key=="--primitive") o.primitive=v;
        else if (key=="--image-format") o.image_format=v;
        else if (key=="--sample" || key=="--seed" || key=="--max-resolution" || key=="--jpeg-quality") {
            const double n=number(v);
            if (n<0 || std::floor(n)!=n || n>4294967295.0) throw std::runtime_error("invalid integer for "+key);
            if (key=="--sample") {o.sample=(size_t)n;sampled=true;}
            if (key=="--seed") o.seed=(uint32_t)n;
            if (key=="--max-resolution") {if(n<128||n>std::numeric_limits<int>::max())throw std::runtime_error("resolution must be 128..2147483647");o.resolution=(int)n;}
            if (key=="--jpeg-quality") {if(n<1||n>100)throw std::runtime_error("JPEG quality must be 1..100");o.jpeg_quality=(int)n;}
        } else if (key=="--max-displacement") o.warp.max_displacement=(float)number(v);
        else if (key=="--min-overlap") o.warp.min_overlap=(float)number(v);
        else if (key=="--max-cycle") o.warp.max_cycle=(float)number(v);
        else if (key=="--min-correlation") o.warp.min_correlation=(float)number(v);
        else throw std::runtime_error("unknown image-alignment option: "+key);
    }
    if(o.keep_alignment_data)o.review=true;
    if(o.image_format=="jpg")o.image_format="jpeg";
    if(o.residual_warp&&!o.refine_cameras)throw std::runtime_error("--residual-warp requires --refine-cameras");
    if(o.refine_cameras){if(o.resume||o.render_only)throw std::runtime_error("camera refinement requires a fresh complete run");}
    if(o.image_format!="png"&&o.image_format!="jpeg")throw std::runtime_error("image format must be png, jpeg or jpg");
    if(!o.comparisons.empty()) {
        if(o.comparisons.size()<2||o.output.empty()||!o.data.empty()||!o.reference.empty()||o.all||o.resume||o.render_only||sampled||conversion_option)
            throw std::runtime_error("use --compare RUN at least twice with a fresh --output and no dataset conversion options");
        return o;
    }
    if(o.resume&&!o.keep_alignment_data)throw std::runtime_error("--resume requires --keep-alignment-data");
    if(o.render_only&&!o.review)throw std::runtime_error("--render-only requires --review or --keep-alignment-data");
    if (o.data.empty() || o.reference.empty() || o.output.empty() || !o.sample ||
        (o.all && sampled) || !(o.warp.min_overlap>0 && o.warp.min_overlap<=1) ||
        !(o.warp.max_cycle>0) || o.warp.max_displacement<0 ||
        !(o.warp.min_correlation>0 && o.warp.min_correlation<=1) ||
        (o.primitive!="3dgs" && o.primitive!="mip" && o.primitive!="3dgut") ||
        (o.precision!="float32" && o.precision!="mixed" && o.precision!="auto"))
        throw std::runtime_error("invalid image-alignment options; see --help");
    o.data=fs::canonical(o.data);o.reference=fs::canonical(o.reference);
    return o;
}

std::string stamp(const fs::path& p) {
    return p.string()+":"+std::to_string(fs::file_size(p))+":"+std::to_string(fs::last_write_time(p).time_since_epoch().count());
}

void text_file(const fs::path& p,const std::string& contents) {
    std::ofstream out(p);out<<contents;if(!out)throw std::runtime_error("cannot write "+p.string());
}

TorchTensorView tv(void* data,std::initializer_list<int64_t> shape) {
    return {(int64_t)data,4,std::vector<int64_t>(shape)};
}

nn::Image byte_image(const A::Image& a) {
    nn::Image out;out.width=a.width;out.height=a.height;out.data.resize(a.rgb.size());
    for(size_t i=0;i<out.data.size();++i)out.data[i]=(uint8_t)std::clamp(std::lround(a.rgb[i]*255),0l,255l);
    return out;
}

A::Image resized(const nn::Image& a,int w,int h) {
    nn::Image b;b.width=w;b.height=h;b.data.resize((size_t)w*h*3);
    spirula::resize_lanczos3<uint8_t,3>(a.data.data(),a.height,a.width,b.data.data(),h,w);
    A::Image out;out.width=w;out.height=h;out.rgb.resize(b.data.size());
    for(size_t i=0;i<b.data.size();++i)out.rgb[i]=b.data[i]/255.0f;
    return out;
}

void save_image(const A::Image& a,const fs::path& p) {
    if(!nn::save_image(byte_image(a),p.string(),-1))throw std::runtime_error("cannot save "+p.string());
}

std::vector<uint8_t> source_mask(const fs::path& photo,const std::string& mask,int w,int h) {
    std::vector<uint8_t> out;
    int mw=0,mh=0,c=0;
    if(!mask.empty()) {
        auto* pixels=stbi_load(mask.c_str(),&mw,&mh,&c,1);
        if(!pixels)throw std::runtime_error("cannot decode original mask: "+mask);
        out.assign((size_t)w*h,255);
#pragma omp parallel for schedule(static)
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const int x0=x*(int64_t)mw/w,x1=std::min(mw-1,(int)(((int64_t)(x+1)*mw+w-1)/w)-1);
            const int y0=y*(int64_t)mh/h,y1=std::min(mh-1,(int)(((int64_t)(y+1)*mh+h-1)/h)-1);
            for(int j=y0;j<=y1;++j)for(int i=x0;i<=x1;++i)if(pixels[(size_t)j*mw+i]<128)out[(size_t)y*w+x]=0;
        }
        stbi_image_free(pixels);
    }
    if(stbi_info(photo.string().c_str(),&mw,&mh,&c) && c==4) {
        auto* rgba=stbi_load(photo.string().c_str(),&mw,&mh,&c,4);
        if(!rgba || mw!=w || mh!=h) {stbi_image_free(rgba);throw std::runtime_error("cannot decode source alpha");}
        if(out.empty())out.assign((size_t)w*h,255);
        for(size_t i=0;i<out.size();++i)if(rgba[i*4+3]<255)out[i]=0;
        stbi_image_free(rgba);
    }
    return out;
}

struct Cache {A::Image render;std::vector<float> depth,opacity;};

void write_cache(const Cache& c,const fs::path& p) {
    std::ofstream out(p,std::ios::binary);const int dims[2]={c.render.width,c.render.height};
    out.write((const char*)dims,sizeof(dims));
    for(const auto* v:{&c.render.rgb,&c.depth,&c.opacity})out.write((const char*)v->data(),v->size()*sizeof(float));
    if(!out)throw std::runtime_error("cannot save render cache");
}

void clamp_render(Cache& c) {
    for(float& value:c.render.rgb) {
        if(!std::isfinite(value))throw std::runtime_error("nonfinite rendered color");
        value=std::clamp(value,0.0f,1.0f);
    }
}

Cache read_cache(const fs::path& p) {
    Cache c;std::ifstream in(p,std::ios::binary);int dims[2]{};in.read((char*)dims,sizeof(dims));
    const uint64_t pixels=(uint64_t)std::max(0,dims[0])*std::max(0,dims[1]);
    if(!in||dims[0]<16||dims[1]<16||pixels>std::numeric_limits<int>::max()||
       fs::file_size(p)!=sizeof(dims)+pixels*5*sizeof(float))throw std::runtime_error("invalid render cache");
    c.render.width=dims[0];c.render.height=dims[1];const size_t n=(size_t)dims[0]*dims[1];
    c.render.rgb.resize(n*3);c.depth.resize(n);c.opacity.resize(n);
    for(auto* v:{&c.render.rgb,&c.depth,&c.opacity})in.read((char*)v->data(),v->size()*sizeof(float));
    if(!in)throw std::runtime_error("truncated render cache");
    clamp_render(c);
    return c;
}

spirula::roma::PairPrediction prediction(spirula::roma::Session& session,const Cache& cache,const A::Image& photo,
                                       const spirula::roma::MatchOptions& options,const fs::path& path,const fs::path& checkpoint,bool& cached) {
    cached=false;
    const auto file=path/"matches.roma",checksum=path/"matches.sha256";
    spirula::roma::PairPrediction p;std::ifstream hash(checksum);std::string expected;hash>>expected;
    if(!path.empty()&&fs::exists(file)&&expected.size()==64&&spirula::sha256_file(file.string())==expected) {
        std::ifstream in(file,std::ios::binary);char magic[8]{};in.read(magic,8);
        if(std::string(magic,8)=="SSROMA03"&&spirula::roma::read_prediction(in,p.forward,options.high_width,options.high_height,true)&&
           spirula::roma::read_prediction(in,p.backward,options.high_width,options.high_height,true)&&in.peek()==EOF){cached=true;return p;}
    }
    if(!session.loaded())session.load(checkpoint.string(),options.precision);
    return session.match(cache.render.rgb.data(),photo.width,photo.height,photo.rgb.data(),photo.width,photo.height,options);
}

void save_prediction(const spirula::roma::PairPrediction& p,const fs::path& path) {
    const auto file=path/"matches.roma",checksum=path/"matches.sha256";
    const auto part=path/"matches.part";
    {std::ofstream out(part,std::ios::binary);out.write("SSROMA03",8);
        spirula::roma::write_prediction(out,p.forward);spirula::roma::write_prediction(out,p.backward);
        out.flush();if(!out)throw std::runtime_error("cannot save raw RoMa predictions");}
    fs::rename(part,file);text_file(checksum,spirula::sha256_file(file.string())+"\n");
}

std::string matching_summary(const A::Field& f) {
    const auto& d=f.diagnostics;std::ostringstream json;
    json<<"{\"sampled\":"<<d.sampled<<",\"finite\":"<<d.finite<<",\"overlap\":"<<d.overlap<<",\"cycle\":"<<d.cycle
        <<",\"bounded\":"<<d.bounded<<",\"old_opacity\":"<<d.old_opacity<<",\"old_depth\":"<<d.old_depth
        <<",\"old_correlation\":"<<d.old_correlation<<",\"old_accepted\":"<<d.old_accepted<<",\"mean_overlap\":"<<d.mean_overlap
        <<",\"median_cycle\":"<<d.median_cycle<<",\"median_displacement\":"<<d.median_displacement<<'}';
    return json.str();
}

std::string diagnostics(const A::Field& f,const spirula::roma::PairPrediction& p,const A::Settings& settings,const fs::path& path,
                        const spirula::roma::PairPrediction* initial=nullptr) {
    const auto json=matching_summary(f);
    text_file(path/"matching.json",json);
    for(const std::string name:{"overlap","cycle","raw-flow","deformation","alignment-error-before","alignment-error-after"}) {
        const auto& map_matches=name=="alignment-error-before"&&initial?*initial:p;
        A::Image image;image.width=f.width;image.height=f.height;image.rgb.resize((size_t)f.width*f.height*3);
        for(int y=0;y<f.height;++y)for(int x=0;x<f.width;++x) {
            const size_t k=(size_t)y*f.width+x;float qx=0,qy=0,c=0,bx=0,by=0,bc=0;
            const bool good=A::detail::match(map_matches.forward,x,y,f.width,f.height,qx,qy,c);
            if(name=="overlap")for(int j=0;j<3;++j)image.rgb[3*k+j]=good?c:0;
            else if(name=="cycle") {
                const bool reverse=good&&A::detail::match(map_matches.backward,qx,qy,f.width,f.height,bx,by,bc);
                const float error=reverse?std::min(1.0f,std::hypot(bx-x,by-y)/5):1;
                image.rgb[3*k]=error;image.rgb[3*k+1]=1-error;
            }else if(name=="alignment-error-before"||name=="alignment-error-after") {
                const bool reliable=good&&c>=settings.min_overlap&&qx>=0&&qy>=0&&qx<=f.width-1&&qy<=f.height-1&&
                    A::detail::match(map_matches.backward,qx,qy,f.width,f.height,bx,by,bc)&&bc>=settings.min_overlap&&std::hypot(bx-x,by-y)<=settings.max_cycle;
                if(!reliable){for(int j=0;j<3;++j)image.rgb[3*k+j]=.15f;continue;}
                float sx=(float)x,sy=(float)y;
                if(name=="alignment-error-after"&&!f.source(x,y,sx,sy)){image.rgb[3*k]=image.rgb[3*k+2]=1;continue;}
                const float error=std::min(1.0f,std::hypot(qx-sx,qy-sy)/10);
                image.rgb[3*k]=error;image.rgb[3*k+1]=1-error;
            }else {
                const bool valid=name=="raw-flow"?good:f.source(x,y,qx,qy);
                if(valid){image.rgb[3*k]=std::clamp(.5f+(qx-x)/20,0.0f,1.0f);image.rgb[3*k+1]=std::clamp(.5f+(qy-y)/20,0.0f,1.0f);image.rgb[3*k+2]=.5f;}
            }
        }
        save_image(image,path/(name+".png"));
    }
    return json;
}

A::Audit previews(const nn::Image& corrected,const std::vector<uint8_t>& mask,const A::Coverage& coverage,
                  const A::Field& field,const A::Image& photo,const A::Image& render,const fs::path& path) {
    const bool save=!path.empty();
    const int w=field.width,h=field.height;auto preview=resized(corrected,w,h);
    nn::Image display,bad;display.width=bad.width=w;display.height=bad.height=h;
    display.channels=4;if(save){display.data.resize((size_t)w*h*4);bad.data.resize((size_t)w*h*3);}
    A::Field audit_field;audit_field.width=w;audit_field.height=h;audit_field.audit=field.audit;audit_field.valid.resize((size_t)w*h);
    std::vector<uint8_t> coverage_preview(save?(size_t)w*h:0);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const int x0=(int)((int64_t)x*corrected.width/w),x1=(int)(((int64_t)(x+1)*corrected.width+w-1)/w);
        const int y0=(int)((int64_t)y*corrected.height/h),y1=(int)(((int64_t)(y+1)*corrected.height+h-1)/h);
        size_t usable=0,covered=0,masked=0,total=0;
        for(int yy=y0;yy<std::min(y1,corrected.height);++yy)for(int xx=x0;xx<std::min(x1,corrected.width);++xx) {
            const size_t j=(size_t)yy*corrected.width+xx;++total;usable+=mask[j]!=0;
            covered+=coverage.reason[j]!=1;masked+=coverage.reason[j]==2;
        }
        const size_t p=(size_t)y*w+x;
        audit_field.valid[p]=usable==total?255:0;
        if(save) {
            for(int c=0;c<3;++c)display.data[4*p+c]=(uint8_t)std::clamp(std::lround(preview.rgb[3*p+c]*255),0l,255l);
            display.data[4*p+3]=(uint8_t)(255*usable/total);coverage_preview[p]=(uint8_t)(255*covered/total);
            bad.data[3*p]=(uint8_t)(255*(total-covered)/total);bad.data[3*p+1]=0;bad.data[3*p+2]=(uint8_t)(255*masked/total);
        }
    }
    if(save&&(!nn::save_image(display,(path/"corrected.png").string(),-1)||!nn::save_image(bad,(path/"ignored.png").string(),-1)||
       !nn::save_gray_png(coverage_preview.data(),w,h,(path/"coverage.png").string())))throw std::runtime_error("cannot save alignment previews");
    return A::audit(photo,preview,render,audit_field);
}

int run(const Options& o) {
    if(!o.comparisons.empty()){A::compare_runs(o.output,o.comparisons);return 0;}
    const auto dest=fs::weakly_canonical(fs::absolute(o.output));
    if(!o.metrics.empty()) {
        for(const auto& dir:{o.data,dest}) {
            const auto relative=o.metrics.lexically_relative(dir);
            if(relative.empty()||(!relative.is_absolute()&&*relative.begin()!=".."))
                throw std::runtime_error("metrics must be outside the source and output datasets");
        }
        if(fs::exists(o.metrics))throw std::runtime_error("metrics path already exists");
    }
    const auto rel=dest.lexically_relative(o.data);
    if(rel.empty() || (!rel.is_absolute() && *rel.begin()!=".."))throw std::runtime_error("output must be outside the read-only dataset");
    if(fs::exists(dest) && !o.resume)throw std::runtime_error("output already exists; use a fresh directory or --resume");
    if(!fs::exists(dest) && o.resume)throw std::runtime_error("--resume requires an existing alignment output");
    DatasetParserConfig cfg;cfg.probe_image_size=probe_image_size;
    ParsedDataset ds=parse_dataset(o.data.string(),cfg,"");
    if(!ds.num_cameras)throw std::runtime_error("dataset has no registered views");
    std::map<fs::path,ColmapImage> source_poses;std::map<int32_t,ColmapImage> corrected_poses;
    spirula::SparseImageObservations observations;
    if(o.refine_cameras) {
        bool text=false;const auto model=find_colmap_model(o.data.string(),"",&text);
        if(model.empty())throw std::runtime_error("camera refinement requires COLMAP; use --warp-only for other datasets");
        if(fs::exists(fs::path(model)/"frames.bin")||fs::exists(fs::path(model)/"frames.txt"))throw std::runtime_error("camera refinement does not support rig frames");
        for(const auto& [id,pose]:text?read_images_text(model):read_images_binary(model)) {
            auto path=(o.data/"images"/pose.name).lexically_normal();
            if(!fs::exists(path)&&fs::exists(o.data/pose.name))path=(o.data/pose.name).lexically_normal();
            source_poses.emplace(path,pose);
        }
    }
    std::vector<size_t> selected((size_t)ds.num_cameras);std::iota(selected.begin(),selected.end(),0);
    if(!o.all){std::mt19937 rng(o.seed);for(size_t i=selected.size();i>1;--i)std::swap(selected[i-1],selected[std::uniform_int_distribution<size_t>(0,i-1)(rng)]);selected.resize(std::min(o.sample,selected.size()));std::sort(selected.begin(),selected.end());}
    if(o.residual_warp) {
        std::vector<int32_t> ids;
        for(size_t i:selected)ids.push_back(source_poses.at(fs::path(ds.image_filenames[i]).lexically_normal()).image_id);
        observations=spirula::read_sparse_observations(o.data.string(),ids);
    }
    std::ostringstream identity;identity<<"image-align-v2\n"<<stamp(o.reference)<<'\n'<<o.data.string()<<'\n'<<o.seed<<' '<<o.resolution<<' '<<o.primitive
        <<' '<<o.warp.min_overlap<<' '<<o.warp.max_cycle<<' '<<o.warp.max_displacement<<' '<<o.warp.min_correlation<<'\n';
    const auto checkpoint=o.checkpoint.empty()?fs::path(nn::cached_path(spirula::roma::kOfficialCheckpoint)):fs::canonical(o.checkpoint);
    if(fs::exists(checkpoint))identity<<stamp(checkpoint)<<'\n';
    const auto post=bake_post_split(ds,false,false);
    std::set<fs::path> names;
    std::vector<fs::path> relative,dirs;
    for(size_t i:selected) {
        const auto name=fs::path(ds.image_filenames[i]).lexically_relative(o.data/"images");
        if(name.empty() || name.is_absolute() || *name.begin()=="..")throw std::runtime_error("image is not under the dataset images directory");
        auto output_name=name;if(!o.refine_cameras||o.residual_warp)output_name.replace_extension(o.image_format=="jpeg"?".jpg":".png");
        if(!names.insert(output_name).second)throw std::runtime_error("duplicate corrected image path");
        if(!ds.redistort.empty() && ds.redistort[i].source_model>=0)throw std::runtime_error("alignment export requires an exactly supported source lens");
        relative.push_back(output_name);dirs.push_back(fs::path("review")/std::to_string(i));
        identity<<stamp(ds.image_filenames[i])<<'\n';
        if(!ds.mask_filenames.empty()&&!ds.mask_filenames[i].empty())identity<<stamp(ds.mask_filenames[i])<<'\n';
        identity<<i<<' '<<ds.widths[i]<<' '<<ds.heights[i]<<' '<<ds.camera_models[i]<<' '<<ds.camera_distortions[i]<<'\n';
        for(int j=0;j<12;++j)identity<<std::setprecision(12)<<ds.c2w[i*12+j]<<' ';identity<<'\n';
        for(int j=0;j<4;++j)identity<<ds.intrins[i*4+j]<<' ';identity<<'\n';
        for(int j=0;j<8;++j)identity<<ds.dist_coeffs[i*8+j]<<' ';identity<<'\n';
    }
    const auto render_key=identity.str();
    identity<<"warp-version=2 precision="<<o.precision<<'\n';
    if(o.refine_cameras)identity<<"refine-cameras=1\n";
    if(o.residual_warp)identity<<"residual-warp=1\n";
    if(o.image_format=="jpeg")identity<<"image-format=jpeg jpeg-quality="<<o.jpeg_quality<<'\n';
    const auto key=identity.str();
    if(o.resume) {std::ifstream in(dest/"provenance.txt");const std::string previous((std::istreambuf_iterator<char>(in)),{});if(previous!=key)throw std::runtime_error("resume provenance differs: source, model, selection or settings changed");}
    else {A::checked_output(o.data,dest);fs::create_directories(dest);if(o.review)text_file(dest/"provenance.txt",key);}
    std::ostringstream selection;selection<<"{\"population\":"<<ds.num_cameras<<",\"seed\":"<<o.seed<<",\"sources\":[";
    for(size_t j=0;j<selected.size();++j){if(j)selection<<',';selection<<json_quote(ds.image_filenames[selected[j]]);}
    selection<<"]}";if(o.review)text_file(dest/"selection.json",selection.str());
    if(o.review)for(const auto& d:dirs)fs::create_directories(dest/d);
    bool reuse_matches=false;
    if(!o.reuse_renders.empty()) {
        std::ifstream in(o.reuse_renders/"provenance.txt");const std::string previous((std::istreambuf_iterator<char>(in)),{});
        const auto begin=previous.find('\n');const auto end=previous.find("warp-version=");
        const auto expected=render_key.substr(render_key.find('\n')+1);
        if(begin==std::string::npos||previous.substr(begin+1,end==std::string::npos?end:end-begin-1)!=expected)
            throw std::runtime_error("render reuse provenance differs: model, selection, cameras or rendering settings changed");
        reuse_matches=previous.find("warp-version=2 precision="+o.precision+"\n")!=std::string::npos;
        if(o.keep_alignment_data)for(const auto& d:dirs)for(const char* name:{"render.bin","render.png"})
            if(!fs::exists(dest/d/name))fs::copy_file(o.reuse_renders/d/name,dest/d/name);
    }
    std::optional<spirula::SplatCloud> reference;
    bool scene_loaded=false;
    auto release_scene=[&](){if(scene_loaded){engine_scene_free(0);engine_reset();scene_loaded=false;}};
    auto view_size=[&](size_t i) {
        const double scale=o.resolution?std::min(1.0,(double)o.resolution/std::max(ds.widths[i],ds.heights[i])):1.0;
        return std::pair{std::max(16,(int)std::lround(ds.widths[i]*scale)),std::max(16,(int)std::lround(ds.heights[i]*scale))};
    };
    auto render_view=[&](size_t k,const spirula::Sim3* correction=nullptr) {
        if(!scene_loaded) {
            if(!reference)reference=spirula::read_splat_ply(o.reference.string());
            auto& cloud=*reference;engine_reset();
            engine_scene_set_data_3dgs(0,cloud.num,tv(cloud.means.data(),{cloud.num,3}),tv(cloud.quats.data(),{cloud.num,4}),
                tv(cloud.scales.data(),{cloud.num,3}),tv(cloud.opacities.data(),{cloud.num,1}),tv(cloud.features_dc.data(),{cloud.num,3}),
                tv(cloud.features_sh.data(),{cloud.num,cloud.coeffs(),3}),cloud.sb_lobes);engine_scene_activate(0);scene_loaded=true;
        }
        const size_t i=selected[k];const auto [w,h]=view_size(i);
        float intr[4];for(int j=0;j<4;++j)intr[j]=post.intrins[i*4+j]*(j%2==0?(float)w/ds.widths[i]:(float)h/ds.heights[i]);
        float view[16];std::copy_n(post.viewmats.data()+i*16,16,view);
        if(correction)for(int r=0;r<3;++r)for(int col=0;col<4;++col) {
            double value=col==3?correction->t[r]:0;
            for(int j=0;j<3;++j)value+=correction->R[3*r+j]*post.viewmats[i*16+4*j+col];
            view[4*r+col]=(float)value;
        }
        set_camera_params(w,h,camera_model_to_string((CameraModelType)ds.camera_models[i]),
            camera_distortion_to_string((CameraDistortionType)ds.camera_distortions[i]),
            tv(view,{1,4,4}),tv(intr,{1,4}),tv((void*)(post.dist_coeffs.data()+i*8),{1,8}));
        forward_3dgs(o.primitive,reference->sh_degree,true,true,0);
        Cache c;c.render.width=w;c.render.height=h;c.render.rgb.resize((size_t)w*h*3);c.depth.resize((size_t)w*h);c.opacity.resize((size_t)w*h);
        const TorchTensorView null{0,4,{}};
        engine_copy_render_to_host(tv(c.render.rgb.data(),{1,h,w,3}),null,tv(c.opacity.data(),{1,h,w,1}),null,tv(c.depth.data(),{1,h,w,1}));
        for(auto& a:c.opacity)a=1-a;return c;
    };
    if(o.keep_alignment_data||o.render_only) {
        std::future<void> writing;
        for(size_t k=0;k<selected.size();++k) {
            if(fs::exists(dest/dirs[k]/"render.bin"))continue;
            auto cache=render_view(k);
            if(writing.valid())writing.get();
            auto write=[&,k,cache=std::move(cache)]() {
                if(o.keep_alignment_data)write_cache(cache,dest/dirs[k]/"render.bin");save_image(cache.render,dest/dirs[k]/"render.png");
                std::printf("%s\n",format(M::progress,{(long long)(k+1),(long long)selected.size(),ds.image_filenames[selected[k]]}).c_str());
            };
            if(o.pipeline)writing=std::async(std::launch::async,std::move(write));else write();
        }
        if(writing.valid())writing.get();
        release_scene();reference.reset();
    }
    if(o.render_only)return 0;
    spirula::roma::Session matcher;
    const auto precision=o.precision=="float32"?spirula::roma::InferencePrecision::Float32:o.precision=="mixed"?spirula::roma::InferencePrecision::Mixed:spirula::roma::InferencePrecision::Automatic;
    auto matching_options=[&](int w,int h) {
        auto options=spirula::roma::MatchOptions::preset("precise");
        options.low_width=800;options.low_height=std::max(16,(int)std::lround(800.0*h/w/16)*16);
        options.high_width=(w+3)/4*4;options.high_height=(h+3)/4*4;options.precision=precision;return options;
    };
    std::vector<std::string> records;size_t succeeded=0,retained=0;
    int cpu_threads=1;
#ifdef _OPENMP
    cpu_threads=std::max(1,omp_get_max_threads()-(o.pipeline?2:0));
#endif
    struct Input {
        nn::Image original;A::Image photo;std::vector<uint8_t> mask,working_mask;Cache cache;
        A::CameraAlignment alignment;std::optional<ColmapImage> corrected_pose;
        std::optional<spirula::roma::PairPrediction> initial_matches;
        double camera_audit_before=0,camera_audit_after=0,camera_coarse_before=0,camera_coarse_after=0;
        size_t camera_candidates=0;std::string camera_method="unchanged";
        A::MatchAudit camera_matches;
        int32_t recovery_seed=-1;
        double recovery_validation_error=0;
        std::vector<std::string> recovery_attempts;
    };
    auto prepare=[&](size_t k) {
#ifdef _OPENMP
        if(o.pipeline)omp_set_num_threads(1);
#endif
        Input input;const size_t i=selected[k];const auto [w,h]=view_size(i);
        input.original=nn::load_image(ds.image_filenames[i]);
        if(input.original.empty()||input.original.width!=ds.widths[i]||input.original.height!=ds.heights[i])
            throw std::runtime_error("source dimensions differ from parsed camera");
        input.photo=resized(input.original,w,h);
        input.mask=source_mask(ds.image_filenames[i],ds.mask_filenames.empty()?"":ds.mask_filenames[i],input.original.width,input.original.height);
        if(!input.mask.empty()) {
            input.working_mask.resize((size_t)w*h);
            for(int y=0;y<h;++y)for(int x=0;x<w;++x)
                input.working_mask[(size_t)y*w+x]=input.mask[(size_t)std::min(input.original.height-1,(int)((y+.5)*input.original.height/h))*input.original.width+std::min(input.original.width-1,(int)((x+.5)*input.original.width/w))];
        }
        const auto cache_path=o.keep_alignment_data?dest/dirs[k]/"render.bin":o.reuse_renders.empty()?fs::path{}:o.reuse_renders/dirs[k]/"render.bin";
        if(!cache_path.empty()&&fs::exists(cache_path)) {
            input.cache=read_cache(cache_path);
            if(input.cache.render.width!=w||input.cache.render.height!=h)throw std::runtime_error("cached render dimensions differ from selected resolution");
        }
        return input;
    };
    auto camera_for=[&](size_t i,int w,int h) {
        camhost::Camera camera;camera.width=w;camera.height=h;
        camera.model=ds.camera_models[i];camera.tier=ds.camera_distortions[i];
        camera.fx=post.intrins[i*4]*(double)w/ds.widths[i];camera.fy=post.intrins[i*4+1]*(double)h/ds.heights[i];
        camera.cx=post.intrins[i*4+2]*(double)w/ds.widths[i];camera.cy=post.intrins[i*4+3]*(double)h/ds.heights[i];
        std::copy_n(post.dist_coeffs.data()+i*8,8,camera.dist);
        return camera;
    };
    auto refine_view=[&](size_t k,Input& input,spirula::roma::PairPrediction& matches,bool cached,const spirula::Sim3& base=spirula::Sim3{},bool save=true) {
        const size_t i=selected[k];auto& cache=input.cache;const auto& photo=input.photo;
        const auto& working_mask=input.working_mask;const int w=photo.width,h=photo.height;
        const bool keep=o.keep_alignment_data&&save,review=o.review&&save;
        const auto path=review?dest/dirs[k]:fs::path{};
        if(keep&&!cached)save_prediction(matches,path);
        auto settings=o.warp;
        if(!base.is_identity()&&settings.max_displacement==0)settings.max_displacement=std::hypot((float)w,(float)h);
        input.alignment=A::refine_camera(cache.depth,cache.opacity,matches,settings,camera_for(i,w,h),working_mask,&photo,&cache.render);
        A::Field audit; audit.width=w;audit.height=h;
        if(o.review||!o.metrics.empty()) {
            audit.valid=working_mask.empty()?std::vector<uint8_t>((size_t)w*h,255):working_mask;
            const auto pg=A::detail::gray(photo),rg=A::detail::gray(cache.render);
            for(int y=16;y<h-16;y+=40)for(int x=16;x<w-16;x+=40)
                if(A::detail::correlation(rg,pg,w,h,x,y,x,y,3)>-1)audit.audit.push_back({(float)x,(float)y});
            input.camera_audit_before=A::audit(photo,photo,cache.render,audit).before;
        }
        const bool proposed=input.alignment.candidate||input.alignment.photometric_candidate||input.alignment.rotation_candidate;
        if(proposed) {
            matcher.unload();std::optional<Cache> best,geometric;spirula::Sim3 correction;double best_gain=0;
            std::optional<spirula::roma::PairPrediction> validated_matches;
            int method=0;
            const bool available[3]={input.alignment.candidate,input.alignment.photometric_candidate,input.alignment.rotation_candidate};
            const spirula::Sim3 corrections[3]={input.alignment.correction,input.alignment.photometric_correction,input.alignment.rotation_correction};
            for(int candidate_method=0;candidate_method<3;++candidate_method) {
                if(!available[candidate_method])continue;
                const auto delta=corrections[candidate_method]*base;
                auto candidate=render_view(k,&delta);clamp_render(candidate);++input.camera_candidates;
                const auto score=A::audit_camera(photo,cache.render,candidate.render,working_mask);
                if(score.accepted&&score.gain>best_gain) {
                    best=std::move(candidate);correction=delta;best_gain=score.gain;method=candidate_method;
                    input.camera_coarse_before=score.coarse_before;input.camera_coarse_after=score.coarse_after;
                } else if(candidate_method==0)geometric=std::move(candidate);
            }
            release_scene();
            if(!best&&geometric) {
                bool matched=false;
                auto fresh=prediction(matcher,*geometric,photo,matching_options(w,h),{},checkpoint,matched);
                input.camera_matches=A::audit_camera_matches(matches,fresh,w,h,settings,working_mask);
                if(input.camera_matches.accepted) {
                    best=std::move(geometric);correction=corrections[0]*base;method=0;
                    validated_matches=std::move(fresh);
                }
            }
            input.alignment.accepted=best.has_value();
            input.alignment.status=best?A::CameraFitStatus::ImageValidated:A::CameraFitStatus::ImageRejected;
            if(validated_matches)input.alignment.status=A::CameraFitStatus::MatchValidated;
            if(best) {
                input.alignment.correction=correction;
                if(method==1)input.alignment.error_after=input.alignment.photometric_error_after;
                if(method==2)input.alignment.error_after=input.alignment.rotation_error_after;
                input.camera_method=method==2?"image_rotation":method==1?"image_pyramid":"roma";
                auto pose=source_poses.at(fs::path(ds.image_filenames[i]).lexically_normal());
                sfm::Pose old_pose{sfm::quaternionToRotation(pose.qvec),{pose.tvec[0],pose.tvec[1],pose.tvec[2]}},delta;
                std::copy_n(correction.R,9,delta.R.begin());delta.t={correction.t[0],correction.t[1],correction.t[2]};
                const auto updated=sfm::composePose(delta,old_pose);
                pose.qvec=sfm::rotationToQuaternion(updated.R);pose.tvec={updated.t.x,updated.t.y,updated.t.z};
                if(review)save_image(cache.render,path/"initial-render.png");
                if(review&&!o.residual_warp)input.initial_matches=std::move(matches);else matches={};
                cache=std::move(*best);
                const auto refined_path=keep?path/"refined":fs::path{};
                if(review)save_image(cache.render,path/"render.png");
                if(keep) {
                    write_cache(cache,path/"refined-render.bin");fs::create_directories(refined_path);
                }
                bool refined_cached=false;
                matches=validated_matches?std::move(*validated_matches):prediction(matcher,cache,photo,matching_options(w,h),refined_path,checkpoint,refined_cached);
                if(keep&&!refined_cached)save_prediction(matches,refined_path);
                input.corrected_pose=pose;
            }
        }
        if(o.review||!o.metrics.empty())input.camera_audit_after=A::audit(photo,photo,cache.render,audit).before;
    };
    auto recover_view=[&](size_t k,Input& input,spirula::roma::PairPrediction& matches) {
        const size_t i=selected[k];const auto original=source_poses.find(fs::path(ds.image_filenames[i]).lexically_normal());
        if(original==source_poses.end())return;
        auto transform=[](const ColmapImage& pose) {
            spirula::Sim3 t;const auto r=sfm::quaternionToRotation(pose.qvec);
            std::copy(r.begin(),r.end(),t.R);std::copy(pose.tvec.begin(),pose.tvec.end(),t.t);return t;
        };
        const auto inverse=transform(original->second).inverse();
        struct Hypothesis {ColmapImage pose;spirula::Sim3 delta;Cache cache;double score;};
        std::vector<Hypothesis> hypotheses;auto left=original,right=std::next(original);
        matcher.unload();
        for(int distance=0;distance<64&&hypotheses.size()<4;++distance) {
            for(bool previous:{true,false}) {
                auto at=source_poses.end();
                if(previous&&left!=source_poses.begin())at=--left;
                if(!previous&&right!=source_poses.end())at=right++;
                if(at==source_poses.end())continue;
                const auto delta=transform(at->second)*inverse;if(delta.is_identity(1e-8))continue;
                bool duplicate=false;
                for(const auto& h:hypotheses)duplicate|=(delta*h.delta.inverse()).is_identity(1e-8);
                if(duplicate)continue;
                auto candidate=render_view(k,&delta);clamp_render(candidate);
                const auto quality=A::audit_camera(input.photo,candidate.render,input.photo,input.working_mask);
                const bool usable=quality.fine_count>=50&&quality.coarse_count>=50;
                std::ostringstream note;note<<"{\"seed\":"<<at->second.image_id<<",\"fine_ncc\":"<<quality.fine_before<<",\"coarse_ncc\":"<<quality.coarse_before<<'}';
                input.recovery_attempts.push_back(note.str());
                hypotheses.push_back({at->second,delta,std::move(candidate),usable?.5*(quality.fine_before+quality.coarse_before):-2});
                if(hypotheses.size()==4)break;
            }
        }
        release_scene();
        std::sort(hypotheses.begin(),hypotheses.end(),[](const auto& a,const auto& b){return a.score>b.score;});
        for(size_t j=0;j<hypotheses.size();++j) {
            auto& h=hypotheses[j];if(h.score<=-2)continue;
            Input trial;trial.photo=input.photo;trial.working_mask=input.working_mask;trial.cache=std::move(h.cache);
            bool cached=false;auto candidate_matches=prediction(matcher,trial.cache,trial.photo,matching_options(trial.photo.width,trial.photo.height),{},checkpoint,cached);
            refine_view(k,trial,candidate_matches,true,h.delta,false);
            std::ostringstream note;note<<"{\"seed\":"<<h.pose.image_id<<",\"samples\":"<<trial.alignment.samples<<",\"holdout\":"<<trial.alignment.holdout
                <<",\"before\":"<<trial.alignment.error_before<<",\"after\":"<<trial.alignment.error_after<<",\"refined\":"<<(trial.corrected_pose?"true":"false")<<'}';
            input.recovery_attempts.push_back(note.str());
            const auto camera=camera_for(i,trial.photo.width,trial.photo.height);
            auto validation=A::refine_camera(trial.cache.depth,trial.cache.opacity,candidate_matches,o.warp,camera,trial.working_mask);
            if(trial.corrected_pose&&validation.error_before>3&&validation.candidate) {
                const auto current=transform(*trial.corrected_pose)*inverse;
                refine_view(k,trial,candidate_matches,true,current,false);
                validation=A::refine_camera(trial.cache.depth,trial.cache.opacity,candidate_matches,o.warp,camera,trial.working_mask);
            }
            if(validation.samples<128||validation.holdout<10||validation.error_before>3)continue;
            if(!trial.corrected_pose) {
                auto pose=original->second;pose.qvec=h.pose.qvec;pose.tvec=h.pose.tvec;trial.corrected_pose=pose;
                trial.alignment.correction=h.delta;trial.alignment.error_after=trial.alignment.error_before;
            }
            trial.alignment.accepted=true;trial.alignment.status=A::CameraFitStatus::ImageValidated;
            const auto quality=A::audit_camera(input.photo,trial.cache.render,input.photo,input.working_mask);
            if(quality.fine_count<50||quality.coarse_count<50||quality.fine_before<.1||quality.coarse_before<.2)continue;
            const auto path=dest/dirs[k];
            if(o.review) {
                save_image(input.cache.render,path/"initial-render.png");save_image(trial.cache.render,path/"render.png");
                if(o.keep_alignment_data){write_cache(trial.cache,path/"refined-render.bin");fs::create_directories(path/"refined");save_prediction(candidate_matches,path/"refined");}
                if(!o.residual_warp)input.initial_matches=std::move(matches);
            }
            input.cache=std::move(trial.cache);matches=std::move(candidate_matches);
            input.alignment=trial.alignment;input.corrected_pose=trial.corrected_pose;
            input.camera_candidates+=trial.camera_candidates;input.camera_method="neighbor/"+trial.camera_method;
            input.recovery_seed=h.pose.image_id;
            input.recovery_validation_error=validation.error_before;
            input.camera_audit_after=trial.camera_audit_after;
            input.camera_coarse_before=trial.camera_coarse_before;input.camera_coarse_after=trial.camera_coarse_after;
            return;
        }
    };
    auto source_encoding=[&](size_t i) {
        auto ext=fs::path(ds.image_filenames[i]).extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char c){return (char)std::tolower(c);});
        return (o.refine_cameras&&!o.residual_warp) || (o.image_format=="jpeg"?(ext==".jpg"||ext==".jpeg"):ext==".png");
    };
    auto retain_source=[&](size_t k,Input& input) {
        const size_t i=selected[k];const int width=ds.widths[i],height=ds.heights[i];
        const auto image=dest/"images"/relative[k],mask_path=dest/"masks"/fs::path(relative[k].string()+".png");
        fs::create_directories(image.parent_path());fs::create_directories(mask_path.parent_path());
        if(source_encoding(i))fs::copy_file(ds.image_filenames[i],image,fs::copy_options::overwrite_existing);
        else {
            if(input.original.empty())input.original=nn::load_image(ds.image_filenames[i]);
            if(input.original.empty()||!nn::save_image(input.original,image.string(),o.image_format=="jpeg"?o.jpeg_quality:-1))
                throw std::runtime_error("cannot retain original photograph");
        }
        auto mask=input.mask;
        if(mask.empty())mask=source_mask(ds.image_filenames[i],ds.mask_filenames.empty()?"":ds.mask_filenames[i],width,height);
        if(mask.empty())mask.assign((size_t)width*height,255);
        if(!nn::save_gray_png(mask.data(),width,height,mask_path.string()))throw std::runtime_error("cannot retain source mask");
        if(o.refine_cameras) {
            auto pose=input.corrected_pose.value_or(source_poses.at(fs::path(ds.image_filenames[i]).lexically_normal()));
            pose.name=relative[k].generic_string();corrected_poses[pose.image_id]=pose;
        }
        return (size_t)std::count(mask.begin(),mask.end(),uint8_t(255));
    };
    auto finish=[&](size_t k,Input input,spirula::roma::PairPrediction matches,bool matches_cached,std::exception_ptr failure) {
#ifdef _OPENMP
        omp_set_num_threads(cpu_threads);
#endif
        const size_t i=selected[k];const auto path=o.review?dest/dirs[k]:fs::path{};
        std::ostringstream record;record<<"{\"source\":"<<json_quote(ds.image_filenames[i])<<",\"preview\":"<<json_quote(dirs[k].generic_string());
        try {
            if(failure)std::rethrow_exception(failure);
            auto& cache=input.cache;auto& original=input.original;auto& photo=input.photo;const auto& input_mask=input.mask;
            const int w=cache.render.width,h=cache.render.height;
            if(o.review) {
                if(o.keep_alignment_data&&!o.refine_cameras&&!matches_cached)save_prediction(matches,path);
                save_image(photo,path/"original.png");
            }
            const auto& working_mask=input.working_mask;
            const auto& alignment=input.alignment;const auto& initial_matches=input.initial_matches;
            auto corrected_pose=input.corrected_pose;
            const auto camera_audit_before=input.camera_audit_before,camera_audit_after=input.camera_audit_after;
            auto field=A::fit(photo,cache.render,cache.depth,cache.opacity,matches,o.warp,working_mask);
            if(o.refine_cameras&&!o.residual_warp) {
                A::use_identity(field);field.fallback=!alignment.accepted;
                if(!working_mask.empty())field.valid=working_mask;
            }
            nn::Image corrected;corrected.width=original.width;corrected.height=original.height;
            std::vector<uint8_t> mask;A::Coverage coverage;
            A::resample(original.data,original.width,original.height,input_mask,field,corrected.data,mask,&coverage);
            auto score=previews(corrected,mask,coverage,field,photo,cache.render,path);
            const auto candidate=score;std::string fallback=o.refine_cameras&&!o.residual_warp?(alignment.accepted?"":A::camera_fit_status(alignment.status)):
                field.fallback?(field.anchors<12?"insufficient_matching_support":"unresolved_deformation"):"";
            if((!o.refine_cameras||o.residual_warp)&&fallback.empty()&&(score.count<10||score.after<=score.before+.0001))fallback="no_measured_alignment_improvement";
            if(!fallback.empty()&&!field.fallback) {
                if(o.review)fs::copy_file(path/"corrected.png",path/"candidate.png",fs::copy_options::overwrite_existing);
                if(o.keep_alignment_data)A::save_field(field,path/"candidate-warp.bin");
                A::use_identity(field);
                corrected.data=original.data;mask=input_mask.empty()?std::vector<uint8_t>((size_t)corrected.width*corrected.height,255):input_mask;
                coverage.pixels=coverage.covered=mask.size();coverage.masked=0;coverage.reason.resize(mask.size());
                for(size_t j=0;j<mask.size();++j){coverage.reason[j]=mask[j]?0:2;coverage.masked+=!mask[j];}
                score=previews(corrected,mask,coverage,field,photo,cache.render,path);
            }
            const auto alignment_score=score;
            const auto image=fs::path("images")/relative[k],mask_name=fs::path("masks")/fs::path(relative[k].string()+".png");
            fs::create_directories((dest/image).parent_path());fs::create_directories((dest/mask_name).parent_path());
            if(((o.refine_cameras&&!o.residual_warp)||field.fallback)&&source_encoding(i))fs::copy_file(ds.image_filenames[i],dest/image);
            else if(!nn::save_image(corrected,(dest/image).string(),o.image_format=="jpeg"?o.jpeg_quality:-1))throw std::runtime_error("cannot save native corrected image");
            if(!nn::save_gray_png(mask.data(),corrected.width,corrected.height,(dest/mask_name).string()))throw std::runtime_error("cannot save native mask");
            if((!o.refine_cameras||o.residual_warp)&&o.image_format=="jpeg"&&(o.review||!o.metrics.empty())) {
                corrected=nn::load_image((dest/image).string());
                if(corrected.empty()||corrected.width!=original.width||corrected.height!=original.height)
                    throw std::runtime_error("exported JPEG cannot be decoded at native dimensions");
                score=previews(corrected,mask,coverage,field,photo,cache.render,path);
            }
            const auto matching=o.review?diagnostics(field,matches,o.warp,path,!o.residual_warp&&initial_matches?&*initial_matches:nullptr):matching_summary(field);matches={};original={};
            if(o.keep_alignment_data)A::save_field(field,path/"warp.bin");
            const size_t count=coverage.covered-coverage.masked;
            if(count!=(size_t)std::count(mask.begin(),mask.end(),uint8_t(255)))throw std::runtime_error("coverage accounting differs from training mask");
            record<<",\"preview_available\":"<<(o.review?"true":"false")<<",\"width\":"<<corrected.width<<",\"height\":"<<corrected.height<<",\"anchors\":"<<field.anchors
                <<",\"valid_fraction\":"<<(double)count/mask.size()<<",\"coverage_fraction\":"<<(double)coverage.covered/coverage.pixels
                <<",\"source_mask_loss_fraction\":"<<(double)coverage.masked/coverage.pixels<<",\"coverage_loss_pixels\":"<<coverage.pixels-coverage.covered
                <<",\"source_mask_loss_pixels\":"<<coverage.masked<<",\"confidence_loss_pixels\":0,\"correction_scale\":"<<field.correction_scale
                <<",\"warp_available\":"<<(o.keep_alignment_data?"true":"false")
                <<",\"alignment_error_maps\":"<<(o.review?"true":"false")<<",\"pose_prior\":"<<(alignment.accepted?"true":"false")
                <<",\"camera_status\":"<<json_quote(o.refine_cameras?A::camera_fit_status(alignment.status):"not_requested")<<",\"pose_samples\":"<<alignment.samples<<",\"pose_holdout\":"<<alignment.holdout
                <<",\"recovery_seed_image\":"<<input.recovery_seed<<",\"recovery_validation_error\":"<<input.recovery_validation_error
                <<",\"camera_method\":"<<json_quote(input.camera_method)<<",\"camera_candidates\":"<<input.camera_candidates
                <<",\"camera_match_count\":"<<input.camera_matches.common_count
                <<",\"camera_match_before_count\":"<<input.camera_matches.before_count<<",\"camera_match_after_count\":"<<input.camera_matches.after_count
                <<",\"camera_match_before\":"<<input.camera_matches.before<<",\"camera_match_after\":"<<input.camera_matches.after
                <<",\"camera_match_supported_before\":"<<input.camera_matches.supported_before<<",\"camera_match_supported_after\":"<<input.camera_matches.supported_after
                <<",\"camera_coarse_before\":"<<input.camera_coarse_before<<",\"camera_coarse_after\":"<<input.camera_coarse_after
                <<",\"pose_error_before\":"<<alignment.error_before<<",\"pose_error_after\":"<<alignment.error_after
                <<",\"camera_refined\":"<<(corrected_pose?"true":"false")<<",\"camera_audit_before\":"<<camera_audit_before<<",\"camera_audit_after\":"<<camera_audit_after
                <<",\"audit_count\":"<<score.count<<",\"audit_invalid\":"<<score.invalid<<",\"audit_before\":"<<score.before<<",\"audit_after\":"<<score.after
                <<",\"alignment_audit_before\":"<<alignment_score.before<<",\"alignment_audit_after\":"<<alignment_score.after
                <<",\"gradient_error_before\":"<<score.gradient_before<<",\"gradient_error_after\":"<<score.gradient_after
                <<",\"candidate_audit_after\":"<<candidate.after<<",\"candidate_audit_count\":"<<candidate.count
                <<",\"matching\":"<<matching<<",\"fallback_reason\":"<<json_quote(fallback);
            if(o.review&&(!o.all||o.residual_warp)) {
                const auto native=dirs[k]/fs::path("original-native"+fs::path(ds.image_filenames[i]).extension().string());
                if(!fs::exists(dest/native))fs::copy_file(ds.image_filenames[i],dest/native);
                record<<",\"original_native\":"<<json_quote(native.generic_string());
            }
            record<<",\"recovery_attempts\":[";
            for(size_t j=0;j<input.recovery_attempts.size();++j){if(j)record<<',';record<<input.recovery_attempts[j];}
            record<<']';
            record<<",\"status\":"<<json_quote(fallback.empty()?"ok":"fallback")<<",\"image\":"<<json_quote(image.generic_string())<<",\"mask\":"<<json_quote(mask_name.generic_string())
                ;
            if(o.residual_warp) {
                if(!corrected_pose)corrected_pose=source_poses.at(fs::path(ds.image_filenames[i]).lexically_normal());
                corrected_pose->name=relative[k].generic_string();
                auto mapped=observations.at(corrected_pose->image_id);
                const auto outside=A::transport_observations(field,corrected.width,corrected.height,mapped);
                record<<",\"observations_outside_source\":"<<outside;
                observations.at(corrected_pose->image_id)=std::move(mapped);
            }
            if(corrected_pose)corrected_poses[corrected_pose->image_id]=*corrected_pose;
            if(!fallback.empty()) {
                ++retained;
                std::fprintf(stderr,"%s\n",format(M::unchanged,{ds.image_filenames[i],fallback}).c_str());
            }
            ++succeeded;
        }catch(const std::exception& e){
            record.str("");record.clear();
            record<<"{\"source\":"<<json_quote(ds.image_filenames[i])<<",\"preview\":"<<json_quote(dirs[k].generic_string())
                <<",\"preview_available\":false,\"error\":"<<json_quote(e.what());
            try {
                const auto kept=retain_source(k,input),pixels=(size_t)ds.widths[i]*ds.heights[i];
                record<<",\"status\":\"fallback\",\"fallback_reason\":"<<json_quote(e.what())
                    <<",\"width\":"<<ds.widths[i]<<",\"height\":"<<ds.heights[i]
                    <<",\"valid_fraction\":"<<(double)kept/pixels<<",\"coverage_loss_pixels\":0,\"source_mask_loss_pixels\":"<<pixels-kept
                    <<",\"confidence_loss_pixels\":0,\"camera_refined\":"<<(input.corrected_pose?"true":"false")
                    <<",\"image\":"<<json_quote((fs::path("images")/relative[k]).generic_string())
                    <<",\"mask\":"<<json_quote((fs::path("masks")/fs::path(relative[k].string()+".png")).generic_string());
                ++succeeded;++retained;
                std::fprintf(stderr,"%s\n",format(M::unchanged,{ds.image_filenames[i],e.what()}).c_str());
            }catch(const std::exception& output_error) {
                record<<",\"status\":\"failed\",\"output_error\":"<<json_quote(output_error.what());
                std::fprintf(stderr,"%s\n",format(M::error,{output_error.what()}).c_str());
            }
        }
        record<<'}';if(o.review||!o.metrics.empty())records.push_back(record.str());
        std::printf("%s\n",format(M::progress,{(long long)(k+1),(long long)selected.size(),ds.image_filenames[i]}).c_str());
        if(!o.review&&o.metrics.empty())return;
        std::ostringstream manifest;manifest<<"{\"version\":2,\"dataset\":"<<json_quote(o.data.string())<<",\"reference\":"<<json_quote(o.reference.string())
            <<",\"population\":"<<ds.num_cameras<<",\"seed\":"<<o.seed<<",\"selected_count\":"<<selected.size()
            <<",\"max_resolution\":"<<o.resolution<<",\"checkpoint\":"<<json_quote(checkpoint.string())
            <<",\"refine_cameras\":"<<(o.refine_cameras?"true":"false")<<",\"residual_warp\":"<<(o.residual_warp?"true":"false")
            <<",\"image_format\":"<<json_quote(o.refine_cameras&&!o.residual_warp?"source":o.image_format)<<",\"jpeg_quality\":"<<((!o.refine_cameras||o.residual_warp)&&o.image_format=="jpeg"?std::to_string(o.jpeg_quality):"null")
            <<",\"min_overlap\":"<<o.warp.min_overlap<<",\"max_cycle\":"<<o.warp.max_cycle
            <<",\"max_displacement\":"<<o.warp.max_displacement<<",\"min_correlation\":"<<o.warp.min_correlation
            <<",\"depth_jump\":"<<o.warp.depth_jump<<",\"spacing\":"<<o.warp.spacing
            <<",\"precision\":"<<json_quote(o.precision)<<",\"mask_mode\":\"ignore\",\"method\":"
            <<json_quote(o.residual_warp?"Rigid camera refinement in the fixed world frame, then RoMa residual correction; one resampling of original photographs; transported COLMAP observations":o.refine_cameras?"RoMa v2 and splat ray depth; rigid camera refinement in a fixed world frame; original photographs unchanged":"RoMa v2; complete source mesh, robust data fit, image-guided bending, supported depth boundaries; coverage and original-mask validity only")<<",\"views\":[";
        for(size_t j=0;j<records.size();++j){if(j)manifest<<',';manifest<<records[j];}manifest<<"]}";
        if(!o.metrics.empty())text_file(o.metrics,manifest.str());
        if(o.review){text_file(dest/"manifest.json",manifest.str());A::write_report(dest,manifest.str());}
    };
    const auto policy=o.pipeline?std::launch::async:std::launch::deferred;
    auto prefetched=std::async(policy,prepare,size_t(0));
    std::future<void> completed;
    for(size_t k=0;k<selected.size();++k) {
        Input input;spirula::roma::PairPrediction matches;bool matches_cached=false;std::exception_ptr failure;
        try {input=prefetched.get();}catch(...){failure=std::current_exception();}
        if(k+1<selected.size())prefetched=std::async(policy,prepare,k+1);
        if(!failure)try {
            if(input.cache.render.rgb.empty())input.cache=render_view(k);
            release_scene();clamp_render(input.cache);
            if(o.review&&!fs::exists(dest/dirs[k]/"render.png"))save_image(input.cache.render,dest/dirs[k]/"render.png");
            const int w=input.photo.width,h=input.photo.height;
            const auto options=matching_options(w,h);
            const auto match_path=o.keep_alignment_data?dest/dirs[k]:reuse_matches?o.reuse_renders/dirs[k]:fs::path{};
            matches=prediction(matcher,input.cache,input.photo,options,match_path,checkpoint,matches_cached);
            if(o.refine_cameras) {
                refine_view(k,input,matches,matches_cached);
                if(input.alignment.samples<64||input.alignment.holdout<10)recover_view(k,input,matches);
            }
        }catch(...){failure=std::current_exception();}
        if(!o.keep_alignment_data||o.refine_cameras){matcher.unload();release_scene();}
        if(completed.valid())completed.get();
        completed=std::async(policy,finish,k,std::move(input),std::move(matches),matches_cached,failure);
        if(!o.pipeline)completed.get();
    }
    if(completed.valid())completed.get();
    matcher.unload();
    if(o.refine_cameras) {
        int32_t cached_id=-1;size_t observation=0;
        auto transport=[&](int32_t id,double& x,double& y) {
            if(cached_id!=id){cached_id=id;observation=0;}
            const auto& point=observations.at(id).at(observation++);x=point[0];y=point[1];
        };
        spirula::sparse_write_camera_copy(o.data.string(),dest.string(),corrected_poses,
            o.residual_warp?spirula::SparseObservationTransform(transport):spirula::SparseObservationTransform{});
    }
    std::printf("%s\n",format(M::complete,{(o.review?dest/"index.html":dest).string(),(long long)succeeded,(long long)(selected.size()-succeeded),(long long)retained}).c_str());
    return succeeded==selected.size()?0:1;
}
}
#endif

int spirula_align_main(int argc,char** argv) {
    app::set_program_name(argc?argv[0]:nullptr,"spirula align-images");
    try {
#ifdef SS_HAVE_ROMA
        for(int i=1;i<argc;++i)if(std::string(argv[i])=="--help"){help();return 0;}
        return run(parse(argc,argv));
#else
        throw std::runtime_error("image alignment requires SS_BUILD_SAM=ON");
#endif
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",spirula::i18n::format(spirula::i18n::msg::align::error,{e.what()}).c_str());return 1;}
}
