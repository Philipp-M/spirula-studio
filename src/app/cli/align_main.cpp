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

#include <algorithm>
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
    fs::path data,reference,output,checkpoint,reuse_renders;
    std::vector<fs::path> comparisons;
    size_t sample=10;
    uint32_t seed=20261031;
    int resolution=0;
    int jpeg_quality=95;
    bool all=false,resume=false,render_only=false,keep_alignment_data=false,pipeline=true;
    std::string primitive="3dgs";
    std::string precision="float32";
    std::string image_format="png";
    A::Settings warp;
};

void help() {
    std::printf("%s\n\n%s --data DATASET --reference MODEL --output DIRECTORY\n\n%s\n\n",
        M::title.get(),app::program_name().c_str(),M::help.get());
    std::puts("--sample 10 | --all\n--seed 20261031\n--max-resolution PIXELS\n--checkpoint PATH\n--primitive 3dgs|mip|3dgut\n--precision float32|mixed|auto\n--image-format png|jpeg\n--jpeg-quality 95\n--reuse-renders PREVIOUS_OUTPUT\n--max-displacement 40\n--min-overlap 0.7\n--max-cycle 1.5\n--min-correlation 0.75\n--keep-alignment-data\n--no-pipeline\n--resume\n--render-only\n--compare RUN --compare RUN --output COMPARISON");
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
        if (key=="--keep-alignment-data") {o.keep_alignment_data=true;continue;}
        if (key=="--no-pipeline") {o.pipeline=false;continue;}
        if (key=="--all") {o.all=true;continue;}
        if (key=="--resume") {o.resume=true;continue;}
        if (key=="--render-only") {o.render_only=true;continue;}
        if (i+1==argc) throw std::runtime_error("missing value for "+key);
        const std::string v=argv[++i];
        if (key=="--data") o.data=v;
        else if (key=="--reference") o.reference=v;
        else if (key=="--output") o.output=v;
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
    if(o.image_format=="jpg")o.image_format="jpeg";
    if(o.image_format!="png"&&o.image_format!="jpeg")throw std::runtime_error("image format must be png, jpeg or jpg");
    if(!o.comparisons.empty()) {
        if(o.comparisons.size()<2||o.output.empty()||!o.data.empty()||!o.reference.empty()||o.all||o.resume||o.render_only||sampled||conversion_option)
            throw std::runtime_error("use --compare RUN at least twice with a fresh --output and no dataset conversion options");
        return o;
    }
    if((o.resume||o.render_only)&&!o.keep_alignment_data)throw std::runtime_error("--resume and --render-only require --keep-alignment-data");
    if (o.data.empty() || o.reference.empty() || o.output.empty() || !o.sample ||
        (o.all && sampled) || !(o.warp.min_overlap>0 && o.warp.min_overlap<=1) ||
        !(o.warp.max_cycle>0) || !(o.warp.max_displacement>0) ||
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

std::string diagnostics(const A::Field& f,const spirula::roma::PairPrediction& p,const fs::path& path) {
    const auto& d=f.diagnostics;std::ostringstream json;
    json<<"{\"sampled\":"<<d.sampled<<",\"finite\":"<<d.finite<<",\"overlap\":"<<d.overlap<<",\"cycle\":"<<d.cycle
        <<",\"bounded\":"<<d.bounded<<",\"old_opacity\":"<<d.old_opacity<<",\"old_depth\":"<<d.old_depth
        <<",\"old_correlation\":"<<d.old_correlation<<",\"old_accepted\":"<<d.old_accepted<<",\"mean_overlap\":"<<d.mean_overlap
        <<",\"median_cycle\":"<<d.median_cycle<<",\"median_displacement\":"<<d.median_displacement<<'}';
    text_file(path/"matching.json",json.str());
    for(const std::string name:{"overlap","cycle","raw-flow","deformation"}) {
        A::Image image;image.width=f.width;image.height=f.height;image.rgb.resize((size_t)f.width*f.height*3);
        for(int y=0;y<f.height;++y)for(int x=0;x<f.width;++x) {
            const size_t k=(size_t)y*f.width+x;float qx=0,qy=0,c=0,bx=0,by=0,bc=0;
            const bool good=A::detail::match(p.forward,x,y,f.width,f.height,qx,qy,c);
            if(name=="overlap")for(int j=0;j<3;++j)image.rgb[3*k+j]=good?c:0;
            else if(name=="cycle") {
                const bool reverse=good&&A::detail::match(p.backward,qx,qy,f.width,f.height,bx,by,bc);
                const float error=reverse?std::min(1.0f,std::hypot(bx-x,by-y)/5):1;
                image.rgb[3*k]=error;image.rgb[3*k+1]=1-error;
            }else {
                const bool valid=name=="raw-flow"?good:f.source(x,y,qx,qy);
                if(valid){image.rgb[3*k]=std::clamp(.5f+(qx-x)/20,0.0f,1.0f);image.rgb[3*k+1]=std::clamp(.5f+(qy-y)/20,0.0f,1.0f);image.rgb[3*k+2]=.5f;}
            }
        }
        save_image(image,path/(name+".png"));
    }
    return json.str();
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
    const auto rel=dest.lexically_relative(o.data);
    if(rel.empty() || (!rel.is_absolute() && *rel.begin()!=".."))throw std::runtime_error("output must be outside the read-only dataset");
    if(fs::exists(dest) && !o.resume)throw std::runtime_error("output already exists; use a fresh directory or --resume");
    if(!fs::exists(dest) && o.resume)throw std::runtime_error("--resume requires an existing alignment output");
    DatasetParserConfig cfg;cfg.probe_image_size=probe_image_size;
    ParsedDataset ds=parse_dataset(o.data.string(),cfg,"");
    if(!ds.num_cameras)throw std::runtime_error("dataset has no registered views");
    std::vector<size_t> selected((size_t)ds.num_cameras);std::iota(selected.begin(),selected.end(),0);
    if(!o.all){std::mt19937 rng(o.seed);for(size_t i=selected.size();i>1;--i)std::swap(selected[i-1],selected[std::uniform_int_distribution<size_t>(0,i-1)(rng)]);selected.resize(std::min(o.sample,selected.size()));std::sort(selected.begin(),selected.end());}
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
        auto output_name=name;output_name.replace_extension(o.image_format=="jpeg"?".jpg":".png");
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
    if(o.image_format=="jpeg")identity<<"image-format=jpeg jpeg-quality="<<o.jpeg_quality<<'\n';
    const auto key=identity.str();
    if(o.resume) {std::ifstream in(dest/"provenance.txt");const std::string previous((std::istreambuf_iterator<char>(in)),{});if(previous!=key)throw std::runtime_error("resume provenance differs: source, model, selection or settings changed");}
    else {A::checked_output(o.data,dest);fs::create_directories(dest);if(o.keep_alignment_data)text_file(dest/"provenance.txt",key);}
    std::ostringstream selection;selection<<"{\"population\":"<<ds.num_cameras<<",\"seed\":"<<o.seed<<",\"sources\":[";
    for(size_t j=0;j<selected.size();++j){if(j)selection<<',';selection<<json_quote(ds.image_filenames[selected[j]]);}
    selection<<"]}";if(o.keep_alignment_data)text_file(dest/"selection.json",selection.str());
    if(o.keep_alignment_data)for(const auto& d:dirs)fs::create_directories(dest/d);
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
    auto render_view=[&](size_t k) {
        if(!scene_loaded) {
            if(!reference)reference=spirula::read_splat_ply(o.reference.string());
            auto& cloud=*reference;engine_reset();
            engine_scene_set_data_3dgs(0,cloud.num,tv(cloud.means.data(),{cloud.num,3}),tv(cloud.quats.data(),{cloud.num,4}),
                tv(cloud.scales.data(),{cloud.num,3}),tv(cloud.opacities.data(),{cloud.num,1}),tv(cloud.features_dc.data(),{cloud.num,3}),
                tv(cloud.features_sh.data(),{cloud.num,cloud.coeffs(),3}),cloud.sb_lobes);engine_scene_activate(0);scene_loaded=true;
        }
        const size_t i=selected[k];const auto [w,h]=view_size(i);
        float intr[4];for(int j=0;j<4;++j)intr[j]=post.intrins[i*4+j]*(j%2==0?(float)w/ds.widths[i]:(float)h/ds.heights[i]);
        set_camera_params(w,h,camera_model_to_string((CameraModelType)ds.camera_models[i]),
            camera_distortion_to_string((CameraDistortionType)ds.camera_distortions[i]),
            tv((void*)(post.viewmats.data()+i*16),{1,4,4}),tv(intr,{1,4}),tv((void*)(post.dist_coeffs.data()+i*8),{1,8}));
        forward_3dgs(o.primitive,reference->sh_degree,true,true,0);
        Cache c;c.render.width=w;c.render.height=h;c.render.rgb.resize((size_t)w*h*3);c.depth.resize((size_t)w*h);c.opacity.resize((size_t)w*h);
        const TorchTensorView null{0,4,{}};
        engine_copy_render_to_host(tv(c.render.rgb.data(),{1,h,w,3}),null,tv(c.opacity.data(),{1,h,w,1}),null,tv(c.depth.data(),{1,h,w,1}));
        for(auto& a:c.opacity)a=1-a;return c;
    };
    if(o.keep_alignment_data) {
        std::future<void> writing;
        for(size_t k=0;k<selected.size();++k) {
            if(fs::exists(dest/dirs[k]/"render.bin"))continue;
            auto cache=render_view(k);
            if(writing.valid())writing.get();
            auto write=[&,k,cache=std::move(cache)]() {
                write_cache(cache,dest/dirs[k]/"render.bin");save_image(cache.render,dest/dirs[k]/"render.png");
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
    std::vector<std::string> records;size_t succeeded=0;
    int cpu_threads=1;
#ifdef _OPENMP
    cpu_threads=std::max(1,omp_get_max_threads()-(o.pipeline?2:0));
#endif
    struct Input {nn::Image original;A::Image photo;std::vector<uint8_t> mask;Cache cache;};
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
        const auto cache_path=o.keep_alignment_data?dest/dirs[k]/"render.bin":o.reuse_renders.empty()?fs::path{}:o.reuse_renders/dirs[k]/"render.bin";
        if(!cache_path.empty()&&fs::exists(cache_path)) {
            input.cache=read_cache(cache_path);
            if(input.cache.render.width!=w||input.cache.render.height!=h)throw std::runtime_error("cached render dimensions differ from selected resolution");
        }
        return input;
    };
    auto finish=[&](size_t k,Input input,spirula::roma::PairPrediction matches,bool matches_cached,std::exception_ptr failure) {
#ifdef _OPENMP
        omp_set_num_threads(cpu_threads);
#endif
        const size_t i=selected[k];const auto path=o.keep_alignment_data?dest/dirs[k]:fs::path{};
        std::ostringstream record;record<<"{\"source\":"<<json_quote(ds.image_filenames[i])<<",\"preview\":"<<json_quote(dirs[k].generic_string());
        try {
            if(failure)std::rethrow_exception(failure);
            auto& cache=input.cache;auto& original=input.original;auto& photo=input.photo;const auto& input_mask=input.mask;
            const int w=cache.render.width,h=cache.render.height;
            if(o.keep_alignment_data) {
                if(!matches_cached)save_prediction(matches,path);
                save_image(photo,path/"original.png");
            }
            std::vector<uint8_t> working_mask;
            if(!input_mask.empty()) {
                working_mask.resize((size_t)w*h);
                for(int y=0;y<h;++y)for(int x=0;x<w;++x)
                    working_mask[(size_t)y*w+x]=input_mask[(size_t)std::min(original.height-1,(int)((y+.5)*original.height/h))*original.width+std::min(original.width-1,(int)((x+.5)*original.width/w))];
            }
            auto field=A::fit(photo,cache.render,cache.depth,cache.opacity,matches,o.warp,working_mask);
            nn::Image corrected;corrected.width=original.width;corrected.height=original.height;
            std::vector<uint8_t> mask;A::Coverage coverage;
            A::resample(original.data,original.width,original.height,input_mask,field,corrected.data,mask,&coverage);
            auto score=previews(corrected,mask,coverage,field,photo,cache.render,path);
            const auto candidate=score;std::string fallback=field.fallback?(field.anchors<12?"insufficient_matching_support":"unresolved_deformation"):"";
            if(fallback.empty()&&(score.count<10||score.after<=score.before+.0001))fallback="no_measured_alignment_improvement";
            if(!fallback.empty()&&!field.fallback) {
                if(o.keep_alignment_data){fs::copy_file(path/"corrected.png",path/"candidate.png",fs::copy_options::overwrite_existing);A::save_field(field,path/"candidate-warp.bin");}
                A::use_identity(field);
                corrected.data=original.data;mask=input_mask.empty()?std::vector<uint8_t>((size_t)corrected.width*corrected.height,255):input_mask;
                coverage.pixels=coverage.covered=mask.size();coverage.masked=0;coverage.reason.resize(mask.size());
                for(size_t j=0;j<mask.size();++j){coverage.reason[j]=mask[j]?0:2;coverage.masked+=!mask[j];}
                score=previews(corrected,mask,coverage,field,photo,cache.render,path);
            }
            const auto alignment_score=score;
            const auto image=fs::path("images")/relative[k],mask_name=fs::path("masks")/fs::path(relative[k].string()+".png");
            fs::create_directories((dest/image).parent_path());fs::create_directories((dest/mask_name).parent_path());
            if(!nn::save_image(corrected,(dest/image).string(),o.image_format=="jpeg"?o.jpeg_quality:-1)||
               !nn::save_gray_png(mask.data(),corrected.width,corrected.height,(dest/mask_name).string()))throw std::runtime_error("cannot save native corrected image or mask");
            if(o.image_format=="jpeg"&&o.keep_alignment_data) {
                corrected=nn::load_image((dest/image).string());
                if(corrected.empty()||corrected.width!=original.width||corrected.height!=original.height)
                    throw std::runtime_error("exported JPEG cannot be decoded at native dimensions");
                score=previews(corrected,mask,coverage,field,photo,cache.render,path);
            }
            const auto matching=o.keep_alignment_data?diagnostics(field,matches,path):"{}";matches={};original={};
            if(o.keep_alignment_data)A::save_field(field,path/"warp.bin");
            const size_t count=coverage.covered-coverage.masked;
            if(count!=(size_t)std::count(mask.begin(),mask.end(),uint8_t(255)))throw std::runtime_error("coverage accounting differs from training mask");
            record<<",\"preview_available\":true,\"width\":"<<corrected.width<<",\"height\":"<<corrected.height<<",\"anchors\":"<<field.anchors
                <<",\"valid_fraction\":"<<(double)count/mask.size()<<",\"coverage_fraction\":"<<(double)coverage.covered/coverage.pixels
                <<",\"source_mask_loss_fraction\":"<<(double)coverage.masked/coverage.pixels<<",\"coverage_loss_pixels\":"<<coverage.pixels-coverage.covered
                <<",\"source_mask_loss_pixels\":"<<coverage.masked<<",\"confidence_loss_pixels\":0,\"correction_scale\":"<<field.correction_scale
                <<",\"audit_count\":"<<score.count<<",\"audit_invalid\":"<<score.invalid<<",\"audit_before\":"<<score.before<<",\"audit_after\":"<<score.after
                <<",\"alignment_audit_before\":"<<alignment_score.before<<",\"alignment_audit_after\":"<<alignment_score.after
                <<",\"gradient_error_before\":"<<score.gradient_before<<",\"gradient_error_after\":"<<score.gradient_after
                <<",\"candidate_audit_after\":"<<candidate.after<<",\"candidate_audit_count\":"<<candidate.count
                <<",\"matching\":"<<matching<<",\"fallback_reason\":"<<json_quote(fallback);
            if(o.keep_alignment_data&&!o.all) {
                const auto native=dirs[k]/fs::path("original-native"+fs::path(ds.image_filenames[i]).extension().string());
                if(!fs::exists(dest/native))fs::copy_file(ds.image_filenames[i],dest/native);
                record<<",\"original_native\":"<<json_quote(native.generic_string());
            }
            record<<",\"status\":"<<json_quote(fallback.empty()?"ok":"fallback")<<",\"image\":"<<json_quote(image.generic_string())<<",\"mask\":"<<json_quote(mask_name.generic_string())
                ;
            ++succeeded;
        }catch(const std::exception& e){
            std::error_code ec;
            fs::remove(dest/"images"/relative[k],ec);
            fs::remove(dest/"masks"/fs::path(relative[k].string()+".png"),ec);
            record<<",\"status\":\"failed\",\"error\":"<<json_quote(e.what());
            std::fprintf(stderr,"%s\n",format(M::error,{e.what()}).c_str());
        }
        record<<'}';if(o.keep_alignment_data)records.push_back(record.str());
        std::printf("%s\n",format(M::progress,{(long long)(k+1),(long long)selected.size(),ds.image_filenames[i]}).c_str());
        if(!o.keep_alignment_data)return;
        std::ostringstream manifest;manifest<<"{\"version\":2,\"dataset\":"<<json_quote(o.data.string())<<",\"reference\":"<<json_quote(o.reference.string())
            <<",\"population\":"<<ds.num_cameras<<",\"seed\":"<<o.seed<<",\"selected_count\":"<<selected.size()
            <<",\"max_resolution\":"<<o.resolution<<",\"checkpoint\":"<<json_quote(checkpoint.string())
            <<",\"image_format\":"<<json_quote(o.image_format)<<",\"jpeg_quality\":"<<(o.image_format=="jpeg"?std::to_string(o.jpeg_quality):"null")
            <<",\"min_overlap\":"<<o.warp.min_overlap<<",\"max_cycle\":"<<o.warp.max_cycle
            <<",\"max_displacement\":"<<o.warp.max_displacement<<",\"min_correlation\":"<<o.warp.min_correlation
            <<",\"depth_jump\":"<<o.warp.depth_jump<<",\"spacing\":"<<o.warp.spacing
            <<",\"precision\":"<<json_quote(o.precision)<<",\"mask_mode\":\"ignore\",\"method\":\"RoMa v2; complete source mesh, robust multiscale flow, soft depth guidance; coverage and original-mask validity only\",\"views\":[";
        for(size_t j=0;j<records.size();++j){if(j)manifest<<',';manifest<<records[j];}manifest<<"]}";
        text_file(dest/"manifest.json",manifest.str());A::write_report(dest,manifest.str());
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
            if(!o.keep_alignment_data){release_scene();clamp_render(input.cache);}
            auto options=spirula::roma::MatchOptions::preset("precise");
            const int w=input.photo.width,h=input.photo.height;
            const double aspect=(double)h/w;options.low_width=800;options.low_height=std::max(16,(int)std::lround(800*aspect/16)*16);
            options.high_width=(w+3)/4*4;options.high_height=(h+3)/4*4;options.precision=precision;
            const auto match_path=o.keep_alignment_data?dest/dirs[k]:reuse_matches?o.reuse_renders/dirs[k]:fs::path{};
            matches=prediction(matcher,input.cache,input.photo,options,match_path,checkpoint,matches_cached);
        }catch(...){failure=std::current_exception();}
        if(!o.keep_alignment_data){matcher.unload();release_scene();}
        if(completed.valid())completed.get();
        completed=std::async(policy,finish,k,std::move(input),std::move(matches),matches_cached,failure);
        if(!o.pipeline)completed.get();
    }
    if(completed.valid())completed.get();
    matcher.unload();
    std::printf("%s\n",format(M::complete,{(o.keep_alignment_data?dest/"index.html":dest).string(),(long long)succeeded,(long long)(selected.size()-succeeded)}).c_str());
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
