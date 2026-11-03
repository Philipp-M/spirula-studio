#include "app/cli/AlignCompare.h"

#ifdef SS_HAVE_ROMA
#include "align/Report.h"
#include "align/Warp.h"
#include "core/LanczosResize.h"
#include "data/Json.h"
#include "data/JsonWrite.h"
#include "external/stb_image.h"
#include "nn/io/Image.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace spirula::align {
namespace {
namespace fs=std::filesystem;

const JsonValue& required(const JsonValue& v,const std::string& key) {
    const auto* found=v.find(key);if(!found)throw std::runtime_error("comparison manifest is missing "+key);return *found;
}
std::string text(const JsonValue& v,const std::string& key) {return required(v,key).as_string();}
void set_string(JsonValue& v,const std::string& key,const std::string& value) {
    JsonValue item;item.type=JsonValue::Type::String;item.str=value;
    for(auto& pair:v.obj)if(pair.first==key){pair.second=std::move(item);return;}
    v.obj.push_back({key,std::move(item)});
}
void set_value(JsonValue& v,const std::string& key,JsonValue item) {
    for(auto& pair:v.obj)if(pair.first==key){pair.second=std::move(item);return;}
    v.obj.push_back({key,std::move(item)});
}
void write_text(const fs::path& path,const std::string& value) {
    std::ofstream out(path);out<<value;if(!out)throw std::runtime_error("cannot write comparison file: "+path.string());
}
fs::path asset(const fs::path& run,const JsonValue& view,const std::string& key) {
    const auto path=fs::canonical(run/text(view,key)),rel=path.lexically_relative(run);
    if(rel.empty()||rel.is_absolute()||*rel.begin()=="..")throw std::runtime_error("comparison asset escaped its run directory");
    return path;
}
std::string link(const fs::path& file,const fs::path& output) {return file.lexically_relative(output).generic_string();}
std::string normalized_provenance(const fs::path& run) {
    std::ifstream in(run/"provenance.txt");if(!in)throw std::runtime_error("comparison requires run provenance");
    std::string out,line;size_t row=0;
    while(std::getline(in,line)) {
        if(row==3){std::istringstream values(line);std::string seed,resolution,rest;values>>seed>>resolution;std::getline(values,rest);line=seed+" RESOLUTION"+rest;}
        out+=line+'\n';++row;
    }
    if(row<7)throw std::runtime_error("incomplete comparison provenance");return out;
}
Image resize_rgb(const nn::Image& input,int w,int h) {
    std::vector<uint8_t> resized((size_t)w*h*3);resize_lanczos3<uint8_t,3>(input.data.data(),input.height,input.width,resized.data(),h,w);
    Image out;out.width=w;out.height=h;out.rgb.resize(resized.size());
    for(size_t i=0;i<resized.size();++i)out.rgb[i]=resized[i]/255.0f;return out;
}
std::vector<uint8_t> load_mask(const fs::path& path,int w,int h) {
    int width=0,height=0,channels=0;auto* pixels=stbi_load(path.string().c_str(),&width,&height,&channels,1);
    if(!pixels||width!=w||height!=h){stbi_image_free(pixels);throw std::runtime_error("comparison native mask dimensions differ");}
    std::vector<uint8_t> mask(pixels,pixels+(size_t)w*h);stbi_image_free(pixels);return mask;
}
std::vector<uint8_t> mask_coverage(const std::vector<uint8_t>& mask,int sw,int sh,int w,int h) {
    std::vector<uint8_t> out((size_t)w*h);
#pragma omp parallel for schedule(static)
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const int x0=(int)((int64_t)x*sw/w),x1=(int)(((int64_t)(x+1)*sw+w-1)/w);
        const int y0=(int)((int64_t)y*sh/h),y1=(int)(((int64_t)(y+1)*sh+h-1)/h);size_t sum=0,count=0;
        for(int yy=y0;yy<std::min(y1,sh);++yy)for(int xx=x0;xx<std::min(x1,sw);++xx){sum+=mask[(size_t)yy*sw+xx];++count;}
        out[(size_t)y*w+x]=(uint8_t)(sum/count);
    }
    return out;
}
void save_preview(const Image& image,const fs::path& path,const std::vector<uint8_t>& alpha={}) {
    nn::Image out;out.width=image.width;out.height=image.height;out.channels=alpha.empty()?3:4;out.data.resize((size_t)out.width*out.height*out.channels);
    for(size_t p=0;p<(size_t)out.width*out.height;++p) {
        for(int c=0;c<3;++c)out.data[out.channels*p+c]=(uint8_t)std::clamp(std::lround(image.rgb[3*p+c]*255),0l,255l);
        if(!alpha.empty())out.data[4*p+3]=alpha[p];
    }
    if(!nn::save_image(out,path.string(),-1))throw std::runtime_error("cannot save common comparison preview");
}

struct Run {fs::path root;JsonValue manifest;int resolution;};
}

void compare_runs(const fs::path& output,const std::vector<fs::path>& directories) {
    if(directories.size()<2)throw std::runtime_error("comparison requires at least two completed runs");
    std::vector<Run> runs;std::set<int> sizes;
    for(const auto& directory:directories) {
        const auto root=fs::canonical(directory);auto manifest=json_parse_file((root/"manifest.json").string());
        if(const auto* refined=manifest.find("refine_cameras");refined&&refined->as_bool())
            throw std::runtime_error("resolution comparison requires fixed camera poses; generate its runs with --warp-only");
        const int resolution=(int)required(manifest,"max_resolution").as_int();
        if(!sizes.insert(resolution).second)throw std::runtime_error("comparison resolutions must be distinct");
        runs.push_back({root,std::move(manifest),resolution});
    }
    std::sort(runs.begin(),runs.end(),[](const Run& a,const Run& b){return b.resolution==0?a.resolution!=0:a.resolution!=0&&a.resolution<b.resolution;});
    const auto provenance=normalized_provenance(runs.front().root);
    const auto& base=runs.front().manifest;const auto& base_views=required(base,"views").arr;
    if(base_views.empty())throw std::runtime_error("comparison has no views");
    std::set<std::string> sources;
    for(const auto& view:base_views)if(!sources.insert(text(view,"source")).second)throw std::runtime_error("comparison contains duplicate source views");
    for(const auto& run:runs) {
        if(normalized_provenance(run.root)!=provenance)throw std::runtime_error("comparison runs differ beyond maximum resolution");
        const auto& views=required(run.manifest,"views").arr;
        if(views.size()!=base_views.size())throw std::runtime_error("comparison selections differ");
        for(size_t i=0;i<views.size();++i) {
            if(text(views[i],"source")!=text(base_views[i],"source"))throw std::runtime_error("comparison source order differs");
            const auto status=text(views[i],"status");
            if(status!="ok"&&status!="fallback")throw std::runtime_error("comparison requires completed native exports for every selected view");
        }
    }
    const auto dest=checked_output(fs::canonical(text(base,"dataset")),output);fs::create_directories(dest);
    JsonWriter manifest;manifest.object().field("version",3).field("comparison",true).field("dataset",text(base,"dataset"))
        .field("reference",text(base,"reference")).field("population",(long long)required(base,"population").as_int())
        .field("seed",(long long)required(base,"seed").as_int()).field("selected_count",(long long)base_views.size())
        .field("precision",text(base,"precision")).field("evaluation_note","Identical patches and shared valid support, evaluated against one common highest-resolution render.");
    manifest.key("variants").array();
    for(const auto& run:runs) {
        manifest.object().field("resolution",run.resolution).field("run",link(run.root,dest));
        if(fs::exists(run.root/"timing.json")){manifest.key("timing");json_write(manifest,json_parse_file((run.root/"timing.json").string()));}
        manifest.end();
    }
    manifest.end().key("views").array();
    const int preview_size=runs.back().resolution?std::min(2560,runs.back().resolution):2560;
    for(size_t i=0;i<base_views.size();++i) {
        const auto& highest_view=required(runs.back().manifest,"views").arr[i];
        const auto reference=nn::load_image((asset(runs.back().root,highest_view,"preview")/"render.png").string());
        const auto original_file=base_views[i].find("original_native")?asset(runs.front().root,base_views[i],"original_native"):fs::canonical(text(base_views[i],"source"));
        const auto original=nn::load_image(original_file.string());
        if(reference.empty()||original.empty())throw std::runtime_error("comparison cannot decode original or reference");
        const int nw=original.width,nh=original.height;
        const double scale=std::min(1.0,(double)preview_size/std::max(nw,nh));
        const int pw=std::max(16,(int)std::lround(nw*scale)),ph=std::max(16,(int)std::lround(nh*scale));
        const auto preview=fs::path("review")/fs::path(text(base_views[i],"preview")).filename();fs::create_directories(dest/preview);
        save_preview(resize_rgb(original,pw,ph),dest/preview/"original.png");
        save_preview(resize_rgb(reference,pw,ph),dest/preview/"render.png");
        const int coarse=std::min(1280,preview_size);std::vector<int> levels{coarse};if(preview_size!=coarse)levels.push_back(preview_size);
        std::vector<std::vector<Image>> corrected(levels.size());std::vector<std::vector<std::vector<uint8_t>>> masks(levels.size());
        std::vector<JsonValue> versions;
        for(const auto& run:runs) {
            auto version=required(run.manifest,"views").arr[i];const auto image_path=asset(run.root,version,"image"),mask_path=asset(run.root,version,"mask");
            const auto image=nn::load_image(image_path.string());
            if(image.empty()||image.width!=nw||image.height!=nh||required(version,"width").as_int()!=nw||required(version,"height").as_int()!=nh)
                throw std::runtime_error("comparison native photograph dimensions differ");
            const auto mask=load_mask(mask_path,nw,nh);const auto vpreview=preview/std::to_string(run.resolution);fs::create_directories(dest/vpreview);
            for(size_t level=0;level<levels.size();++level) {
                const double factor=std::min(1.0,(double)levels[level]/std::max(nw,nh));
                const int w=std::max(16,(int)std::lround(nw*factor)),h=std::max(16,(int)std::lround(nh*factor));
                corrected[level].push_back(resize_rgb(image,w,h));masks[level].push_back(mask_coverage(mask,nw,nh,w,h));
            }
            save_preview(corrected.back().back(),dest/vpreview/"corrected.png",masks.back().back());
            set_string(version,"diagnostic_preview",link(asset(run.root,version,"preview"),dest));
            set_string(version,"mask_preview",link(asset(run.root,version,"preview")/"ignored.png",dest));
            set_string(version,"preview",vpreview.generic_string());set_string(version,"image",link(image_path,dest));set_string(version,"mask",link(mask_path,dest));
            JsonValue resolution;resolution.type=JsonValue::Type::Number;resolution.num=run.resolution;set_value(version,"resolution",resolution);versions.push_back(std::move(version));
        }
        std::vector<JsonWriter> scores(runs.size());for(auto& row:scores)row.array();
        for(size_t level=0;level<levels.size();++level) {
            const int w=corrected[level].front().width,h=corrected[level].front().height;
            const auto audit=audit_common(resize_rgb(original,w,h),corrected[level],resize_rgb(reference,w,h),masks[level]);
            for(size_t v=0;v<runs.size();++v)scores[v].object().field("resolution",levels[level])
                .field("count",(long long)audit[v].count).field("invalid",(long long)audit[v].invalid)
                .field("before",audit[v].before).field("after",audit[v].after)
                .field("gradient_before",audit[v].gradient_before).field("gradient_after",audit[v].gradient_after).end();
        }
        JsonValue view=base_views[i];set_string(view,"image",link(asset(runs.front().root,base_views[i],"image"),dest));set_string(view,"mask",link(asset(runs.front().root,base_views[i],"mask"),dest));set_string(view,"preview",preview.generic_string());set_string(view,"original_native",link(original_file,dest));
        JsonValue choices;choices.type=JsonValue::Type::Array;
        for(size_t v=0;v<versions.size();++v){scores[v].end();set_value(versions[v],"common_audits",json_parse(scores[v].str()));choices.arr.push_back(std::move(versions[v]));}
        set_value(view,"versions",std::move(choices));json_write(manifest,view);
    }
    manifest.end().end();write_text(dest/"manifest.json",manifest.str());write_report(dest,manifest.str());
}
}  // namespace spirula::align
#endif
