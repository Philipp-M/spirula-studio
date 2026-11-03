#include "data/ImageProbe.h"
#include "data/Json.h"
#include "external/stb_image.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if(argc!=3&&argc!=4)throw std::runtime_error("usage: image_alignment_dataset_test OUTPUT EXPECTED_COUNT [RETAINED_REFERENCE]");
        namespace fs=std::filesystem;
        const auto root=fs::canonical(argv[1]);const size_t expected=std::stoull(argv[2]);
        const auto reference=argc==4?fs::canonical(argv[3]):root;
        const auto manifest=json_parse_file((reference/"manifest.json").string());
        auto same_file=[](const fs::path& a,const fs::path& b) {
            if(fs::file_size(a)!=fs::file_size(b))return false;
            std::ifstream left(a,std::ios::binary),right(b,std::ios::binary);
            return std::equal(std::istreambuf_iterator<char>(left),std::istreambuf_iterator<char>(),std::istreambuf_iterator<char>(right),std::istreambuf_iterator<char>());
        };
        if(argc==4) {
            for(const auto& entry:fs::directory_iterator(root)) {
                const auto name=entry.path().filename().string();
                if(!entry.is_directory()||(name!="images"&&name!="masks"&&name!="sparse"))
                    throw std::runtime_error("compact export retained intermediate files");
            }
            if(fs::exists(reference/"sparse"))for(const auto& entry:fs::recursive_directory_iterator(reference/"sparse"))
                if(entry.is_regular_file()&&!same_file(entry.path(),root/entry.path().lexically_relative(reference)))
                    throw std::runtime_error("compact export changed the camera model or transported observations");
        }
        const bool cameras=manifest.find("refine_cameras")&&manifest.find("refine_cameras")->as_bool();
        const bool residual=manifest.find("residual_warp")&&manifest.find("residual_warp")->as_bool();
        const bool source_format=manifest.find("image_format")&&manifest.find("image_format")->as_string()=="source";
        const bool jpeg_format=manifest.find("image_format")&&manifest.find("image_format")->as_string()=="jpeg";
        const auto* views=manifest.find("views");
        if(!views||views->arr.size()!=expected)throw std::runtime_error("selected report count differs");
        if(cameras&&!fs::is_directory(root/"sparse/0"))throw std::runtime_error("camera-refinement output is missing its COLMAP model");
        if(!cameras&&fs::exists(root/"sparse"))throw std::runtime_error("image-only correction copied a sparse model");
        if(residual&&source_format)throw std::runtime_error("residual correction was reported as unchanged source encoding");
        for(const char* name:{"transforms.json","training-options.json","seed.ply"})
            if(fs::exists(root/name))throw std::runtime_error("export copied camera metadata or a point cloud");
        std::set<fs::path> images;
        for(const auto& v:views->arr) {
            const auto status=v.find("status")?v.find("status")->as_string():"";
            if(status!="ok"&&status!="fallback")throw std::runtime_error("sample includes a failed export");
            const auto image=fs::canonical(root/v.find("image")->as_string());
            const auto mask=fs::canonical(root/v.find("mask")->as_string());
            if(argc==4&&(!same_file(image,reference/v.find("image")->as_string())||!same_file(mask,reference/v.find("mask")->as_string())))
                throw std::runtime_error("compact export differs from retained-data images or masks");
            unsigned char image_magic[2]{},mask_magic[2]{};
            std::ifstream(image,std::ios::binary).read((char*)image_magic,2);
            std::ifstream(mask,std::ios::binary).read((char*)mask_magic,2);
            const bool jpeg=source_format?image_magic[0]==0xff:jpeg_format;
            if((!source_format&&image.extension()!=(jpeg?".jpg":".png"))||image_magic[0]!=(jpeg?0xff:0x89)||image_magic[1]!=(jpeg?0xd8:0x50)||
               mask.extension()!=".png"||mask_magic[0]!=0x89||mask_magic[1]!=0x50)
                throw std::runtime_error("export encoding differs from selected image format or lossless mask policy");
            if(source_format) {
                std::ifstream source(v.find("source")->as_string(),std::ios::binary),copy(image,std::ios::binary);
                if(!std::equal(std::istreambuf_iterator<char>(source),std::istreambuf_iterator<char>(),std::istreambuf_iterator<char>(copy),std::istreambuf_iterator<char>()))
                    throw std::runtime_error("camera refinement changed the original photograph");
            }
            const auto ir=image.lexically_relative(root/"images"),mr=mask.lexically_relative(root/"masks");
            if(mr!=fs::path(ir.string()+".png"))throw std::runtime_error("mask filename differs from COLMAP's image-name-plus-PNG convention");
            if(ir.empty()||mr.empty()||ir.is_absolute()||mr.is_absolute()||*ir.begin()==".."||*mr.begin()==".."||!images.insert(image).second)
                throw std::runtime_error("output path escaped or duplicated");
            int iw=0,ih=0,ic=0,w=0,h=0,c=0;
            if(!stbi_info(image.string().c_str(),&iw,&ih,&ic))throw std::runtime_error("corrected image header cannot be decoded");
            int sw=0,sh=0;
            if(!probe_image_size(v.find("source")->as_string().c_str(),&sw,&sh)||iw!=sw||ih!=sh)
                throw std::runtime_error("export dimensions differ from original photograph");
            auto* pixels=stbi_load(mask.string().c_str(),&w,&h,&c,1);
            if(!pixels||w!=iw||h!=ih||c!=1||ic!=3||w!=v.find("width")->as_int()||h!=v.find("height")->as_int()) {
                stbi_image_free(pixels);throw std::runtime_error("native image and mask dimensions differ");
            }
            size_t kept=0;bool binary=true;
            for(size_t i=0;i<(size_t)w*h;++i){binary&=pixels[i]==0||pixels[i]==255;kept+=pixels[i]==255;}
            stbi_image_free(pixels);
            if(!binary||!kept||std::abs((double)kept/((size_t)w*h)-v.find("valid_fraction")->as_double())>2e-6)
                throw std::runtime_error("mask values or reported coverage differ");
            const size_t absent=(size_t)v.find("coverage_loss_pixels")->as_double(),masked=(size_t)v.find("source_mask_loss_pixels")->as_double();
            if(kept+absent+masked!=(size_t)w*h||v.find("confidence_loss_pixels")->as_int()!=0)
                throw std::runtime_error("excluded pixels are not explained by warp coverage and source masks");
            if(status=="fallback"&&v.find("fallback_reason")->as_string().empty())throw std::runtime_error("unchanged fallback was not explained");
        }
        std::cout<<"PASS: complete sample, native image/mask dimensions, binary coverage and safe output paths";
        std::cout<<(source_format?", byte-identical source photographs\n":cameras?", camera model exported\n":", no copied cameras or point cloud\n");
        if(argc==4)std::cout<<"PASS: only final dataset files retained; byte-identical images, masks and sparse model\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
