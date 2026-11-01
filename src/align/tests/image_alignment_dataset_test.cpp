#include "data/ImageProbe.h"
#include "data/Json.h"
#include "external/stb_image.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("usage: image_alignment_dataset_test OUTPUT EXPECTED_COUNT");
        namespace fs=std::filesystem;
        const auto root=fs::canonical(argv[1]);const size_t expected=std::stoull(argv[2]);
        const auto manifest=json_parse_file((root/"manifest.json").string());
        const bool jpeg=manifest.find("image_format")&&manifest.find("image_format")->as_string()=="jpeg";
        const auto* views=manifest.find("views");
        if(!views||views->arr.size()!=expected)throw std::runtime_error("selected report count differs");
        for(const char* name:{"transforms.json","training-options.json","seed.ply"})
            if(fs::exists(root/name))throw std::runtime_error("export copied camera metadata or a point cloud");
        std::set<fs::path> images;
        for(const auto& v:views->arr) {
            const auto status=v.find("status")?v.find("status")->as_string():"";
            if(status!="ok"&&status!="fallback")throw std::runtime_error("sample includes a failed export");
            const auto image=fs::canonical(root/v.find("image")->as_string());
            const auto mask=fs::canonical(root/v.find("mask")->as_string());
            unsigned char image_magic[2]{},mask_magic[2]{};
            std::ifstream(image,std::ios::binary).read((char*)image_magic,2);
            std::ifstream(mask,std::ios::binary).read((char*)mask_magic,2);
            if(image.extension()!=(jpeg?".jpg":".png")||image_magic[0]!=(jpeg?0xff:0x89)||image_magic[1]!=(jpeg?0xd8:0x50)||
               mask.extension()!=".png"||mask_magic[0]!=0x89||mask_magic[1]!=0x50)
                throw std::runtime_error("export encoding differs from selected image format or lossless mask policy");
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
        std::cout<<"PASS: complete sample, native image/mask dimensions, binary coverage, safe output paths, no copied cameras or point cloud\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
