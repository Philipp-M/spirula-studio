#include "align/Warp.h"
#include "align/WarpMath.h"
#include "roma/PredictionFile.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace A=spirula::align;
namespace fs=std::filesystem;

void check(bool ok,const char* what) {if(!ok)throw std::runtime_error(what);}

int main() {
    try {
        const int w=256,h=160;
        A::Image render,photo;render.width=photo.width=w;render.height=photo.height=h;
        render.rgb.resize((size_t)w*h*3);photo.rgb.resize(render.rgb.size());
        auto signal=[](int x,int y){return .5f+.16f*std::sin(.67f*x+.13f*y)+.13f*std::cos(.21f*x-.48f*y)+.08f*std::sin(.91f*y+.31f*x);};
        std::vector<float> depth((size_t)w*h),opacity(depth.size(),1);
        spirula::roma::PairPrediction p;
        for(auto* v:{&p.forward,&p.backward}) {v->width=w;v->height=h;v->warp.resize((size_t)w*h*2);v->overlap.assign((size_t)w*h,1);}
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){
            const size_t i=(size_t)y*w+x;
            for(int c=0;c<3;++c){render.rgb[3*i+c]=signal(x,y);photo.rgb[3*i+c]=signal(x<132?x-4:x,y);}
            depth[i]=x<128?1:10;
            p.forward.warp[2*i]=2*(x+(x<128?4:0)+.5f)/w-1;
            p.backward.warp[2*i]=2*(x-(x<132?4:0)+.5f)/w-1;
            p.forward.warp[2*i+1]=p.backward.warp[2*i+1]=2*(y+.5f)/h-1;
            if(x>=128&&x<132)p.forward.overlap[i]=0;
        }
        auto f=A::fit(photo,render,depth,opacity,p,{});
        float sx,sy;
        check(f.source(100,80,sx,sy)&&std::abs(sx-104)<.15f,"foreground alignment lost");
        check(f.source(160,80,sx,sy)&&std::abs(sx-160)<.15f,"foreground displacement leaked into background");
        check(!f.source(130,80,sx,sy),"disoccluded background was marked usable");
        std::vector<uint8_t> original(photo.rgb.size()),mask(depth.size(),255),corrected,valid;
        for(size_t i=0;i<original.size();++i)original[i]=(uint8_t)std::lround(photo.rgb[i]*255);
        for(int y=0;y<h;++y)for(int x=180;x<184;++x)mask[(size_t)y*w+x]=0;
        A::resample(original,w,h,mask,f,corrected,valid);
        check(valid[(size_t)80*w+100]==255,"supported foreground lost native coverage");
        check(valid[(size_t)80*w+130]==0,"native mask filled a visibility hole");
        check(valid[(size_t)80*w+181]==0,"original source mask was ignored");
        check(std::abs(corrected[((size_t)80*w+100)*3]-(int)std::lround(render.rgb[((size_t)80*w+100)*3]*255))<3,"native pull-map sign or pixel center is wrong");
        A::Image result;result.width=w;result.height=h;result.rgb.resize(corrected.size());
        for(size_t i=0;i<corrected.size();++i)result.rgb[i]=corrected[i]/255.0f;
        const auto score=A::audit(photo,result,render,f);
        check(score.count>10&&score.after>score.before,"reserved actual-output audit did not detect improvement");
        const auto camera_improvement=A::audit_camera(render,photo,render);
        check(camera_improvement.accepted&&camera_improvement.gain>0,"camera image validation missed improved render agreement");
        check(!A::audit_camera(render,render,photo).accepted,"camera image validation accepted a worse actual render");
        check(!A::audit_camera(render,photo,render,std::vector<uint8_t>((size_t)w*h,0)).accepted,
              "masked image evidence validated a camera change");
        auto audit_mask=valid;
        for(int y=30;y<100;++y)for(int x=40;x<100;++x)audit_mask[(size_t)y*w+x]=0;
        const auto common=A::audit_common(photo,{result,photo},render,{valid,audit_mask});
        check(common.size()==2&&common[0].count==common[1].count&&common[0].invalid==common[1].invalid,
              "comparison evaluated different pixels for different resolutions");
        check(common[0].count>0&&std::abs(common[0].before-common[1].before)<1e-10,
              "comparison baseline changed between versions");
        const auto unavailable=A::audit_common(photo,{result,photo},render,{valid,std::vector<uint8_t>((size_t)w*h,0)});
        check(!unavailable[0].count&&!unavailable[1].count,"comparison ignored shared source coverage");
        double source_area=0;
        for(const auto& t:f.triangles) {
            source_area+=A::detail::cross(t.source[0],t.source[1],t.source[2])*.5;
            check(A::detail::cross(t.target[0],t.target[1],t.target[2])>0,"fitted mesh folded");
        }
        check(std::abs(source_area-w*h)<.1,"surface separation discarded source area");
        auto identity=p;
        for(auto* direction:{&identity.forward,&identity.backward})for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;
            direction->warp[2*k]=2*(x+.5f)/w-1;direction->warp[2*k+1]=2*(y+.5f)/h-1;
            direction->overlap[k]=0;
        }
        auto noisy_depth=depth;std::fill(opacity.begin(),opacity.end(),0);
        for(size_t k=0;k<noisy_depth.size();++k)noisy_depth[k]=k%3?std::sin((float)k)*100:std::numeric_limits<float>::quiet_NaN();
        const auto unchanged=A::fit(photo,render,noisy_depth,opacity,identity,{});
        check(unchanged.fallback,"no-evidence correction was not labelled as a fallback");
        A::Coverage coverage;
        A::resample(original,w,h,{},unchanged,corrected,valid,&coverage);
        check(coverage.covered==coverage.pixels&&coverage.masked==0,"low confidence, absent opacity or noisy depth created holes");
        check(corrected==original,"identity fallback changed photographic details or borders");
        check(std::all_of(valid.begin(),valid.end(),[](uint8_t v){return v==255;}),"identity warp lost source-border coverage");
        for(auto* direction:{&identity.forward,&identity.backward})std::fill(direction->overlap.begin(),direction->overlap.end(),1);
        A::Image blank=photo;std::fill(blank.rgb.begin(),blank.rgb.end(),.5f);
        const auto textureless=A::fit(blank,blank,noisy_depth,opacity,identity,{});
        A::resample(original,w,h,mask,textureless,corrected,valid,&coverage);
        check(coverage.covered==coverage.pixels,"textureless regions or depth noise created a geometric gap");
        check(valid==mask,"mask transport eroded an unchanged source mask");
        check(coverage.masked==(size_t)h*4,"mask exclusions do not have exact pixel accounting");
        const int nw=w*2+17,nh=h*2+13;
        std::vector<uint8_t> native((size_t)nw*nh*3);
        for(size_t k=0;k<native.size();++k)native[k]=(uint8_t)(k*37);
        A::resample(native,nw,nh,{},unchanged,corrected,valid,&coverage);
        check(corrected==native&&coverage.covered==coverage.pixels,"native-resolution identity warp blurred detail or introduced mesh cracks");
        A::Settings no_refinement;no_refinement.refine_radius=0;
        auto translation=identity;
        for(auto* direction:{&translation.forward,&translation.backward})for(size_t k=0;k<depth.size();++k)
            direction->warp[2*k]+=(direction==&translation.backward?2.0f:-2.0f)*3/w;
        std::fill(noisy_depth.begin(),noisy_depth.end(),0);
        const auto match_improvement=A::audit_camera_matches(translation,identity,w,h,{});
        check(match_improvement.accepted&&match_improvement.before>2.99&&match_improvement.after<.001,
              "fresh matching audit missed a corrected camera displacement");
        check(!A::audit_camera_matches(identity,translation,w,h,{}).accepted,
              "fresh matching audit accepted a larger displacement");
        auto lost=identity;
        for(int y=0;y<h;++y)for(int x=w/2;x<w;++x)lost.backward.overlap[(size_t)y*w+x]=0;
        check(!A::audit_camera_matches(translation,lost,w,h,{}).accepted,
              "losing difficult correspondences passed camera validation");
        auto difficult=identity,partial=identity;
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;
            if(x>w*.9) {
                difficult.forward.warp[2*k]+=8.0f/w;difficult.backward.warp[2*k]-=8.0f/w;
                partial.backward.overlap[k]=0;
            }
        }
        check(!A::audit_camera_matches(difficult,partial,w,h,{}).accepted,
              "discarding displaced matches masqueraded as camera improvement");
        check(!A::audit_camera_matches(translation,identity,w,h,{},std::vector<uint8_t>((size_t)w*h,0)).accepted,
              "masked matches validated a camera correction");
        auto inconsistent=identity;inconsistent.forward=translation.forward;
        check(!A::audit_camera_matches(translation,inconsistent,w,h,{}).accepted,
              "one-way identity matches passed bidirectional camera validation");
        const auto shifted=A::fit(blank,blank,noisy_depth,opacity,translation,no_refinement);
        A::resample(original,w,h,{},shifted,corrected,valid,&coverage);
        for(int y=0;y<h;++y)for(int x=5;x<w-5;++x)check(valid[(size_t)y*w+x],"smooth translation created an interior hole");
        check(coverage.pixels-coverage.covered==(size_t)h*3,"translation coverage differs from its warped footprint");
        auto inverse=shifted;
        for(auto& t:inverse.triangles)std::swap(t.source,t.target);
        A::index_field(inverse);
        check(inverse.source(50,80,sx,sy,true)&&std::abs(sx-53)<.01f&&std::abs(sy-80)<.01f,
              "observation transport inverted the image correction");
        check(!inverse.source(w-1,80,sx,sy)&&inverse.source(w-1,80,sx,sy,true)&&sx>w-1,
              "observation transport lost coordinates outside the output crop");
        std::vector<std::array<double,2>> observations{{10.5,20.5},{0,0},{(double)nw,(double)nh},
            {-2,30},{nw+12.0,30},{30,-70},{30,nh+62.0}};
        const auto before_transport=observations;
        check(A::transport_observations(shifted,nw,nh,observations)==4,"off-image observations were not preserved and counted");
        for(size_t i=0;i<observations.size();++i) {
            const double dx=i<3?3.0*nw/w:0;
            check(std::abs(observations[i][0]-before_transport[i][0]-dx)<.001&&
                  std::abs(observations[i][1]-before_transport[i][1])<.001,
                  "observation transport dropped, clamped or changed an off-image record");
        }
        auto smooth=identity;
        auto movement=[&](float x,float y){return 2*std::sin(3.14159265f*x/(w-1))*std::sin(3.14159265f*y/(h-1));};
        for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            const size_t k=(size_t)y*w+x;float inverse=(float)x;
            for(int i=0;i<10;++i)inverse=x-movement(inverse,(float)y);
            smooth.backward.warp[2*k]=2*(x+movement((float)x,(float)y)+.5f)/w-1;
            smooth.forward.warp[2*k]=2*(inverse+.5f)/w-1;
            if(std::hypot(x-w*.5f,y-h*.5f)<30)smooth.forward.overlap[k]=smooth.backward.overlap[k]=0;
        }
        const auto interpolated=A::fit(blank,blank,noisy_depth,opacity,smooth,no_refinement);
        A::resample(original,w,h,{},interpolated,corrected,valid,&coverage);
        for(int y=4;y<h-4;++y)for(int x=4;x<w-4;++x)
            check(valid[(size_t)y*w+x],"low-confidence patch perforated a nonuniform smooth warp");
        check(interpolated.source(w*.5f,h*.5f,sx,sy)&&std::abs(sx-w*.5f)>1,"weakly matched area did not inherit surrounding movement");
        auto rejected=A::fit(blank,blank,noisy_depth,opacity,smooth,no_refinement,std::vector<uint8_t>((size_t)w*h,0));
        check(rejected.fallback&&rejected.anchors==0,"masked source observations anchored the deformation");
        auto folded=unchanged;
        for(auto& t:folded.triangles)for(auto& v:t.target)v[0]=w-1-v[0];
        A::use_identity(folded);
        check(folded.source(0,0,sx,sy)&&std::abs(sx)<1e-5&&std::abs(sy)<1e-5,"identity fallback retained an invalid deformation");
        std::stringstream prediction(std::ios::in|std::ios::out|std::ios::binary);
        identity.forward.precision.assign((size_t)w*h*3,1);
        spirula::roma::write_prediction(prediction,identity.forward);spirula::roma::Prediction restored;
        check(spirula::roma::read_prediction(prediction,restored,w,h,true)&&restored.warp==identity.forward.warp,"shared prediction cache changed matching coordinates");
        const auto temp=fs::temp_directory_path()/"spirula-image-alignment-test";
        fs::create_directories(temp/"source");fs::create_directories(temp/"other");
        A::save_field(f,temp/"warp.bin");const auto loaded=A::load_field(temp/"warp.bin");
        check(loaded.source(100,80,sx,sy)&&std::abs(sx-104)<.15f,"saved field does not reproduce the warp");
        auto refused=[&](auto action){try{action();return false;}catch(const std::exception&){return true;}};
        check(refused([&]{A::checked_output(temp/"source",temp/"source"/"new");}),"nested source output accepted");
        const auto alias=temp/"alias";if(!fs::exists(alias))fs::create_directory_symlink(temp/"source",alias);
        check(refused([&]{A::checked_output(temp/"source",alias/"new");}),"symlink alias bypassed source protection");
        check(refused([&]{A::checked_output(temp/"source",temp/"other");}),"existing output accepted");
        check(A::checked_output(temp/"source",temp/"fresh")==temp/"fresh","fresh external output refused");
        std::ofstream(temp/"bad.bin",std::ios::binary)<<"SSALIGN1";
        check(refused([&]{A::load_field(temp/"bad.bin");}),"truncated field accepted");
        std::cout<<"PASS: source-area conservation, physical occlusion gaps, confidence-independent coverage, depth noise, textureless areas, exact mask transport, sharp native borders, crack-free mesh sampling, translation footprint, audit, cache and field round trips, path protection\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
