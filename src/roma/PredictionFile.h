#pragma once

#include "roma/Roma.h"

#include <istream>
#include <ostream>

namespace spirula::roma {
inline void write_prediction(std::ostream& out,const Prediction& p) {
    const int32_t dims[2]={p.width,p.height};out.write(reinterpret_cast<const char*>(dims),sizeof dims);
    for(const auto* data:{&p.warp,&p.overlap,&p.precision})
        out.write(reinterpret_cast<const char*>(data->data()),(std::streamsize)(data->size()*sizeof(float)));
}
inline bool read_prediction(std::istream& in,Prediction& p,int w,int h,bool present) {
    int32_t dims[2];in.read(reinterpret_cast<char*>(dims),sizeof dims);
    if(!in||dims[0]!=(present?w:0)||dims[1]!=(present?h:0)||w<1||h<1||(int64_t)w*h>67108864)return false;
    p.width=dims[0];p.height=dims[1];const size_t n=present?(size_t)w*h:0;
    p.warp.resize(n*2);p.overlap.resize(n);p.precision.resize(n*3);
    for(auto* data:{&p.warp,&p.overlap,&p.precision}) {
        in.read(reinterpret_cast<char*>(data->data()),(std::streamsize)(data->size()*sizeof(float)));
        if(!in)return false;for(float value:*data)if(!std::isfinite(value))return false;
    }
    return true;
}
}  // namespace spirula::roma
