#include "align/WarpMath.h"

namespace spirula::align::detail {
namespace {
std::vector<float> low_pass(std::vector<float> values, int w, int h) {
    std::vector<float> tmp(values.size());
    constexpr float kernel[5] = {1.f/16,4.f/16,6.f/16,4.f/16,1.f/16};
    for (int pass = 0; pass < 4; ++pass) {
#pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            float sum = 0;
            for (int j = -2; j <= 2; ++j) {
                const int xx = pass%2 ? x : std::clamp(x+j,0,w-1);
                const int yy = pass%2 ? std::clamp(y+j,0,h-1) : y;
                sum += kernel[j+2]*values[(size_t)yy*w+xx];
            }
            tmp[(size_t)y*w+x] = sum;
        }
        values.swap(tmp);
    }
    return values;
}
using Profile = std::array<float,99>;
bool profile(const std::vector<float>& image,int w,int h,float x,float y,float nx,float ny,Profile& out) {
    int k = 0; float energy = 0;
    for (int t = -8; t <= 8; t += 2) for (int n = -5; n <= 5; ++n) {
        float a,b;
        if (!sample(image,w,h,1,x+nx*(n-1)-ny*t,y+ny*(n-1)+nx*t,&a) ||
            !sample(image,w,h,1,x+nx*(n+1)-ny*t,y+ny*(n+1)+nx*t,&b)) return false;
        out[k++] = b-a; energy += (b-a)*(b-a);
    }
    if (energy < .005f) return false;
    const float scale = 1/std::sqrt(energy);
    for (auto& v : out) v *= scale;
    return true;
}
float score(const Profile& a,const Profile& b) {
    float sum = 0; for (size_t i = 0; i < a.size(); ++i) sum += a[i]*b[i]; return sum;
}
bool search(const Profile& source,const std::vector<float>& target,int w,int h,
            float x,float y,float nx,float ny,float& offset,float& quality) {
    std::array<float,33> scores; scores.fill(-2);
    int best = 16;
    for (int j = 0; j < 33; ++j) {
        Profile p;
        if (profile(target,w,h,x+(j-16)*nx,y+(j-16)*ny,nx,ny,p)) scores[j] = score(source,p);
        if (scores[j] > scores[best]) best = j;
    }
    quality = scores[best];
    if (best == 0 || best == 32 || quality < .65f) return false;
    for (int j = 0; j < 33; ++j) if (std::abs(j-best) > 3 && scores[j] > quality-.15f) return false;
    const float curvature = scores[best-1]-2*quality+scores[best+1];
    offset = best-16+(curvature < -1e-5f ? std::clamp(.5f*(scores[best-1]-scores[best+1])/curvature,-.5f,.5f) : 0);
    return true;
}
}

std::vector<std::array<float,4>> structure_matches(const Field& f,const std::vector<float>& photo,
        const std::vector<float>& render,const std::vector<float>& depth,const std::vector<float>& opacity,
        const roma::PairPrediction& matches,const Settings& settings,const std::vector<uint8_t>& mask) {
    const int w = f.width,h = f.height;
    const auto a = low_pass(photo,w,h), b = low_pass(render,w,h);
    std::vector<std::array<float,4>> anchors(f.displacement.size());
#pragma omp parallel for schedule(dynamic)
    for (int gy = 0; gy < f.rows; ++gy) for (int gx = 0; gx < f.columns; ++gx) {
        if (gx%5 == 2 && gy%5 == 2) continue;
        float x = std::min(gx*f.spacing,w)-.5f,y = std::min(gy*f.spacing,h)-.5f;
        if (x < 16 || y < 16 || x > w-17 || y > h-17) continue;
        int ix = (int)x,iy = (int)y;
        float xx = 0,xy = 0,yy = 0; bool valid = true;
        for (int j = -4; j <= 4; ++j) for (int i = -4; i <= 4; ++i) {
            const size_t k = (size_t)(iy+j)*w+ix+i;
            if (!mask.empty() && !mask[k]) valid = false;
            const float dx = a[k+1]-a[k-1],dy = a[k+w]-a[k-w];
            xx += dx*dx;xy += dx*dy;yy += dy*dy;
        }
        const float trace = xx+yy;
        if (!valid) continue;
        if (trace < .015f || std::hypot(xx-yy,2*xy) < .85f*trace) {
            float best = 0,bx = x,by = y; std::array<float,3> tensor{};
            for (int cy = iy-4; cy <= iy+4; cy += 2) for (int cx = ix-4; cx <= ix+4; cx += 2) {
                float sx = 0,sxy = 0,sy = 0;
                for (int j = -2; j <= 2; ++j) for (int i = -2; i <= 2; ++i) {
                    const size_t k = (size_t)(cy+j)*w+cx+i;
                    const float dx = a[k+1]-a[k-1],dy = a[k+w]-a[k-w];
                    sx += dx*dx;sxy += dx*dy;sy += dy*dy;
                }
                const float energy = sx+sy;
                if (energy < .005f || std::hypot(sx-sy,2*sxy) < .9f*energy || energy <= best) continue;
                best = energy;bx = (float)cx;by = (float)cy;tensor = {sx,sxy,sy};
            }
            if (best == 0) continue;
            x = bx;y = by;ix = (int)x;iy = (int)y;xx = tensor[0];xy = tensor[1];yy = tensor[2];
            if (x < 16 || y < 16 || x > w-17 || y > h-17) continue;
        }
        if (!mask.empty()) {
            for (int j = -15; j <= 15 && valid; ++j) for (int i = -15; i <= 15; ++i)
                if (!mask[(size_t)(iy+j)*w+ix+i]) {valid=false;break;}
            if (!valid) continue;
        }
        const float angle = .5f*std::atan2(2*xy,xx-yy),nx = std::cos(angle),ny = std::sin(angle);
        float qx,qy,c;
        if (!match(matches.backward,x,y,w,h,qx,qy,c) || std::hypot(qx-x,qy-y) > displacement_limit(settings,w,h)) continue;
        Profile source,target;
        float offset,quality;
        if (!profile(a,w,h,x,y,nx,ny,source) || !search(source,b,w,h,qx,qy,nx,ny,offset,quality)) continue;
        qx += offset*nx;qy += offset*ny;
        if (qx < 16 || qy < 16 || qx > w-17 || qy > h-17 || std::hypot(qx-x,qy-y) > displacement_limit(settings,w,h)) continue;
        float backward,reverse;
        if (!profile(b,w,h,qx,qy,nx,ny,target) || !search(target,a,w,h,x,y,nx,ny,backward,reverse) || std::abs(backward) > 1) continue;
        const int rx = (int)std::lround(qx),ry = (int)std::lround(qy);
        float near_depth = 1e30f,far_depth = 0; int missing_depth = 0;
        for (int j = -3; j <= 3; j += 3) for (int i = -3; i <= 3; i += 3) {
            const size_t k = (size_t)(ry+j)*w+rx+i;
            if (opacity[k] >= .5f && depth[k] > 0 && std::isfinite(depth[k])) near_depth = std::min(near_depth,depth[k]);
        }
        for (auto d : {std::array<int,2>{-32,0},{32,0},{0,-32},{0,32}}) {
            const size_t k = (size_t)std::clamp(ry+d[1],0,h-1)*w+std::clamp(rx+d[0],0,w-1);
            if (depth[k] > 0 && std::isfinite(depth[k])) far_depth = std::max(far_depth,depth[k]);
            else ++missing_depth;
        }
        if (!(near_depth < 1e30f) || (!(far_depth > 3*near_depth) && missing_depth < 2)) continue;
        // Sub-grid ridge samples share the local mesh displacement.
        anchors[(size_t)gy*f.columns+gx] = {nx,ny,nx*(qx-x)+ny*(qy-y),16*std::clamp((std::min(quality,reverse)-.6f)/.3f,0.f,1.f)};
    }
    const auto candidates = anchors;
    for (int y = 0; y < f.rows; ++y) for (int x = 0; x < f.columns; ++x) {
        auto& a = anchors[(size_t)y*f.columns+x];
        if (a[3] == 0) continue;
        bool supported = false;
        for (int j = std::max(0,y-2); j <= std::min(f.rows-1,y+2); ++j)
            for (int i = std::max(0,x-2); i <= std::min(f.columns-1,x+2); ++i) {
                if (i == x && j == y) continue;
                const auto& b = candidates[(size_t)j*f.columns+i];
                const float cosine = a[0]*b[0]+a[1]*b[1];
                if (b[3] > 0 && std::abs(cosine) > .95f && std::abs(a[2]-std::copysign(1.f,cosine)*b[2]) < 3) supported = true;
            }
        if (!supported) {a[3]=0;continue;}
        std::vector<float> offsets;
        for (int j = std::max(0,y-6); j <= std::min(f.rows-1,y+6); ++j)
            for (int i = std::max(0,x-6); i <= std::min(f.columns-1,x+6); ++i) {
                const auto& b = candidates[(size_t)j*f.columns+i];
                const float cosine = a[0]*b[0]+a[1]*b[1];
                if (b[3] > 0 && std::abs(cosine) > .95f) offsets.push_back(std::copysign(1.f,cosine)*b[2]);
            }
        if (offsets.size() >= 5) {
            auto middle = offsets.begin()+offsets.size()/2;
            std::nth_element(offsets.begin(),middle,offsets.end());
            if (std::abs(a[2]-*middle) > 3) a[3] = 0;
        }
    }
    return anchors;
}
}  // namespace spirula::align::detail
