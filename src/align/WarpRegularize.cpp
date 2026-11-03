#include "align/WarpMath.h"

#include <cstddef>

namespace spirula::align::detail {
namespace {
using Vec = std::array<float,2>;
constexpr int bands = 5, row_size = 1+8*bands;
using Row = std::array<float,row_size>;
constexpr std::array<std::array<int,2>,4> directions{{{1,0},{0,1},{1,1},{1,-1}}};

struct EdgeConstraints {
    std::vector<std::array<float,4>> bending, extrapolation;
};

EdgeConstraints edge_tangents(const Field& f, const std::vector<float>& photo,
                                            const std::vector<uint8_t>& mask, const std::vector<std::array<float,4>>& structure) {
    EdgeConstraints result;result.bending.resize(f.displacement.size());result.extrapolation.resize(f.displacement.size());
    std::vector<uint8_t> active(f.displacement.size());
    for (int y = 0; y < f.rows; ++y) for (int x = 0; x < f.columns; ++x) if (structure[(size_t)y*f.columns+x][3] > 0)
        for (int j = std::max(0,y-8); j <= std::min(f.rows-1,y+8); ++j)
            for (int i = std::max(0,x-8); i <= std::min(f.columns-1,x+8); ++i) active[(size_t)j*f.columns+i] = 1;
#pragma omp parallel for schedule(static)
    for (int gy = 0; gy < f.rows; ++gy) for (int gx = 0; gx < f.columns; ++gx) {
        const int x = std::min(gx*f.spacing, f.width-1), y = std::min(gy*f.spacing, f.height-1);
        float xx = 0, xy = 0, yy = 0; int count = 0;
        for (int j = std::max(1,y-f.spacing); j <= std::min(f.height-2,y+f.spacing); ++j)
            for (int i = std::max(1,x-f.spacing); i <= std::min(f.width-2,x+f.spacing); ++i) {
                const size_t k = (size_t)j*f.width+i;
                if (!mask.empty() && (!mask[k] || !mask[k-1] || !mask[k+1] || !mask[k-f.width] || !mask[k+f.width])) continue;
                const float dx = .5f*(photo[k+1]-photo[k-1]), dy = .5f*(photo[k+f.width]-photo[k-f.width]);
                xx += dx*dx; xy += dx*dy; yy += dy*dy; ++count;
            }
        const float trace = xx+yy;
        if (!count || trace < 1e-8f) continue;
        const float coherence = std::hypot(xx-yy,2*xy)/trace;
        const float confidence = std::clamp((coherence-.7f)/.3f,0.0f,1.0f)*std::clamp((trace/count-.0004f)/.002f,0.0f,1.0f);
        for (int d = 0; d < 4; ++d) {
            const auto v = directions[d];
            const float along = (v[0]*v[0]*xx+2*v[0]*v[1]*xy+v[1]*v[1]*yy)/(trace*(v[0]*v[0]+v[1]*v[1]));
            result.bending[(size_t)gy*f.columns+gx][d] = confidence*std::clamp(1-5*along,0.0f,1.0f);
        }
        if (!active[(size_t)gy*f.columns+gx]) continue;
        for (int cy = std::max(3,y-f.spacing); cy <= std::min(f.height-4,y+f.spacing); cy += 2)
            for (int cx = std::max(3,x-f.spacing); cx <= std::min(f.width-4,x+f.spacing); cx += 2) {
                float xx = 0,xy = 0,yy = 0; int count = 0;
                for (int j = cy-2; j <= cy+2; ++j) for (int i = cx-2; i <= cx+2; ++i) {
                    const size_t k = (size_t)j*f.width+i;
                    if (!mask.empty() && (!mask[k] || !mask[k-1] || !mask[k+1] || !mask[k-f.width] || !mask[k+f.width])) continue;
                    const float dx = .5f*(photo[k+1]-photo[k-1]),dy = .5f*(photo[k+f.width]-photo[k-f.width]);
                    xx += dx*dx;xy += dx*dy;yy += dy*dy;++count;
                }
                const float trace = xx+yy;
                if (count < 20 || trace < 1e-8f) continue;
                const float coherence = std::hypot(xx-yy,2*xy)/trace;
                const float confidence = std::clamp((coherence-.85f)/.15f,0.f,1.f)*std::clamp((trace/count-.001f)/.003f,0.f,1.f);
                for (int d = 0; d < 4; ++d) {
                    const auto v = directions[d];
                    const float along = (v[0]*v[0]*xx+2*v[0]*v[1]*xy+v[1]*v[1]*yy)/(trace*(v[0]*v[0]+v[1]*v[1]));
                    auto& edge = result.extrapolation[(size_t)gy*f.columns+gx][d];
                    edge = std::max(edge,confidence*std::clamp(1-5*along,0.f,1.f));
                }
            }
    }
    return result;
}

double dot(const std::vector<Vec>& a, const std::vector<Vec>& b) {
    double sum = 0;
#pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t k = 0; k < a.size(); ++k) for (int c = 0; c < 2; ++c) sum += (double)a[k][c]*b[k][c];
    return sum;
}
}

std::vector<float> depth_support(const Field& f, const std::vector<float>& weights) {
    std::vector<float> support(weights.size());
#pragma omp parallel for schedule(static)
    for (int y = 0; y < f.rows; ++y) for (int x = 0; x < f.columns; ++x) {
        const size_t k = (size_t)y*f.columns+x;
        if (!(f.depth[k] > 0)) continue;
        float sum = 0; int count = 0;
        for (int j = std::max(0,y-2); j <= std::min(f.rows-1,y+2); ++j)
            for (int i = std::max(0,x-2); i <= std::min(f.columns-1,x+2); ++i) {
                const size_t q = (size_t)j*f.columns+i;
                if (q == k || weights[q] <= .05f || !(f.depth[q] > 0) ||
                    std::abs(f.depth[k]-f.depth[q]) > .12f*std::min(f.depth[k],f.depth[q])) continue;
                sum += weights[q]; ++count;
            }
        if (count >= 3) support[k] = sum/count;
    }
    return support;
}

void regularize(Field& f, const std::vector<Vec>& data, const std::vector<float>& weights,
                const std::vector<float>& support, const std::vector<float>& photo,
                const std::vector<uint8_t>& mask, const std::vector<std::array<float,4>>& structure,bool preserve_lines) {
    const size_t n = data.size();
    const auto broad = f.displacement;
    const auto edges = edge_tangents(f, photo, mask, structure);
    const auto lines = preserve_lines?straight_lines(f,photo,mask):std::vector<LineConstraint>{};
    f.source_lines=lines.size();
    std::vector<float> line_weights(lines.size()),line_diagonal(n);
    std::vector<Row> matrix(n);
    std::array<int,row_size> offsets{};
    auto coupling = [&](size_t a, size_t b) {
        if (support[a] <= .12f || support[b] <= .12f) return 1.0f;
        const float dz = std::abs(std::log(f.depth[a]/f.depth[b]));
        return 1/(1+std::pow(dz/.08f,4.0f));
    };
    for (int d = 0; d < 4; ++d) {
        const auto v = directions[d]; const int offset = v[0]+v[1]*f.columns, slot = 1+2*bands*d;
        for (int band = 0; band < bands; ++band) {
            offsets[slot+2*band] = -(1<<band)*offset;
            offsets[slot+2*band+1] = (1<<band)*offset;
        }
        for (int y = 0; y < f.rows; ++y) for (int x = 0; x < f.columns; ++x) {
            const int bx = x+v[0], by = y+v[1];
            if (bx < 0 || bx >= f.columns || by < 0 || by >= f.rows) continue;
            const size_t k = (size_t)y*f.columns+x, b = (size_t)by*f.columns+bx;
            const float weak = std::clamp(1-std::min(weights[k],weights[b])/.1f,0.f,1.f);
            const float ridge = std::min(edges.extrapolation[k][d],edges.extrapolation[b][d]);
            const float pair = (.35f*coupling(k,b)+8*weak*ridge)/(v[0]*v[0]+v[1]*v[1]);
            matrix[k][0] += pair; matrix[b][0] += pair;
            matrix[k][slot+1] -= pair; matrix[b][slot] -= pair;
            for (int band = 0; band < bands-1; ++band) {
                const int step = 1<<band, ax = x-step*v[0], ay = y-step*v[1];
                const int cx = x+step*v[0], cy = y+step*v[1];
                if (ax < 0 || ax >= f.columns || ay < 0 || ay >= f.rows ||
                    cx < 0 || cx >= f.columns || cy < 0 || cy >= f.rows) continue;
                const size_t a = (size_t)ay*f.columns+ax, c = (size_t)cy*f.columns+cx;
                // Long stencils require a continuous photographic ridge, not just aligned endpoints.
                float tangent = edges.bending[k][d];
                for (int t = -step; t <= step; ++t) {
                    const size_t q = (size_t)((ptrdiff_t)k+t*offset);
                    tangent = std::min(tangent,edges.bending[q][d]);
                }
                const float base = band == 0 ? 4*std::min(coupling(a,k),coupling(k,c)) : 0;
                const float bend = (base+16*tangent)/(v[0]*v[0]+v[1]*v[1]);
                const size_t ids[3] = {a,k,c}; const float coefficients[3] = {1,-2,1};
                for (int i = 0; i < 3; ++i) {
                    matrix[ids[i]][0] += bend*coefficients[i]*coefficients[i];
                    for (int j = i+1; j < 3; ++j) {
                        const float value = bend*coefficients[i]*coefficients[j];
                        const int entry = slot+2*(band+(j-i == 2));
                        matrix[ids[i]][entry+1] += value;
                        matrix[ids[j]][entry] += value;
                    }
                }
            }
        }
    }
    std::vector<Vec> value(n), rhs(n), residual(n), preconditioned(n), direction(n), product(n);
    std::vector<float> diagonal(n);
    auto apply = [&](const std::vector<Vec>& input, std::vector<Vec>& output) {
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < n; ++k) {
            const auto& a = structure[k];
            const float normal = a[3]*(a[0]*input[k][0]+a[1]*input[k][1]);
            for (int c = 0; c < 2; ++c) output[k][c] = (matrix[k][0]+diagonal[k])*input[k][c]+normal*a[c];
            for (int j = 1; j < row_size; ++j) if (matrix[k][j] != 0) {
                const size_t q = (size_t)((ptrdiff_t)k+offsets[j]);
                for (int c = 0; c < 2; ++c) output[k][c] += matrix[k][j]*input[q][c];
            }
        }
        for(size_t j=0;j<lines.size();++j)if(line_weights[j]>0) {
            const auto& line=lines[j];float value=0;
            for(int i=0;i<line.count;++i)for(int c=0;c<2;++c)
                value+=line.coefficients[i]*line.normal[c]*input[line.nodes[i]][c];
            for(int i=0;i<line.count;++i)for(int c=0;c<2;++c)
                output[line.nodes[i]][c]+=line_weights[j]*line.coefficients[i]*line.normal[c]*value;
        }
    };
    for (int robust = 0; robust < 4; ++robust) {
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < n; ++k) {
            const float error = std::hypot(broad[k][0]+value[k][0]-data[k][0],broad[k][1]+value[k][1]-data[k][1]);
            const float weight = weights[k]*(robust == 0 ? 1.0f : std::min(1.0f,1.5f/std::max(.001f,error)));
            diagonal[k] = .003f+weight;
            for (int c = 0; c < 2; ++c)
                rhs[k][c] = weight*(data[k][c]-broad[k][c])+structure[k][3]*structure[k][c]*
                    (structure[k][2]-structure[k][0]*broad[k][0]-structure[k][1]*broad[k][1]);
        }
        std::fill(line_diagonal.begin(),line_diagonal.end(),0);
        for(size_t j=0;j<lines.size();++j) {
            const auto& line=lines[j];float curvature=0,base=0;
            for(int i=0;i<line.count;++i)for(int c=0;c<2;++c) {
                const float coefficient=line.coefficients[i]*line.normal[c];
                base+=coefficient*broad[line.nodes[i]][c];
                curvature+=coefficient*(broad[line.nodes[i]][c]+value[line.nodes[i]][c]);
            }
            // Subpixel curvature can be lens correction; larger local bends need photographic support.
            line_weights[j]=robust?32:0;
            const float target=std::clamp(curvature,-.25f,.25f);
            for(int i=0;i<line.count;++i) {
                const size_t k=line.nodes[i];const float coefficient=line.coefficients[i];
                line_diagonal[k]+=line_weights[j]*coefficient*coefficient;
                for(int c=0;c<2;++c)rhs[k][c]-=line_weights[j]*coefficient*line.normal[c]*(base-target);
            }
        }
        apply(value,product);
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < n; ++k) for (int c = 0; c < 2; ++c) {
            residual[k][c] = rhs[k][c]-product[k][c];
            preconditioned[k][c] = residual[k][c]/(matrix[k][0]+diagonal[k]+structure[k][3]+line_diagonal[k]);
        }
        direction = preconditioned;
        double rz = dot(residual,preconditioned);
        const double tolerance = std::max(1e-14,dot(rhs,rhs)*1e-10);
        for (int iteration = 0; iteration < 160 && dot(residual,residual) > tolerance; ++iteration) {
            apply(direction,product);
            const double denominator = dot(direction,product);
            if (!(denominator > 0) || !(rz > 0)) break;
            const float alpha = (float)(rz/denominator);
#pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n; ++k) for (int c = 0; c < 2; ++c) {
                value[k][c] += alpha*direction[k][c];
                residual[k][c] -= alpha*product[k][c];
                preconditioned[k][c] = residual[k][c]/(matrix[k][0]+diagonal[k]+structure[k][3]+line_diagonal[k]);
            }
            const double next = dot(residual,preconditioned);
            const float beta = (float)(next/rz); rz = next;
#pragma omp parallel for schedule(static)
            for (size_t k = 0; k < n; ++k) for (int c = 0; c < 2; ++c)
                direction[k][c] = preconditioned[k][c]+beta*direction[k][c];
        }
    }
#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < n; ++k) for (int c = 0; c < 2; ++c) f.displacement[k][c] = broad[k][c]+value[k][c];
}
}  // namespace spirula::align::detail
