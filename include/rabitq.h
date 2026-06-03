#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
namespace diskann {

// k-bit scalar quantization (k=4 or k=6).
// File format (<prefix>_sqk.bin):
//   [N:u64][D:u64][k:u64]
//   [dim_mins: D*f32][dim_scales: D*f32]
//   [codes: N * ceil(D*k/8) bytes]

inline uint32_t sqk_bytes_per_vec(uint32_t D, uint32_t k){ return (D*k+7)/8; }

// Decode single k-bit code at dimension d.
inline uint32_t sqk_get(const uint8_t *p, uint32_t d, uint32_t k){
    if(k==4){ uint8_t b=p[d>>1]; return (d&1)?(b>>4):(b&0xF); }
    // k==6: 4 dims per 3 bytes
    uint32_t bo=d*6, bi=bo>>3, sh=bo&7;
    uint32_t w=(uint32_t)p[bi]|((uint32_t)p[bi+1]<<8);
    return (w>>sh)&0x3F;
}

// Build per-query LUT: lut[d*(1<<k)+c] = (q[d] - (min_d + c*scale_d))^2
// lut must have D*(1<<k) floats.
inline void sqk_build_lut(const float *q, const float *mins, const float *scales,
                           uint32_t D, uint32_t k, float *lut){
    uint32_t levels=1u<<k;
    for(uint32_t d=0;d<D;d++){
        float qd=q[d];
        for(uint32_t c=0;c<levels;c++){
            float diff=qd-(mins[d]+c*scales[d]); lut[d*levels+c]=diff*diff;
        }
    }
}

// Compute L2^2 distance using LUT.
inline float sqk_dist(const uint8_t *code, const float *lut, uint32_t D, uint32_t k){
    uint32_t levels=1u<<k; float dist=0.f;
    for(uint32_t d=0;d<D;d++) dist+=lut[d*levels+sqk_get(code,d,k)];
    return dist;
}

} // namespace diskann
