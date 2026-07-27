// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2026 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// ------------------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <stdint.h>

__constant__ uint8_t c_scan[64] = {
    0, 1, 8, 9, 2, 3,10,11,16,17,24,25,18,19,26,27,
    4, 5,12,20,13, 6, 7,14,21,28,29,22,15,23,30,31,
   32,33,40,48,41,34,35,42,49,56,57,50,43,36,37,44,
   51,58,59,52,45,38,39,46,53,60,61,54,47,55,62,63
};

__constant__ uint8_t c_luma_qmat[64];
__constant__ uint8_t c_chroma_qmat[64];

#define FIRST_DC_CB 0xB8
#define PACK_CB(c) ((((c) & 3) + 1) | (((c) >> 5) << 3) | ((((c) >> 2) & 7) << 6))
#define FIRST_DC_PACKED PACK_CB(FIRST_DC_CB)

__device__ __forceinline__ int dc_codebook_for(int code) {
    return code <= 0 ? 0x04 : (code <= 2 ? 0x28 : (code <= 4 ? 0x4D : 0x70));
}

__device__ __forceinline__ int run_codebook_for(unsigned run) {
    return run < 2 ? 0x06 :
           run < 4 ? 0x05 :
           run == 4 ? 0x04 :
           run < 9 ? 0x29 :
           run < 15 ? 0x28 : 0x4C;
}

__device__ __forceinline__ int level_codebook_for(unsigned level) {
    return level == 0 ? 0x04 :
           level == 1 ? 0x0A :
           level == 2 ? 0x05 :
           level == 3 ? 0x06 :
           level == 4 ? 0x04 :
           level < 9 ? 0x28 : 0x4C;
}

struct SliceInfo {
    uint32_t offset; uint32_t size;
    uint16_t mb_x; uint16_t mb_y; uint16_t mb_count;
    uint8_t qindex; uint8_t reserved;
};

// --- 64-bit cached bitstream reader ---
struct BitReader {
    const uint8_t *data;
    int size, byte_pos, cache_bits;
    uint64_t cache;
};

__device__ __forceinline__ void br_refill(BitReader *br);

__device__ __forceinline__ void br_init(BitReader *br, const uint8_t *data, int size_bytes) {
    br->data = data; br->size = size_bytes; br->byte_pos = 0;
    br->cache = 0; br->cache_bits = 0;
    br_refill(br);
    br_refill(br);
}
__device__ __forceinline__ void br_refill(BitReader *br) {
    if (br->cache_bits < 33 && br->byte_pos + 4 <= br->size) {
        const uint8_t *p = br->data + br->byte_pos;
        uint32_t word = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
        br->cache |= (uint64_t)word << (32 - br->cache_bits);
        br->cache_bits += 32;
        br->byte_pos += 4;
    }
}
__device__ __forceinline__ int br_bits_left(BitReader *br) { return br->cache_bits + (br->size - br->byte_pos) * 8; }
__device__ __forceinline__ unsigned br_read(BitReader *br, int n) {
    if (n == 0) return 0;
    br_refill(br);
    unsigned val = (unsigned)(br->cache >> (64 - n));
    br->cache <<= n; br->cache_bits -= n;
    return val;
}
__device__ __forceinline__ int br_read1(BitReader *br) {
    br_refill(br);
    int val = (int)(br->cache >> 63);
    br->cache <<= 1; br->cache_bits -= 1;
    return val;
}
__device__ __forceinline__ int br_read_unary(BitReader *br) {
    int total = 0;
    for (;;) {
        br_refill(br);
        if (br->cache_bits <= 0) return total;
        int lz = __clzll(br->cache);
        if (lz < br->cache_bits) {
            total += lz;
            br->cache <<= (lz + 1); br->cache_bits -= (lz + 1);
            return total;
        }
        total += br->cache_bits;
        br->cache = 0; br->cache_bits = 0;
    }
}

__device__ __forceinline__ int decode_codeword(BitReader *br, int codebook) {
    unsigned sb = codebook & 3, ro = codebook >> 5, eo = (codebook >> 2) & 7;
    unsigned q = br_read_unary(br);
    if (q > sb) {
        int rem = (int)eo - (int)sb + (int)q - 1; if (rem < 0) rem = 0;
        unsigned suf = (rem > 0) ? br_read(br, rem) : 0;
        return (int)((1u << rem) + suf - (1u << eo) + ((sb + 1) << ro));
    } else if (ro) { return (int)((q << ro) | br_read(br, ro)); }
    else { return (int)q; }
}

__device__ __forceinline__ int decode_codeword_fast(BitReader *br, int codebook) {
    br_refill(br);
    if (br->cache_bits <= 0) return 0;

    unsigned q = (unsigned)__clzll(br->cache);

    unsigned sb = codebook & 3;
    unsigned sbp1 = sb + 1;
    unsigned ro = codebook >> 5;
    unsigned eo = (codebook >> 2) & 7;
    unsigned bits;
    unsigned value;

    if (q < sbp1) {
        bits = q + 1 + ro;
        uint64_t tail = br->cache << (q + 1);
        unsigned suffix = ro ? (unsigned)(tail >> (64 - ro)) : 0;
        value = (q << ro) | suffix;
    } else {
        int rem_i = (int)eo - (int)sbp1 + (int)q;
        unsigned rem = rem_i > 0 ? (unsigned)rem_i : 0;
        bits = q + 1 + rem;
        uint64_t tail = br->cache << (q + 1);
        unsigned suffix = rem ? (unsigned)(tail >> (64 - rem)) : 0;
        value = (1u << rem) + suffix - (1u << eo) + (sbp1 << ro);
    }

    br->cache <<= bits;
    br->cache_bits -= (int)bits;
    return (int)value;
}

__device__ __forceinline__ int decode_codeword_fast_packed(BitReader *br, int packed) {
    br_refill(br);
    if (br->cache_bits <= 0) return 0;
    if (br->cache == 0) return 0x7fff;

    unsigned q = (unsigned)__clzll(br->cache);

    unsigned sbp1 = packed & 7;
    unsigned ro = (packed >> 3) & 7;
    unsigned eo = (packed >> 6) & 7;
    unsigned bits;
    unsigned value;

    if (q < sbp1) {
        bits = q + 1 + ro;
        uint64_t tail = br->cache << (q + 1);
        unsigned suffix = ro ? (unsigned)(tail >> (64 - ro)) : 0;
        value = (q << ro) | suffix;
    } else {
        int rem_i = (int)eo - (int)sbp1 + (int)q;
        unsigned rem = rem_i > 0 ? (unsigned)rem_i : 0;
        bits = q + 1 + rem;
        uint64_t tail = br->cache << (q + 1);
        unsigned suffix = rem ? (unsigned)(tail >> (64 - rem)) : 0;
        value = (1u << rem) + suffix - (1u << eo) + (sbp1 << ro);
    }

    br->cache <<= bits;
    br->cache_bits -= (int)bits;
    return (int)value;
}

__device__ int tosigned(int x) { return (x >> 1) ^ (-(x & 1)); }

__device__ void decode_dc_coeffs(BitReader *br, int16_t *out, int bps) {
    int code = decode_codeword(br, FIRST_DC_CB);
    int16_t prev = (int16_t)tosigned(code); out[0] = prev;
    int sign = 0; code = 5;
    for (int i = 1; i < bps; i++) {
        code = decode_codeword(br, dc_codebook_for(code));
        if (code) sign ^= -(code & 1); else sign = 0;
        prev += (int16_t)((((code+1)>>1) ^ sign) - sign);
        out[i*64] = prev;
    }
}

__device__ void decode_ac_coeffs(BitReader *br, int16_t *out, int bps) {
    int l2 = 0; { int v=bps; while(v>1){v>>=1;l2++;} }
    int maxc = 64 << l2, mask = bps - 1;
    unsigned run = 4, level = 2, pos = mask;
    for (;;) {
        int bl = br_bits_left(br);
        if (bl <= 0) break;
        if (bl < 32) {
            BitReader save = *br;
            unsigned peek = br_read(br, bl);
            if (peek == 0) break;
            *br = save;
        }
        run = decode_codeword(br, run_codebook_for(run));
        pos += run + 1;
        if (pos >= (unsigned)maxc) break;
        level = decode_codeword(br, level_codebook_for(level));
        level += 1;
        int i = pos >> l2;
        int sign = br_read1(br) ? -1 : 0;
        out[((pos & mask) << 6) + c_scan[i]] = (int16_t)((level ^ sign) - sign);
    }
}

__device__ void decode_dc_coeffs_lut(BitReader *br, int16_t *out, int bps, const int *dc_cb, int lane) {
    int code = decode_codeword_fast(br, FIRST_DC_CB);
    int16_t prev = (int16_t)tosigned(code); out[0] = prev;
    int sign = 0; code = 5;
    for (int i = 1; i < bps; i++) {
        int state = (code + (code & 1)) >> 1;
        state = state < 3 ? state : 3;
        code = decode_codeword_fast(br, dc_cb[state * 32 + lane]);
        if (code) sign ^= -(code & 1); else sign = 0;
        prev += (int16_t)((((code+1)>>1) ^ sign) - sign);
        out[i*64] = prev;
    }
}

template<int BPS>
__device__ __forceinline__ void decode_dc_coeffs_lut_fixed(BitReader *br, int16_t *out, const int *dc_cb, int lane) {
    int code = decode_codeword_fast(br, FIRST_DC_CB);
    int16_t prev = (int16_t)tosigned(code); out[0] = prev;
    int sign = 0; code = 5;
    for (int i = 1; i < BPS; i++) {
        int state = (code + (code & 1)) >> 1;
        state = state < 3 ? state : 3;
        code = decode_codeword_fast(br, dc_cb[state * 32 + lane]);
        if (code) sign ^= -(code & 1); else sign = 0;
        prev += (int16_t)((((code+1)>>1) ^ sign) - sign);
        out[i*64] = prev;
    }
}

__device__ void decode_ac_coeffs_shared(BitReader *br, int16_t *out, int bps, const int *scan) {
    int l2 = 0; { int v=bps; while(v>1){v>>=1;l2++;} }
    int maxc = 64 << l2, mask = bps - 1;
    unsigned run = 4, level = 2, pos = mask;
    for (;;) {
        int bl = br_bits_left(br);
        if (bl <= 0) break;
        if (bl < 32) {
            BitReader save = *br;
            unsigned peek = br_read(br, bl);
            if (peek == 0) break;
            *br = save;
        }
        run = decode_codeword(br, run_codebook_for(run));
        pos += run + 1;
        if (pos >= (unsigned)maxc) break;
        level = decode_codeword(br, level_codebook_for(level));
        level += 1;
        int i = pos >> l2;
        int sign = br_read1(br) ? -1 : 0;
        out[((pos & mask) << 6) + scan[i]] = (int16_t)((level ^ sign) - sign);
    }
}

__device__ void decode_ac_coeffs_lut(
    BitReader *br, int16_t *out, int bps, const int *scan,
    const int *run_cb, const int *level_cb, int lane
) {
    int l2 = 0; { int v=bps; while(v>1){v>>=1;l2++;} }
    int maxc = 64 << l2, mask = bps - 1;
    unsigned run = 4, level = 2, pos = mask;
    for (;;) {
        if (br->byte_pos >= br->size && br->cache == 0) break;
        unsigned run_state = run < 15 ? run : 15;
        run = decode_codeword_fast(br, run_cb[run_state * 32 + lane]);
        pos += run + 1;
        if (pos >= (unsigned)maxc) break;
        unsigned level_state = level < 9 ? level : 9;
        level = decode_codeword_fast(br, level_cb[level_state * 32 + lane]);
        level += 1;
        int i = pos >> l2;
        int sign = br_read1(br) ? -1 : 0;
        out[((pos & mask) << 6) + scan[i * 32 + lane]] = (int16_t)((level ^ sign) - sign);
    }
}

template<int L2, int MASK, int MAXC>
__device__ __forceinline__ void decode_ac_coeffs_lut_fixed(
    BitReader *br, int16_t *out, const int *scan,
    const int *run_cb, const int *level_cb, int lane
) {
    unsigned run = 4, level = 2, pos = MASK;
    for (;;) {
        if (br->byte_pos >= br->size && br->cache == 0) break;
        unsigned run_state = run < 15 ? run : 15;
        run = decode_codeword_fast(br, run_cb[run_state * 32 + lane]);
        pos += run + 1;
        if (pos >= (unsigned)MAXC) break;
        unsigned level_state = level < 9 ? level : 9;
        level = decode_codeword_fast(br, level_cb[level_state * 32 + lane]);
        level += 1;
        int i = pos >> L2;
        int sign = br_read1(br) ? -1 : 0;
        out[((pos & MASK) << 6) + scan[i * 32 + lane]] = (int16_t)((level ^ sign) - sign);
    }
}

__device__ void init_entropy_luts(int tid, int *dc_cb, int *run_cb, int *level_cb) {
    dc_cb[tid] = 0x04;
    dc_cb[32 + tid] = 0x28;
    dc_cb[64 + tid] = 0x4D;
    dc_cb[96 + tid] = 0x70;

    int r = tid;
    run_cb[r] = 0x06;          // state 0
    run_cb[32 + r] = 0x06;     // state 1
    run_cb[64 + r] = 0x05;     // state 2
    run_cb[96 + r] = 0x05;     // state 3
    run_cb[128 + r] = 0x04;    // state 4
    run_cb[160 + r] = 0x29;    // state 5
    run_cb[192 + r] = 0x29;
    run_cb[224 + r] = 0x29;
    run_cb[256 + r] = 0x29;    // state 8
    run_cb[288 + r] = 0x28;    // state 9
    run_cb[320 + r] = 0x28;
    run_cb[352 + r] = 0x28;
    run_cb[384 + r] = 0x28;
    run_cb[416 + r] = 0x28;
    run_cb[448 + r] = 0x28;    // state 14
    run_cb[480 + r] = 0x4C;    // state 15+

    level_cb[r] = 0x04;        // state 0
    level_cb[32 + r] = 0x0A;   // state 1
    level_cb[64 + r] = 0x05;   // state 2
    level_cb[96 + r] = 0x06;   // state 3
    level_cb[128 + r] = 0x04;  // state 4
    level_cb[160 + r] = 0x28;  // state 5
    level_cb[192 + r] = 0x28;
    level_cb[224 + r] = 0x28;
    level_cb[256 + r] = 0x28;  // state 8
    level_cb[288 + r] = 0x4C;  // state 9+
}


// --- IDCT ---
#define W1 (22725 * 4)
#define W2 (21407 * 4)
#define W3 (19265 * 4)
#define W4 (16384 * 4)
#define W5 (12873 * 4)
#define W6 ( 8867 * 4)
#define W7 ( 4520 * 4)
#define ROW_SHIFT 17
#define COL_SHIFT 20
#define ROW_ROUND (1 << (ROW_SHIFT - 1))
#define COL_ROUND 8

__device__ void idct_row(int16_t *row) {
    int x0=row[0],x1=row[1],x2=row[2],x3=row[3],x4=row[4],x5=row[5],x6=row[6],x7=row[7];
    if (!(x1|x2|x3|x4|x5|x6|x7)) { int dc=(x0+1)>>1; for(int i=0;i<8;i++) row[i]=(int16_t)dc; return; }
    int a0=W4*x0+ROW_ROUND,a1=a0,a2=a0,a3=a0;
    a0+=W2*x2;a1+=W6*x2;a2+=-W6*x2;a3+=-W2*x2;
    a0+=W4*x4;a1+=-W4*x4;a2+=-W4*x4;a3+=W4*x4;
    a0+=W6*x6;a1+=-W2*x6;a2+=W2*x6;a3+=-W6*x6;
    int b0=W1*x1,b1=W3*x1,b2=W5*x1,b3=W7*x1;
    b0+=W3*x3;b1+=-W7*x3;b2+=-W1*x3;b3+=-W5*x3;
    b0+=W5*x5;b1+=-W1*x5;b2+=W7*x5;b3+=W3*x5;
    b0+=W7*x7;b1+=-W5*x7;b2+=W3*x7;b3+=-W1*x7;
    row[0]=(int16_t)((a0+b0)>>ROW_SHIFT);row[1]=(int16_t)((a1+b1)>>ROW_SHIFT);
    row[2]=(int16_t)((a2+b2)>>ROW_SHIFT);row[3]=(int16_t)((a3+b3)>>ROW_SHIFT);
    row[4]=(int16_t)((a3-b3)>>ROW_SHIFT);row[5]=(int16_t)((a2-b2)>>ROW_SHIFT);
    row[6]=(int16_t)((a1-b1)>>ROW_SHIFT);row[7]=(int16_t)((a0-b0)>>ROW_SHIFT);
}
__device__ void idct_col(int16_t *col) {
    int x0=col[0],x1=col[8],x2=col[16],x3=col[24],x4=col[32],x5=col[40],x6=col[48],x7=col[56];
    int a0=W4*(x0+COL_ROUND),a1=a0,a2=a0,a3=a0;
    a0+=W2*x2;a1+=W6*x2;a2+=-W6*x2;a3+=-W2*x2;
    if(x4){a0+=W4*x4;a1+=-W4*x4;a2+=-W4*x4;a3+=W4*x4;}
    if(x6){a0+=W6*x6;a1+=-W2*x6;a2+=W2*x6;a3+=-W6*x6;}
    int b0=W1*x1,b1=W3*x1,b2=W5*x1,b3=W7*x1;
    b0+=W3*x3;b1+=-W7*x3;b2+=-W1*x3;b3+=-W5*x3;
    if(x5){b0+=W5*x5;b1+=-W1*x5;b2+=W7*x5;b3+=W3*x5;}
    if(x7){b0+=W7*x7;b1+=-W5*x7;b2+=W3*x7;b3+=-W1*x7;}
    col[0]=(int16_t)((a0+b0)>>COL_SHIFT);col[8]=(int16_t)((a1+b1)>>COL_SHIFT);
    col[16]=(int16_t)((a2+b2)>>COL_SHIFT);col[24]=(int16_t)((a3+b3)>>COL_SHIFT);
    col[32]=(int16_t)((a3-b3)>>COL_SHIFT);col[40]=(int16_t)((a2-b2)>>COL_SHIFT);
    col[48]=(int16_t)((a1-b1)>>COL_SHIFT);col[56]=(int16_t)((a0-b0)>>COL_SHIFT);
}

__device__ void idct_put_cq(int16_t *blk, const uint8_t *qm, int qs, int bd) {
    for (int i=0;i<64;i++) blk[i]=(int16_t)((int)blk[i]*(int)qm[i]*qs);
    for (int i=0;i<8;i++) idct_row(blk+i*8);
    for (int i=0;i<8;i++) blk[i]+=8192;
    for (int i=0;i<8;i++) idct_col(blk+i);
    int mn=4, mx=(1<<bd)-5;
    for (int i=0;i<64;i++) blk[i]=(int16_t)max(mn,min(mx,(int)blk[i]));
}

__device__ void idct_put_cq_shared(int16_t *blk, const int *qm, int qs, int bd) {
    for (int i=0;i<64;i++) blk[i]=(int16_t)((int)blk[i]*qm[i]*qs);
    for (int i=0;i<8;i++) idct_row(blk+i*8);
    for (int i=0;i<8;i++) blk[i]+=8192;
    for (int i=0;i<8;i++) idct_col(blk+i);
    int mn=4, mx=(1<<bd)-5;
    for (int i=0;i<64;i++) blk[i]=(int16_t)max(mn,min(mx,(int)blk[i]));
}

__device__ __forceinline__ void zero_blocks_i16(int16_t *blocks, int block_count) {
    for (int i = 0; i < block_count * 64; i++) {
        blocks[i] = 0;
    }
}

__device__ __forceinline__ void store_block8x8_i16(int16_t *dst, int stride, const int16_t *blk) {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            dst[r * stride + c] = blk[r * 8 + c];
        }
    }
}

__device__ __forceinline__ void prores_chroma444_block_pos(int sub, int *bx, int *by) {
    // ProRes 4:4:4 chroma blocks are coded column-major inside a 16x16 macroblock.
    *bx = (sub >> 1) * 8;
    *by = (sub & 1) * 8;
}

// --- Main kernel ---
extern "C" __global__ void prores_decode_slice(
    const uint8_t *compressed, const SliceInfo *slice_info,
    int16_t *out_y, int16_t *out_cb, int16_t *out_cr,
    int stride_y, int stride_c, int pic_width, int pic_height,
    int is_444, int bits_per_component, int num_slices
) {
    int si_idx = blockIdx.x;
    if (si_idx >= num_slices) return;
    int tid = threadIdx.x;
    extern __shared__ int16_t smem[];

    SliceInfo si = slice_info[si_idx];
    const uint8_t *sd = compressed + si.offset;
    int hs = sd[0]>>3, qs;
    { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; }
    int yds=(sd[2]<<8)|sd[3], uds=(sd[4]<<8)|sd[5], vds;
    if (hs>7) vds=(sd[6]<<8)|sd[7]; else vds=(int)si.size-yds-uds-hs;
    int mbc=si.mb_count, yb=mbc*4, cb=mbc<<(is_444?2:1);

    // Y
    if(tid==0){for(int i=0;i<yb*64;i++)smem[i]=0; BitReader br; br_init(&br,sd+hs,yds); decode_dc_coeffs(&br,smem,yb); decode_ac_coeffs(&br,smem,yb);}
    __syncthreads();
    if(tid<yb){int16_t*b=smem+tid*64; idct_put_cq(b,c_luma_qmat,qs,bits_per_component); int m=tid/4,s=tid%4,bx=(s&1)*8,by=(s>>1)*8; int16_t*d=out_y+(si.mb_y*16+by)*stride_y+(si.mb_x+m)*16+bx; for(int r=0;r<8;r++)for(int c=0;c<8;c++)d[r*stride_y+c]=b[r*8+c];}
    __syncthreads();
    // Cb
    if(tid==0){for(int i=0;i<cb*64;i++)smem[i]=0; BitReader br; br_init(&br,sd+hs+yds,uds); decode_dc_coeffs(&br,smem,cb); decode_ac_coeffs(&br,smem,cb);}
    __syncthreads();
    if(tid<cb){int16_t*b=smem+tid*64; idct_put_cq(b,c_chroma_qmat,qs,bits_per_component); int m,bx,by; if(is_444){m=tid/4;prores_chroma444_block_pos(tid%4,&bx,&by);}else{m=tid/2;int s=tid%2;bx=0;by=s*8;} int ms=is_444?16:8; int16_t*d=out_cb+(si.mb_y*16+by)*stride_c+(si.mb_x+m)*ms+bx; for(int r=0;r<8;r++)for(int c=0;c<8;c++)d[r*stride_c+c]=b[r*8+c];}
    __syncthreads();
    // Cr
    if(tid==0){for(int i=0;i<cb*64;i++)smem[i]=0; int cs=vds;if(cs<0)cs=0; BitReader br; br_init(&br,sd+hs+yds+uds,cs); decode_dc_coeffs(&br,smem,cb); decode_ac_coeffs(&br,smem,cb);}
    __syncthreads();
    if(tid<cb){int16_t*b=smem+tid*64; idct_put_cq(b,c_chroma_qmat,qs,bits_per_component); int m,bx,by; if(is_444){m=tid/4;prores_chroma444_block_pos(tid%4,&bx,&by);}else{m=tid/2;int s=tid%2;bx=0;by=s*8;} int ms=is_444?16:8; int16_t*d=out_cr+(si.mb_y*16+by)*stride_c+(si.mb_x+m)*ms+bx; for(int r=0;r<8;r++)for(int c=0;c<8;c++)d[r*stride_c+c]=b[r*8+c];}
}

// === Lane-parallel decode: each thread decodes one slice independently ===
// grid = ceil(num_slices/32), block = 32
// Each thread: entropy decode + IDCT + output write for its slice
// No __syncthreads needed — all threads are independent

extern "C" __global__ void __launch_bounds__(32, 1) pr_decode_luma(
    const uint8_t *compressed, const SliceInfo *slice_info,
    int16_t *out_y, int stride_y,
    int bits_per_component, int num_slices
) {
    __shared__ int s_qmat[64];
    __shared__ int s_scan_lane[64 * 32];
    __shared__ int s_dc_cb[128];
    __shared__ int s_run_cb[512];
    __shared__ int s_level_cb[320];
    int tid = threadIdx.x;
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i];
    s_qmat[tid] = c_luma_qmat[tid];
    s_qmat[tid + 32] = c_luma_qmat[tid + 32];
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb);
    __syncthreads();

    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= num_slices) return;

    SliceInfo si = slice_info[work_idx];
    const uint8_t *sd = compressed + si.offset;
    int hs = sd[0] >> 3;
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; }
    int yds = (sd[2]<<8)|sd[3];
    int mbc = si.mb_count;
    int yb = mbc * 4; // Y blocks per slice (max 32 for 8 MBs)

    // Local coefficient storage (in registers/local memory)
    __align__(16) int16_t blocks[32 * 64]; // max 32 blocks * 64 coeffs
    zero_blocks_i16(blocks, yb);

    BitReader br;
    br_init(&br, sd + hs, yds);
    if (mbc == 8) {
        decode_dc_coeffs_lut_fixed<32>(&br, blocks, s_dc_cb, tid);
        decode_ac_coeffs_lut_fixed<5, 31, 2048>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid);
    } else {
        decode_dc_coeffs_lut(&br, blocks, yb, s_dc_cb, tid);
        decode_ac_coeffs_lut(&br, blocks, yb, s_scan_lane, s_run_cb, s_level_cb, tid);
    }

    // IDCT + output for each block
    for (int b = 0; b < yb; b++) {
        int16_t *blk = blocks + b * 64;
        idct_put_cq_shared(blk, s_qmat, qs, bits_per_component);
        int mb = b / 4, sub = b % 4;
        int bx = (sub & 1) * 8, by = (sub >> 1) * 8;
        int16_t *dst = out_y + (si.mb_y * 16 + by) * stride_y + (si.mb_x + mb) * 16 + bx;
        store_block8x8_i16(dst, stride_y, blk);
    }
}

extern "C" __global__ void __launch_bounds__(32, 1) pr_decode_chroma422(
    const uint8_t *compressed, const SliceInfo *slice_info,
    int16_t *out_plane, int stride_c,
    int bits_per_component, int num_slices,
    int plane_offset // 0 for Cb, 1 for Cr (selects data offset)
) {
    __shared__ int s_qmat[64];
    __shared__ int s_scan_lane[64 * 32];
    __shared__ int s_dc_cb[128];
    __shared__ int s_run_cb[512];
    __shared__ int s_level_cb[320];
    int tid = threadIdx.x;
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i];
    s_qmat[tid] = c_chroma_qmat[tid];
    s_qmat[tid + 32] = c_chroma_qmat[tid + 32];
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb);
    __syncthreads();

    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= num_slices) return;

    SliceInfo si = slice_info[work_idx];
    const uint8_t *sd = compressed + si.offset;
    int hs = sd[0] >> 3;
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; }
    int yds = (sd[2]<<8)|sd[3];
    int uds = (sd[4]<<8)|sd[5];
    int mbc = si.mb_count;
    int cb = mbc * 2; // chroma 422: 2 blocks per MB

    const uint8_t *plane_data;
    int plane_size;
    if (plane_offset == 0) {
        plane_data = sd + hs + yds;
        plane_size = uds;
    } else {
        int vds;
        if (hs > 7) vds = (sd[6]<<8)|sd[7];
        else vds = (int)si.size - yds - uds - hs;
        plane_data = sd + hs + yds + uds;
        plane_size = vds > 0 ? vds : 0;
    }

    __align__(16) int16_t blocks[16 * 64]; // max 16 chroma blocks (8 MBs * 2)
    zero_blocks_i16(blocks, cb);

    BitReader br;
    br_init(&br, plane_data, plane_size);
    if (si.mb_count == 8) {
        decode_dc_coeffs_lut_fixed<16>(&br, blocks, s_dc_cb, tid);
        decode_ac_coeffs_lut_fixed<4, 15, 1024>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid);
    } else {
        decode_dc_coeffs_lut(&br, blocks, cb, s_dc_cb, tid);
        decode_ac_coeffs_lut(&br, blocks, cb, s_scan_lane, s_run_cb, s_level_cb, tid);
    }

    for (int b = 0; b < cb; b++) {
        int16_t *blk = blocks + b * 64;
        idct_put_cq_shared(blk, s_qmat, qs, bits_per_component);
        int mb = b / 2, sub = b % 2;
        int bx = 0, by = sub * 8;
        int16_t *dst = out_plane + (si.mb_y * 16 + by) * stride_c + (si.mb_x + mb) * 8 + bx;
        store_block8x8_i16(dst, stride_c, blk);
    }
}

extern "C" __global__ void __launch_bounds__(32, 1) pr_decode_chroma422_both(
    const uint8_t *compressed, const SliceInfo *slice_info,
    int16_t *out_cb, int16_t *out_cr, int stride_c,
    int bits_per_component, int num_slices
) {
    __shared__ int s_qmat[64];
    __shared__ int s_scan_lane[64 * 32];
    __shared__ int s_dc_cb[128];
    __shared__ int s_run_cb[512];
    __shared__ int s_level_cb[320];
    int tid = threadIdx.x;
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i];
    s_qmat[tid] = c_chroma_qmat[tid];
    s_qmat[tid + 32] = c_chroma_qmat[tid + 32];
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb);
    __syncthreads();

    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= num_slices) return;

    SliceInfo si = slice_info[work_idx];
    const uint8_t *sd = compressed + si.offset;
    int hs = sd[0] >> 3;
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; }
    int yds = (sd[2]<<8)|sd[3];
    int uds = (sd[4]<<8)|sd[5];
    int vds = (hs > 7) ? ((sd[6]<<8)|sd[7]) : ((int)si.size - yds - uds - hs);
    int cb = si.mb_count * 2;

    __align__(16) int16_t blocks[16 * 64];

    for (int plane = 0; plane < 2; plane++) {
        const uint8_t *plane_data = plane == 0 ? sd + hs + yds : sd + hs + yds + uds;
        int plane_size = plane == 0 ? uds : (vds > 0 ? vds : 0);
        int16_t *out_plane = plane == 0 ? out_cb : out_cr;

        zero_blocks_i16(blocks, cb);

        BitReader br;
        br_init(&br, plane_data, plane_size);
        if (si.mb_count == 8) {
            decode_dc_coeffs_lut_fixed<16>(&br, blocks, s_dc_cb, tid);
            decode_ac_coeffs_lut_fixed<4, 15, 1024>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid);
        } else {
            decode_dc_coeffs_lut(&br, blocks, cb, s_dc_cb, tid);
            decode_ac_coeffs_lut(&br, blocks, cb, s_scan_lane, s_run_cb, s_level_cb, tid);
        }

        for (int b = 0; b < cb; b++) {
            int16_t *blk = blocks + b * 64;
            idct_put_cq_shared(blk, s_qmat, qs, bits_per_component);
            int mb = b / 2, sub = b % 2;
            int by = sub * 8;
            int16_t *dst = out_plane + (si.mb_y * 16 + by) * stride_c + (si.mb_x + mb) * 8;
            store_block8x8_i16(dst, stride_c, blk);
        }
    }
}

#define DEFINE_PR_DECODE_LUMA_LANES(NAME, LANES) \
extern "C" __global__ void __launch_bounds__(32, 1) NAME( \
    const uint8_t *compressed, const SliceInfo *slice_info, \
    int16_t *out_y, int stride_y, \
    int bits_per_component, int num_slices \
) { \
    __shared__ int s_qmat[64]; \
    __shared__ int s_scan_lane[64 * 32]; \
    __shared__ int s_dc_cb[128]; \
    __shared__ int s_run_cb[512]; \
    __shared__ int s_level_cb[320]; \
    int tid = threadIdx.x; \
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i]; \
    s_qmat[tid] = c_luma_qmat[tid]; \
    s_qmat[tid + 32] = c_luma_qmat[tid + 32]; \
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb); \
    __syncthreads(); \
    if (tid >= LANES) return; \
    int work_idx = blockIdx.x * LANES + tid; \
    if (work_idx >= num_slices) return; \
    SliceInfo si = slice_info[work_idx]; \
    const uint8_t *sd = compressed + si.offset; \
    int hs = sd[0] >> 3; \
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; } \
    int yds = (sd[2]<<8)|sd[3]; \
    int mbc = si.mb_count; \
    int yb = mbc * 4; \
    __align__(16) int16_t blocks[32 * 64]; \
    zero_blocks_i16(blocks, yb); \
    BitReader br; \
    br_init(&br, sd + hs, yds); \
    if (mbc == 8) { \
        decode_dc_coeffs_lut_fixed<32>(&br, blocks, s_dc_cb, tid); \
        decode_ac_coeffs_lut_fixed<5, 31, 2048>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid); \
    } else { \
        decode_dc_coeffs_lut(&br, blocks, yb, s_dc_cb, tid); \
        decode_ac_coeffs_lut(&br, blocks, yb, s_scan_lane, s_run_cb, s_level_cb, tid); \
    } \
    for (int b = 0; b < yb; b++) { \
        int16_t *blk = blocks + b * 64; \
        idct_put_cq_shared(blk, s_qmat, qs, bits_per_component); \
        int mb = b / 4, sub = b % 4; \
        int bx = (sub & 1) * 8, by = (sub >> 1) * 8; \
        int16_t *dst = out_y + (si.mb_y * 16 + by) * stride_y + (si.mb_x + mb) * 16 + bx; \
        store_block8x8_i16(dst, stride_y, blk); \
    } \
}

#define DEFINE_PR_DECODE_CHROMA422_BOTH_LANES(NAME, LANES) \
extern "C" __global__ void __launch_bounds__(32, 1) NAME( \
    const uint8_t *compressed, const SliceInfo *slice_info, \
    int16_t *out_cb, int16_t *out_cr, int stride_c, \
    int bits_per_component, int num_slices \
) { \
    __shared__ int s_qmat[64]; \
    __shared__ int s_scan_lane[64 * 32]; \
    __shared__ int s_dc_cb[128]; \
    __shared__ int s_run_cb[512]; \
    __shared__ int s_level_cb[320]; \
    int tid = threadIdx.x; \
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i]; \
    s_qmat[tid] = c_chroma_qmat[tid]; \
    s_qmat[tid + 32] = c_chroma_qmat[tid + 32]; \
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb); \
    __syncthreads(); \
    if (tid >= LANES) return; \
    int work_idx = blockIdx.x * LANES + tid; \
    if (work_idx >= num_slices) return; \
    SliceInfo si = slice_info[work_idx]; \
    const uint8_t *sd = compressed + si.offset; \
    int hs = sd[0] >> 3; \
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; } \
    int yds = (sd[2]<<8)|sd[3]; \
    int uds = (sd[4]<<8)|sd[5]; \
    int vds = (hs > 7) ? ((sd[6]<<8)|sd[7]) : ((int)si.size - yds - uds - hs); \
    int cb = si.mb_count * 2; \
    __align__(16) int16_t blocks[16 * 64]; \
    for (int plane = 0; plane < 2; plane++) { \
        const uint8_t *plane_data = plane == 0 ? sd + hs + yds : sd + hs + yds + uds; \
        int plane_size = plane == 0 ? uds : (vds > 0 ? vds : 0); \
        int16_t *out_plane = plane == 0 ? out_cb : out_cr; \
        zero_blocks_i16(blocks, cb); \
        BitReader br; \
        br_init(&br, plane_data, plane_size); \
        if (si.mb_count == 8) { \
            decode_dc_coeffs_lut_fixed<16>(&br, blocks, s_dc_cb, tid); \
            decode_ac_coeffs_lut_fixed<4, 15, 1024>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid); \
        } else { \
            decode_dc_coeffs_lut(&br, blocks, cb, s_dc_cb, tid); \
            decode_ac_coeffs_lut(&br, blocks, cb, s_scan_lane, s_run_cb, s_level_cb, tid); \
        } \
        for (int b = 0; b < cb; b++) { \
            int16_t *blk = blocks + b * 64; \
            idct_put_cq_shared(blk, s_qmat, qs, bits_per_component); \
            int mb = b / 2, sub = b % 2; \
            int by = sub * 8; \
            int16_t *dst = out_plane + (si.mb_y * 16 + by) * stride_c + (si.mb_x + mb) * 8; \
            store_block8x8_i16(dst, stride_c, blk); \
        } \
    } \
}

DEFINE_PR_DECODE_LUMA_LANES(pr_decode_luma_lanes16, 16)
DEFINE_PR_DECODE_LUMA_LANES(pr_decode_luma_lanes8, 8)
DEFINE_PR_DECODE_CHROMA422_BOTH_LANES(pr_decode_chroma422_both_lanes16, 16)
DEFINE_PR_DECODE_CHROMA422_BOTH_LANES(pr_decode_chroma422_both_lanes8, 8)

// 444 chroma lane-parallel: 4 blocks/MB, same row-major placement as luma/alpha.
#define DEFINE_PR_DECODE_CHROMA444_BOTH_LANES(NAME, LANES) \
extern "C" __global__ void __launch_bounds__(32, 1) NAME( \
    const uint8_t *compressed, const SliceInfo *slice_info, \
    int16_t *out_cb, int16_t *out_cr, int stride_c, \
    int bits_per_component, int num_slices \
) { \
    __shared__ int s_qmat[64]; \
    __shared__ int s_scan_lane[64 * 32]; \
    __shared__ int s_dc_cb[128]; \
    __shared__ int s_run_cb[512]; \
    __shared__ int s_level_cb[320]; \
    int tid = threadIdx.x; \
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i]; \
    s_qmat[tid] = c_chroma_qmat[tid]; \
    s_qmat[tid + 32] = c_chroma_qmat[tid + 32]; \
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb); \
    __syncthreads(); \
    if (tid >= LANES) return; \
    int work_idx = blockIdx.x * LANES + tid; \
    if (work_idx >= num_slices) return; \
    SliceInfo si = slice_info[work_idx]; \
    const uint8_t *sd = compressed + si.offset; \
    int hs = sd[0] >> 3; \
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; } \
    int yds = (sd[2]<<8)|sd[3]; \
    int uds = (sd[4]<<8)|sd[5]; \
    int vds = (hs > 7) ? ((sd[6]<<8)|sd[7]) : ((int)si.size - yds - uds - hs); \
    int cb = si.mb_count * 4; \
    __align__(16) int16_t blocks[32 * 64]; \
    for (int plane = 0; plane < 2; plane++) { \
        const uint8_t *plane_data = plane == 0 ? sd + hs + yds : sd + hs + yds + uds; \
        int plane_size = plane == 0 ? uds : (vds > 0 ? vds : 0); \
        int16_t *out_plane = plane == 0 ? out_cb : out_cr; \
        zero_blocks_i16(blocks, cb); \
        BitReader br; \
        br_init(&br, plane_data, plane_size); \
        if (si.mb_count == 8) { \
            decode_dc_coeffs_lut_fixed<32>(&br, blocks, s_dc_cb, tid); \
            decode_ac_coeffs_lut_fixed<5, 31, 2048>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid); \
        } else { \
            decode_dc_coeffs_lut(&br, blocks, cb, s_dc_cb, tid); \
            decode_ac_coeffs_lut(&br, blocks, cb, s_scan_lane, s_run_cb, s_level_cb, tid); \
        } \
        for (int b = 0; b < cb; b++) { \
            int16_t *blk = blocks + b * 64; \
            idct_put_cq_shared(blk, s_qmat, qs, bits_per_component); \
            int mb = b / 4, sub = b % 4; \
            int bx, by; \
            prores_chroma444_block_pos(sub, &bx, &by); \
            int16_t *dst = out_plane + (si.mb_y * 16 + by) * stride_c + (si.mb_x + mb) * 16 + bx; \
            store_block8x8_i16(dst, stride_c, blk); \
        } \
    } \
}

DEFINE_PR_DECODE_CHROMA444_BOTH_LANES(pr_decode_chroma444_both_lanes16, 16)
DEFINE_PR_DECODE_CHROMA444_BOTH_LANES(pr_decode_chroma444_both_lanes8, 8)

__device__ __forceinline__ int cupr_scale_alpha(int alpha, int alpha_bits, int bit_depth) {
    const int alpha_max = (1 << alpha_bits) - 1;
    alpha = max(0, min(alpha_max, alpha));
    if (alpha_bits > bit_depth) {
        return alpha >> (alpha_bits - bit_depth);
    } else if (alpha_bits < bit_depth) {
        const int shift = bit_depth - alpha_bits;
        return (alpha << shift) | (alpha >> max(1, alpha_bits - shift));
    }
    return alpha;
}

__device__ __forceinline__ void store_alpha_sample_i16(
    int16_t *out_alpha, int stride_a, int width, int height,
    const SliceInfo& si, int sample_idx, int value) {
    const int slice_width = si.mb_count * 16;
    const int x = si.mb_x * 16 + (sample_idx % slice_width);
    const int y = si.mb_y * 16 + (sample_idx / slice_width);
    if (x < width && y < height) {
        out_alpha[y * stride_a + x] = (int16_t)value;
    }
}

// 4444 alpha is delta/RLE packed, not DCT-coded like Y/Cb/Cr.
#define DEFINE_PR_DECODE_ALPHA444_LANES(NAME, LANES) \
extern "C" __global__ void __launch_bounds__(32, 1) NAME( \
    const uint8_t *compressed, const SliceInfo *slice_info, \
    int16_t *out_alpha, int stride_a, \
    int width, int height, int bits_per_component, int alpha_info, int num_slices \
) { \
    int tid = threadIdx.x; \
    if (tid >= LANES) return; \
    int work_idx = blockIdx.x * LANES + tid; \
    if (work_idx >= num_slices) return; \
    SliceInfo si = slice_info[work_idx]; \
    const uint8_t *sd = compressed + si.offset; \
    int hs = sd[0] >> 3; \
    int yds = (sd[2]<<8)|sd[3]; \
    int uds = (sd[4]<<8)|sd[5]; \
    int vds = (hs > 7) ? ((sd[6]<<8)|sd[7]) : ((int)si.size - yds - uds - hs); \
    int ads = (hs > 9) ? ((sd[8]<<8)|sd[9]) : 0; \
    const int alpha_bits = (alpha_info == 2) ? 16 : 8; \
    const int alpha_mask = (1 << alpha_bits) - 1; \
    const int sample_count = si.mb_count * 16 * 16; \
    int sample_idx = 0; \
    int alpha_val = alpha_mask; \
    BitReader br; \
    br_init(&br, sd + hs + yds + uds + vds, ads); \
    while (sample_idx < sample_count && br_bits_left(&br) > 0) { \
        for (;;) { \
            if (br_bits_left(&br) <= 0) break; \
            int diff; \
            if (br_read1(&br)) { \
                diff = br_read(&br, alpha_bits); \
            } else { \
                int code = br_read(&br, alpha_bits == 16 ? 7 : 4); \
                const int sign = code & 1; \
                diff = (code + 2) >> 1; \
                if (sign) diff = -diff; \
            } \
            alpha_val = (alpha_val + diff) & alpha_mask; \
            store_alpha_sample_i16(out_alpha, stride_a, width, height, si, sample_idx++, cupr_scale_alpha(alpha_val, alpha_bits, bits_per_component)); \
            if (sample_idx >= sample_count || br_bits_left(&br) <= 0 || !br_read1(&br)) break; \
        } \
        if (sample_idx >= sample_count || br_bits_left(&br) <= 0) break; \
        int run = br_read(&br, 4); \
        if (run == 0) { \
            run = br_read(&br, 11); \
        } \
        const int value = cupr_scale_alpha(alpha_val, alpha_bits, bits_per_component); \
        for (int i = 0; i < run && sample_idx < sample_count; i++) { \
            store_alpha_sample_i16(out_alpha, stride_a, width, height, si, sample_idx++, value); \
        } \
    } \
}

DEFINE_PR_DECODE_ALPHA444_LANES(pr_decode_alpha444_lanes16, 16)
DEFINE_PR_DECODE_ALPHA444_LANES(pr_decode_alpha444_lanes8, 8)

extern "C" __global__ void __launch_bounds__(32, 1) pr_decode_luma_lanes8_smem(
    const uint8_t *compressed, const SliceInfo *slice_info,
    int16_t *out_y, int stride_y,
    int bits_per_component, int num_slices
) {
    __shared__ int s_qmat[64];
    __shared__ int s_scan_lane[64 * 32];
    __shared__ int s_dc_cb[128];
    __shared__ int s_run_cb[512];
    __shared__ int s_level_cb[320];
    __shared__ int4 s_block_storage[8 * 32 * 64 / 8];
    int tid = threadIdx.x;
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i];
    s_qmat[tid] = c_luma_qmat[tid];
    s_qmat[tid + 32] = c_luma_qmat[tid + 32];
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb);
    __syncthreads();
    if (tid >= 8) return;

    int work_idx = blockIdx.x * 8 + tid;
    if (work_idx >= num_slices) return;

    SliceInfo si = slice_info[work_idx];
    const uint8_t *sd = compressed + si.offset;
    int hs = sd[0] >> 3;
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; }
    int yds = (sd[2]<<8)|sd[3];
    int mbc = si.mb_count;
    int yb = mbc * 4;
    int16_t *blocks = reinterpret_cast<int16_t *>(s_block_storage) + tid * 32 * 64;
    zero_blocks_i16(blocks, yb);

    BitReader br;
    br_init(&br, sd + hs, yds);
    if (mbc == 8) {
        decode_dc_coeffs_lut_fixed<32>(&br, blocks, s_dc_cb, tid);
        decode_ac_coeffs_lut_fixed<5, 31, 2048>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid);
    } else {
        decode_dc_coeffs_lut(&br, blocks, yb, s_dc_cb, tid);
        decode_ac_coeffs_lut(&br, blocks, yb, s_scan_lane, s_run_cb, s_level_cb, tid);
    }

    for (int b = 0; b < yb; b++) {
        int16_t *blk = blocks + b * 64;
        idct_put_cq_shared(blk, s_qmat, qs, bits_per_component);
        int mb = b / 4, sub = b % 4;
        int bx = (sub & 1) * 8, by = (sub >> 1) * 8;
        int16_t *dst = out_y + (si.mb_y * 16 + by) * stride_y + (si.mb_x + mb) * 16 + bx;
        store_block8x8_i16(dst, stride_y, blk);
    }
}

extern "C" __global__ void __launch_bounds__(32, 1) pr_decode_chroma422_both_lanes8_smem(
    const uint8_t *compressed, const SliceInfo *slice_info,
    int16_t *out_cb, int16_t *out_cr, int stride_c,
    int bits_per_component, int num_slices
) {
    __shared__ int s_qmat[64];
    __shared__ int s_scan_lane[64 * 32];
    __shared__ int s_dc_cb[128];
    __shared__ int s_run_cb[512];
    __shared__ int s_level_cb[320];
    __shared__ int4 s_block_storage[8 * 16 * 64 / 8];
    int tid = threadIdx.x;
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i];
    s_qmat[tid] = c_chroma_qmat[tid];
    s_qmat[tid + 32] = c_chroma_qmat[tid + 32];
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb);
    __syncthreads();
    if (tid >= 8) return;

    int work_idx = blockIdx.x * 8 + tid;
    if (work_idx >= num_slices) return;

    SliceInfo si = slice_info[work_idx];
    const uint8_t *sd = compressed + si.offset;
    int hs = sd[0] >> 3;
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; }
    int yds = (sd[2]<<8)|sd[3];
    int uds = (sd[4]<<8)|sd[5];
    int vds = (hs > 7) ? ((sd[6]<<8)|sd[7]) : ((int)si.size - yds - uds - hs);
    int cb = si.mb_count * 2;
    int16_t *blocks = reinterpret_cast<int16_t *>(s_block_storage) + tid * 16 * 64;

    for (int plane = 0; plane < 2; plane++) {
        const uint8_t *plane_data = plane == 0 ? sd + hs + yds : sd + hs + yds + uds;
        int plane_size = plane == 0 ? uds : (vds > 0 ? vds : 0);
        int16_t *out_plane = plane == 0 ? out_cb : out_cr;
        zero_blocks_i16(blocks, cb);

        BitReader br;
        br_init(&br, plane_data, plane_size);
        if (si.mb_count == 8) {
            decode_dc_coeffs_lut_fixed<16>(&br, blocks, s_dc_cb, tid);
            decode_ac_coeffs_lut_fixed<4, 15, 1024>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, tid);
        } else {
            decode_dc_coeffs_lut(&br, blocks, cb, s_dc_cb, tid);
            decode_ac_coeffs_lut(&br, blocks, cb, s_scan_lane, s_run_cb, s_level_cb, tid);
        }

        for (int b = 0; b < cb; b++) {
            int16_t *blk = blocks + b * 64;
            idct_put_cq_shared(blk, s_qmat, qs, bits_per_component);
            int mb = b / 2, sub = b % 2;
            int by = sub * 8;
            int16_t *dst = out_plane + (si.mb_y * 16 + by) * stride_c + (si.mb_x + mb) * 8;
            store_block8x8_i16(dst, stride_c, blk);
        }
    }
}

extern "C" __global__ void __launch_bounds__(32, 1) pr_decode_422_fused(
    const uint8_t *compressed, const SliceInfo *slice_info,
    int16_t *out_y, int16_t *out_cb, int16_t *out_cr,
    int stride_y, int stride_c,
    int bits_per_component, int num_slices
) {
    __shared__ int s_luma_qmat[64];
    __shared__ int s_chroma_qmat[64];
    __shared__ int s_scan_lane[64 * 32];
    __shared__ int s_dc_cb[128];
    __shared__ int s_run_cb[512];
    __shared__ int s_level_cb[320];
    int tid = threadIdx.x;
    int lane = tid;
    for (int i = 0; i < 64; i++) s_scan_lane[i * 32 + tid] = c_scan[i];
    s_luma_qmat[tid] = c_luma_qmat[tid];
    s_luma_qmat[tid + 32] = c_luma_qmat[tid + 32];
    s_chroma_qmat[tid] = c_chroma_qmat[tid];
    s_chroma_qmat[tid + 32] = c_chroma_qmat[tid + 32];
    init_entropy_luts(tid, s_dc_cb, s_run_cb, s_level_cb);
    __syncthreads();

    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_idx >= num_slices) return;

    SliceInfo si = slice_info[work_idx];
    const uint8_t *sd = compressed + si.offset;
    int hs = sd[0] >> 3;
    int qs; { int rq=sd[1]; if(rq<1)rq=1; if(rq>224)rq=224; qs=rq>128?(rq-96)<<2:rq; }
    int yds = (sd[2]<<8)|sd[3];
    int uds = (sd[4]<<8)|sd[5];
    int vds = (hs > 7) ? ((sd[6]<<8)|sd[7]) : ((int)si.size - yds - uds - hs);
    int mbc = si.mb_count;
    int yb = mbc * 4;
    int cb = mbc * 2;

    __align__(16) int16_t blocks[32 * 64];

    zero_blocks_i16(blocks, yb);
    BitReader br_y;
    br_init(&br_y, sd + hs, yds);
    if (mbc == 8) {
        decode_dc_coeffs_lut_fixed<32>(&br_y, blocks, s_dc_cb, lane);
        decode_ac_coeffs_lut_fixed<5, 31, 2048>(&br_y, blocks, s_scan_lane, s_run_cb, s_level_cb, lane);
    } else {
        decode_dc_coeffs_lut(&br_y, blocks, yb, s_dc_cb, lane);
        decode_ac_coeffs_lut(&br_y, blocks, yb, s_scan_lane, s_run_cb, s_level_cb, lane);
    }
    for (int b = 0; b < yb; b++) {
        int16_t *blk = blocks + b * 64;
        idct_put_cq_shared(blk, s_luma_qmat, qs, bits_per_component);
        int mb = b / 4, sub = b % 4;
        int bx = (sub & 1) * 8, by = (sub >> 1) * 8;
        int16_t *dst = out_y + (si.mb_y * 16 + by) * stride_y + (si.mb_x + mb) * 16 + bx;
        store_block8x8_i16(dst, stride_y, blk);
    }

    for (int plane = 0; plane < 2; plane++) {
        const uint8_t *plane_data = plane == 0 ? sd + hs + yds : sd + hs + yds + uds;
        int plane_size = plane == 0 ? uds : (vds > 0 ? vds : 0);
        int16_t *out_plane = plane == 0 ? out_cb : out_cr;

        zero_blocks_i16(blocks, cb);
        BitReader br;
        br_init(&br, plane_data, plane_size);
        if (mbc == 8) {
            decode_dc_coeffs_lut_fixed<16>(&br, blocks, s_dc_cb, lane);
            decode_ac_coeffs_lut_fixed<4, 15, 1024>(&br, blocks, s_scan_lane, s_run_cb, s_level_cb, lane);
        } else {
            decode_dc_coeffs_lut(&br, blocks, cb, s_dc_cb, lane);
            decode_ac_coeffs_lut(&br, blocks, cb, s_scan_lane, s_run_cb, s_level_cb, lane);
        }
        for (int b = 0; b < cb; b++) {
            int16_t *blk = blocks + b * 64;
            idct_put_cq_shared(blk, s_chroma_qmat, qs, bits_per_component);
            int mb = b / 2, sub = b % 2;
            int by = sub * 8;
            int16_t *dst = out_plane + (si.mb_y * 16 + by) * stride_c + (si.mb_x + mb) * 8;
            store_block8x8_i16(dst, stride_c, blk);
        }
    }
}
