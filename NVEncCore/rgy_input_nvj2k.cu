#include "rgy_input_nvj2k_kernels.h"

__device__ __forceinline__ int nvj2k_clamp_int(int v, int lo, int hi) {
    return max(lo, min(hi, v));
}

__device__ __forceinline__ int nvj2k_read_fullres(const NvJ2kPlane plane, int full_w, int full_h, int x, int y) {
    const int sx = nvj2k_clamp_int((int)(((long long)x * plane.width) / full_w), 0, plane.width - 1);
    const int sy = nvj2k_clamp_int((int)(((long long)y * plane.height) / full_h), 0, plane.height - 1);
    const auto line = (const uint16_t *)((const uint8_t *)plane.ptr + (size_t)sy * plane.pitch);
    return (int)line[sx];
}

__device__ __forceinline__ uint8_t nvj2k_to_u8(int v, int precision) {
    if (precision > 8) {
        const int shift = precision - 8;
        const int round = 1 << (shift - 1);
        return (uint8_t)nvj2k_clamp_int((v + round) >> shift, 0, 255);
    }
    return (uint8_t)nvj2k_clamp_int(v << (8 - precision), 0, 255);
}

__device__ __forceinline__ uint16_t nvj2k_to_u16_msb(int v, int precision) {
    if (precision >= 16) {
        return (uint16_t)nvj2k_clamp_int(v, 0, 65535);
    }
    return (uint16_t)nvj2k_clamp_int(v << (16 - precision), 0, 65535);
}

__device__ __forceinline__ int nvj2k_avg2(int a, int b) {
    return (a + b + 1) >> 1;
}

__device__ __forceinline__ int nvj2k_avg4(int a, int b, int c, int d) {
    return (a + b + c + d + 2) >> 2;
}

__global__ void nvj2k_pack_nv12_kernel(
    NvJ2kPlane s0,
    NvJ2kPlane s1,
    NvJ2kPlane s2,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    int pitch_y,
    int pitch_uv,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        dst_y[(size_t)y * pitch_y + x] = nvj2k_to_u8(nvj2k_read_fullres(s0, width, height, x, y), s0.precision);
    }

    if ((x & 1) == 0 && x < width && y < ((height + 1) >> 1)) {
        const int y0 = y << 1;
        const int y1 = min(y0 + 1, height - 1);
        const int x1 = min(x + 1, width - 1);
        const int cb = nvj2k_avg4(
            nvj2k_read_fullres(s1, width, height, x,  y0),
            nvj2k_read_fullres(s1, width, height, x1, y0),
            nvj2k_read_fullres(s1, width, height, x,  y1),
            nvj2k_read_fullres(s1, width, height, x1, y1));
        const int cr = nvj2k_avg4(
            nvj2k_read_fullres(s2, width, height, x,  y0),
            nvj2k_read_fullres(s2, width, height, x1, y0),
            nvj2k_read_fullres(s2, width, height, x,  y1),
            nvj2k_read_fullres(s2, width, height, x1, y1));
        auto uv = dst_uv + (size_t)y * pitch_uv + x;
        uv[0] = nvj2k_to_u8(cb, s1.precision);
        uv[1] = nvj2k_to_u8(cr, s2.precision);
    }
}

__global__ void nvj2k_pack_p010_kernel(
    NvJ2kPlane s0,
    NvJ2kPlane s1,
    NvJ2kPlane s2,
    uint16_t *dst_y,
    uint16_t *dst_uv,
    int pitch_y,
    int pitch_uv,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        dst_y[(size_t)y * pitch_y + x] = nvj2k_to_u16_msb(nvj2k_read_fullres(s0, width, height, x, y), s0.precision);
    }

    if ((x & 1) == 0 && x < width && y < ((height + 1) >> 1)) {
        const int y0 = y << 1;
        const int y1 = min(y0 + 1, height - 1);
        const int x1 = min(x + 1, width - 1);
        const int cb = nvj2k_avg4(
            nvj2k_read_fullres(s1, width, height, x,  y0),
            nvj2k_read_fullres(s1, width, height, x1, y0),
            nvj2k_read_fullres(s1, width, height, x,  y1),
            nvj2k_read_fullres(s1, width, height, x1, y1));
        const int cr = nvj2k_avg4(
            nvj2k_read_fullres(s2, width, height, x,  y0),
            nvj2k_read_fullres(s2, width, height, x1, y0),
            nvj2k_read_fullres(s2, width, height, x,  y1),
            nvj2k_read_fullres(s2, width, height, x1, y1));
        auto uv = dst_uv + (size_t)y * pitch_uv + x;
        uv[0] = nvj2k_to_u16_msb(cb, s1.precision);
        uv[1] = nvj2k_to_u16_msb(cr, s2.precision);
    }
}

__global__ void nvj2k_pack_nv16_kernel(
    NvJ2kPlane s0,
    NvJ2kPlane s1,
    NvJ2kPlane s2,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    int pitch_y,
    int pitch_uv,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    dst_y[(size_t)y * pitch_y + x] = nvj2k_to_u8(nvj2k_read_fullres(s0, width, height, x, y), s0.precision);
    if ((x & 1) == 0) {
        const int x1 = min(x + 1, width - 1);
        const int cb = nvj2k_avg2(nvj2k_read_fullres(s1, width, height, x, y), nvj2k_read_fullres(s1, width, height, x1, y));
        const int cr = nvj2k_avg2(nvj2k_read_fullres(s2, width, height, x, y), nvj2k_read_fullres(s2, width, height, x1, y));
        auto uv = dst_uv + (size_t)y * pitch_uv + x;
        uv[0] = nvj2k_to_u8(cb, s1.precision);
        uv[1] = nvj2k_to_u8(cr, s2.precision);
    }
}

__global__ void nvj2k_pack_p210_kernel(
    NvJ2kPlane s0,
    NvJ2kPlane s1,
    NvJ2kPlane s2,
    uint16_t *dst_y,
    uint16_t *dst_uv,
    int pitch_y,
    int pitch_uv,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    dst_y[(size_t)y * pitch_y + x] = nvj2k_to_u16_msb(nvj2k_read_fullres(s0, width, height, x, y), s0.precision);
    if ((x & 1) == 0) {
        const int x1 = min(x + 1, width - 1);
        const int cb = nvj2k_avg2(nvj2k_read_fullres(s1, width, height, x, y), nvj2k_read_fullres(s1, width, height, x1, y));
        const int cr = nvj2k_avg2(nvj2k_read_fullres(s2, width, height, x, y), nvj2k_read_fullres(s2, width, height, x1, y));
        auto uv = dst_uv + (size_t)y * pitch_uv + x;
        uv[0] = nvj2k_to_u16_msb(cb, s1.precision);
        uv[1] = nvj2k_to_u16_msb(cr, s2.precision);
    }
}

__global__ void nvj2k_pack_planar8_kernel(
    NvJ2kPlane s0,
    NvJ2kPlane s1,
    NvJ2kPlane s2,
    uint8_t *dst0,
    uint8_t *dst1,
    uint8_t *dst2,
    int pitch0,
    int pitch1,
    int pitch2,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    dst0[(size_t)y * pitch0 + x] = nvj2k_to_u8(nvj2k_read_fullres(s0, width, height, x, y), s0.precision);
    dst1[(size_t)y * pitch1 + x] = nvj2k_to_u8(nvj2k_read_fullres(s1, width, height, x, y), s1.precision);
    dst2[(size_t)y * pitch2 + x] = nvj2k_to_u8(nvj2k_read_fullres(s2, width, height, x, y), s2.precision);
}

__global__ void nvj2k_pack_planar16_kernel(
    NvJ2kPlane s0,
    NvJ2kPlane s1,
    NvJ2kPlane s2,
    uint16_t *dst0,
    uint16_t *dst1,
    uint16_t *dst2,
    int pitch0,
    int pitch1,
    int pitch2,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    dst0[(size_t)y * pitch0 + x] = nvj2k_to_u16_msb(nvj2k_read_fullres(s0, width, height, x, y), s0.precision);
    dst1[(size_t)y * pitch1 + x] = nvj2k_to_u16_msb(nvj2k_read_fullres(s1, width, height, x, y), s1.precision);
    dst2[(size_t)y * pitch2 + x] = nvj2k_to_u16_msb(nvj2k_read_fullres(s2, width, height, x, y), s2.precision);
}

// XYZ(DCI) -> linear BT.709 RGB -> BT.1886 (pure 2.4 power) encoded RGB, output as uint16 MSB-aligned planar.
// Why BT.1886 and not BT.709 OETF: DCP-o-matic and most BT.709-to-DCP packagers assume
// a BT.1886 (gamma 2.4) input characteristic on the source side. Using the BT.709 OETF
// here (2.222 + linear toe) would crush the shadows slightly compared to the source,
// because the encoding function is not the mathematical inverse of the decoding function
// the packager used. BT.1886 gives a proper round-trip.
__device__ __forceinline__ float nvj2k_xyz_bt1886_encode(float L) {
    // BT.1886 OETF = x^(1/2.4). Pure power law, no linear toe.
    if (L <= 0.0f) return 0.0f;
    return __powf(L, 1.0f / 2.4f);
}

__global__ void nvj2k_pack_xyz_to_bt709_kernel(
    NvJ2kPlane sX,
    NvJ2kPlane sY,
    NvJ2kPlane sZ,
    uint16_t *dstR,
    uint16_t *dstG,
    uint16_t *dstB,
    int pitchR,
    int pitchG,
    int pitchB,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    // Composed matrix: XYZ(D65) -> linear BT.709 RGB.
    // No chromatic adaptation is applied because DCP-o-matic (and typical RGB->DCP
    // packagers) encode the scene under a D65-referenced illuminant: BT.709 D65 primaries
    // are mapped directly through the BT.709 RGB->XYZ matrix, with no DCI-white rotation.
    // The DCI 2.6 gamma decode and 52.37/48 full-scale factor are applied separately below.
    const float m00 = +3.24096994f, m01 = -1.53738318f, m02 = -0.49861076f;
    const float m10 = -0.96924364f, m11 = +1.87596750f, m12 = +0.04155506f;
    const float m20 = +0.05563008f, m21 = -0.20397696f, m22 = +1.05697151f;

    const int codeX = nvj2k_read_fullres(sX, width, height, x, y);
    const int codeY = nvj2k_read_fullres(sY, width, height, x, y);
    const int codeZ = nvj2k_read_fullres(sZ, width, height, x, y);

    // DCP codes are 12-bit. nvJPEG2000 reports precision==12 for xyz12 streams.
    // If precision differs, fall back to the plane's precision.
    const float normX = __fmul_rn((float)codeX, 1.0f / ((1 << sX.precision) - 1));
    const float normY = __fmul_rn((float)codeY, 1.0f / ((1 << sY.precision) - 1));
    const float normZ = __fmul_rn((float)codeZ, 1.0f / ((1 << sZ.precision) - 1));

    // DCI 2.6 gamma decode + full-scale 52.37/48 factor.
    const float kDciScale = 52.37f / 48.0f;
    const float linX = __powf(normX, 2.6f) * kDciScale;
    const float linY = __powf(normY, 2.6f) * kDciScale;
    const float linZ = __powf(normZ, 2.6f) * kDciScale;

    // 3x3 matrix mul: XYZ(DCI) -> linear BT.709 RGB.
    const float linR = m00 * linX + m01 * linY + m02 * linZ;
    const float linG = m10 * linX + m11 * linY + m12 * linZ;
    const float linB = m20 * linX + m21 * linY + m22 * linZ;

    // Clamp to [0, 1] (gamut compression is out-of-scope; clipping is simple and predictable).
    const float cR = fminf(fmaxf(linR, 0.0f), 1.0f);
    const float cG = fminf(fmaxf(linG, 0.0f), 1.0f);
    const float cB = fminf(fmaxf(linB, 0.0f), 1.0f);

    // BT.1886 (pure 2.4 power) encode to match DCP-o-matic's assumed source characteristic.
    const float eR = nvj2k_xyz_bt1886_encode(cR);
    const float eG = nvj2k_xyz_bt1886_encode(cG);
    const float eB = nvj2k_xyz_bt1886_encode(cB);

    // Output as full-range 16-bit.
    dstR[(size_t)y * pitchR + x] = (uint16_t)nvj2k_clamp_int((int)__float2int_rn(eR * 65535.0f), 0, 65535);
    dstG[(size_t)y * pitchG + x] = (uint16_t)nvj2k_clamp_int((int)__float2int_rn(eG * 65535.0f), 0, 65535);
    dstB[(size_t)y * pitchB + x] = (uint16_t)nvj2k_clamp_int((int)__float2int_rn(eB * 65535.0f), 0, 65535);
}

cudaError_t nvj2k_convert_to_surface_async(
    const NvJ2kPlane src[3],
    uint8_t *dst0,
    uint8_t *dst1,
    uint8_t *dst2,
    int dst_pitch0,
    int dst_pitch1,
    int dst_pitch2,
    int width,
    int height,
    RGY_CSP dst_csp,
    bool xyz_input,
    cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    if (xyz_input) {
        // DCP XYZ input must be gamma-decoded and matrixed to BT.709 RGB before packing.
        // Only RGB_16 output is supported for XYZ input (12-bit XYZ would lose too much
        // precision when re-encoded to 8-bit after gamma).
        if (dst_csp != RGY_CSP_RGB_16 && dst_csp != RGY_CSP_GBR_16) {
            return cudaErrorInvalidValue;
        }
        nvj2k_pack_xyz_to_bt709_kernel<<<grid, block, 0, stream>>>(
            src[0], src[1], src[2],
            (uint16_t *)dst0, (uint16_t *)dst1, (uint16_t *)dst2,
            dst_pitch0 / 2, dst_pitch1 / 2, dst_pitch2 / 2,
            width, height);
        return cudaGetLastError();
    }
    switch (dst_csp) {
    case RGY_CSP_NV12:
        nvj2k_pack_nv12_kernel<<<grid, block, 0, stream>>>(src[0], src[1], src[2], dst0, dst1, dst_pitch0, dst_pitch1, width, height);
        break;
    case RGY_CSP_P010:
        nvj2k_pack_p010_kernel<<<grid, block, 0, stream>>>(src[0], src[1], src[2], (uint16_t *)dst0, (uint16_t *)dst1, dst_pitch0 / 2, dst_pitch1 / 2, width, height);
        break;
    case RGY_CSP_NV16:
        nvj2k_pack_nv16_kernel<<<grid, block, 0, stream>>>(src[0], src[1], src[2], dst0, dst1, dst_pitch0, dst_pitch1, width, height);
        break;
    case RGY_CSP_P210:
        nvj2k_pack_p210_kernel<<<grid, block, 0, stream>>>(src[0], src[1], src[2], (uint16_t *)dst0, (uint16_t *)dst1, dst_pitch0 / 2, dst_pitch1 / 2, width, height);
        break;
    case RGY_CSP_YUV444:
    case RGY_CSP_RGB:
    case RGY_CSP_GBR:
        nvj2k_pack_planar8_kernel<<<grid, block, 0, stream>>>(src[0], src[1], src[2], dst0, dst1, dst2, dst_pitch0, dst_pitch1, dst_pitch2, width, height);
        break;
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_16:
    case RGY_CSP_RGB_16:
    case RGY_CSP_GBR_16:
        nvj2k_pack_planar16_kernel<<<grid, block, 0, stream>>>(src[0], src[1], src[2], (uint16_t *)dst0, (uint16_t *)dst1, (uint16_t *)dst2, dst_pitch0 / 2, dst_pitch1 / 2, dst_pitch2 / 2, width, height);
        break;
    default:
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
