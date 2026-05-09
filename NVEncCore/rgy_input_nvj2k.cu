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
    cudaStream_t stream) {
    dim3 block(16, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
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
