#include "rgy_input_cupr_kernels.h"

#include "rgy_input_cupr_prores_decode.cu"

static_assert(sizeof(CuprSliceInfo) == sizeof(SliceInfo), "SliceInfo layout mismatch");

static const int CUPR_STRATEGY_LANE8 = 1;
static const int CUPR_STRATEGY_LANE16 = 2;
static const int CUPR_STRATEGY_DUAL = 3;
static const int CUPR_STRATEGY_WIDE = 4;

__device__ __forceinline__ int cupr_sample_c422_p210(const int16_t *src_c, int stride_c, int x, int y, int maxv) {
    return max(0, min(maxv, (int)src_c[y * stride_c + (x >> 1)]));
}
__device__ __forceinline__ int cupr_sample_c444_p210(const int16_t *src_c, int stride_c, int x, int y, int maxv) {
    const int a = max(0, min(maxv, (int)src_c[y * stride_c + x]));
    const int b = max(0, min(maxv, (int)src_c[y * stride_c + x + 1]));
    return (a + b + 1) >> 1;
}
__device__ __forceinline__ int cupr_sample_c422_420(const int16_t *src_c, int stride_c, int x, int y0, int y1, int maxv) {
    const int cx = x >> 1;
    const int a = max(0, min(maxv, (int)src_c[y0 * stride_c + cx]));
    const int b = max(0, min(maxv, (int)src_c[y1 * stride_c + cx]));
    return (a + b + 1) >> 1;
}
__device__ __forceinline__ int cupr_sample_c444_420(const int16_t *src_c, int stride_c, int x, int y0, int y1, int maxv) {
    const int a = max(0, min(maxv, (int)src_c[y0 * stride_c + x]));
    const int b = max(0, min(maxv, (int)src_c[y0 * stride_c + x + 1]));
    const int c = max(0, min(maxv, (int)src_c[y1 * stride_c + x]));
    const int d = max(0, min(maxv, (int)src_c[y1 * stride_c + x + 1]));
    return (a + b + c + d + 2) >> 2;
}

__global__ void cupr_pack_p210_kernel(
    const int16_t *src_y,
    const int16_t *src_cb,
    const int16_t *src_cr,
    uint16_t *dst_y,
    uint16_t *dst_uv,
    int src_stride_y,
    int src_stride_c,
    int dst_stride_y,
    int dst_stride_uv,
    int width,
    int height,
    int bit_depth,
    int chroma_format) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int shift = 16 - bit_depth;
    const int maxv = (1 << bit_depth) - 1;
    const int luma = max(0, min(maxv, (int)src_y[y * src_stride_y + x]));
    dst_y[y * dst_stride_y + x] = (uint16_t)(luma << shift);

    if ((x & 1) == 0) {
        int cb, cr;
        if (chroma_format == 3) {
            cb = cupr_sample_c444_p210(src_cb, src_stride_c, x, y, maxv);
            cr = cupr_sample_c444_p210(src_cr, src_stride_c, x, y, maxv);
        } else {
            cb = cupr_sample_c422_p210(src_cb, src_stride_c, x, y, maxv);
            cr = cupr_sample_c422_p210(src_cr, src_stride_c, x, y, maxv);
        }
        const int uv = y * dst_stride_uv + x;
        dst_uv[uv + 0] = (uint16_t)(cb << shift);
        dst_uv[uv + 1] = (uint16_t)(cr << shift);
    }
}

__global__ void cupr_pack_p010_kernel(
    const int16_t *src_y,
    const int16_t *src_cb,
    const int16_t *src_cr,
    uint16_t *dst_y,
    uint16_t *dst_uv,
    int src_stride_y,
    int src_stride_c,
    int dst_stride_y,
    int dst_stride_uv,
    int width,
    int height,
    int bit_depth,
    int chroma_format) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int maxv = (1 << bit_depth) - 1;
    const int shift = 16 - bit_depth;

    if (x < width && y < height) {
        const int luma = max(0, min(maxv, (int)src_y[y * src_stride_y + x]));
        dst_y[y * dst_stride_y + x] = (uint16_t)(luma << shift);
    }

    if ((x & 1) == 0 && x < width && y < ((height + 1) >> 1)) {
        const int y0 = y << 1;
        const int y1 = min(y0 + 1, height - 1);
        int cb, cr;
        if (chroma_format == 3) {
            cb = cupr_sample_c444_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr = cupr_sample_c444_420(src_cr, src_stride_c, x, y0, y1, maxv);
        } else {
            cb = cupr_sample_c422_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr = cupr_sample_c422_420(src_cr, src_stride_c, x, y0, y1, maxv);
        }
        const int uv = y * dst_stride_uv + x;
        dst_uv[uv + 0] = (uint16_t)(cb << shift);
        dst_uv[uv + 1] = (uint16_t)(cr << shift);
    }
}

__global__ void cupr_pack_nv12_kernel(
    const int16_t *src_y,
    const int16_t *src_cb,
    const int16_t *src_cr,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    int src_stride_y,
    int src_stride_c,
    int dst_stride_y,
    int dst_stride_uv,
    int width,
    int height,
    int bit_depth,
    int chroma_format) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int maxv = (1 << bit_depth) - 1;
    const int downshift = bit_depth - 8;
    const int round = downshift > 0 ? (1 << (downshift - 1)) : 0;

    if (x < width && y < height) {
        const int luma = max(0, min(maxv, (int)src_y[y * src_stride_y + x]));
        dst_y[y * dst_stride_y + x] = (uint8_t)max(0, min(255, (luma + round) >> downshift));
    }

    if ((x & 1) == 0 && x < width && y < ((height + 1) >> 1)) {
        const int y0 = y << 1;
        const int y1 = min(y0 + 1, height - 1);
        int cb10, cr10;
        if (chroma_format == 3) {
            cb10 = cupr_sample_c444_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr10 = cupr_sample_c444_420(src_cr, src_stride_c, x, y0, y1, maxv);
        } else {
            cb10 = cupr_sample_c422_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr10 = cupr_sample_c422_420(src_cr, src_stride_c, x, y0, y1, maxv);
        }
        const int uv = y * dst_stride_uv + x;
        dst_uv[uv + 0] = (uint8_t)max(0, min(255, (cb10 + round) >> downshift));
        dst_uv[uv + 1] = (uint8_t)max(0, min(255, (cr10 + round) >> downshift));
    }
}

cudaError_t cupr_upload_qmat_async(const uint8_t luma[64], const uint8_t chroma[64], cudaStream_t stream) {
    auto err = cudaMemcpyToSymbolAsync(c_luma_qmat, luma, 64, 0, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return cudaMemcpyToSymbolAsync(c_chroma_qmat, chroma, 64, 0, cudaMemcpyHostToDevice, stream);
}

__global__ void cupr_pack_nv12a_kernel(
    const int16_t *src_y,
    const int16_t *src_cb,
    const int16_t *src_cr,
    const int16_t *src_alpha,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    uint8_t *dst_alpha,
    int src_stride_y,
    int src_stride_c,
    int dst_stride_y,
    int dst_stride_uv,
    int width,
    int height,
    int bit_depth,
    int chroma_format) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int maxv = (1 << bit_depth) - 1;
    const int downshift = bit_depth - 8;
    const int round = downshift > 0 ? (1 << (downshift - 1)) : 0;

    if (x < width && y < height) {
        const int luma = max(0, min(maxv, (int)src_y[y * src_stride_y + x]));
        dst_y[y * dst_stride_y + x] = (uint8_t)max(0, min(255, (luma + round) >> downshift));
        const int alpha = max(0, min(maxv, (int)src_alpha[y * src_stride_y + x]));
        dst_alpha[y * dst_stride_y + x] = (uint8_t)max(0, min(255, (alpha + round) >> downshift));
    }

    if ((x & 1) == 0 && x < width && y < ((height + 1) >> 1)) {
        const int y0 = y << 1;
        const int y1 = min(y0 + 1, height - 1);
        int cb10, cr10;
        if (chroma_format == 3) {
            cb10 = cupr_sample_c444_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr10 = cupr_sample_c444_420(src_cr, src_stride_c, x, y0, y1, maxv);
        } else {
            cb10 = cupr_sample_c422_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr10 = cupr_sample_c422_420(src_cr, src_stride_c, x, y0, y1, maxv);
        }
        const int uv = y * dst_stride_uv + x;
        dst_uv[uv + 0] = (uint8_t)max(0, min(255, (cb10 + round) >> downshift));
        dst_uv[uv + 1] = (uint8_t)max(0, min(255, (cr10 + round) >> downshift));
    }
}

__global__ void cupr_pack_p010a_kernel(
    const int16_t *src_y,
    const int16_t *src_cb,
    const int16_t *src_cr,
    const int16_t *src_alpha,
    uint16_t *dst_y,
    uint16_t *dst_uv,
    uint16_t *dst_alpha,
    int src_stride_y,
    int src_stride_c,
    int dst_stride_y,
    int dst_stride_uv,
    int width,
    int height,
    int bit_depth,
    int chroma_format) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int maxv = (1 << bit_depth) - 1;
    const int shift = 16 - bit_depth;

    if (x < width && y < height) {
        const int luma = max(0, min(maxv, (int)src_y[y * src_stride_y + x]));
        dst_y[y * dst_stride_y + x] = (uint16_t)(luma << shift);
        const int alpha = max(0, min(maxv, (int)src_alpha[y * src_stride_y + x]));
        dst_alpha[y * dst_stride_y + x] = (uint16_t)(alpha << shift);
    }

    if ((x & 1) == 0 && x < width && y < ((height + 1) >> 1)) {
        const int y0 = y << 1;
        const int y1 = min(y0 + 1, height - 1);
        int cb, cr;
        if (chroma_format == 3) {
            cb = cupr_sample_c444_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr = cupr_sample_c444_420(src_cr, src_stride_c, x, y0, y1, maxv);
        } else {
            cb = cupr_sample_c422_420(src_cb, src_stride_c, x, y0, y1, maxv);
            cr = cupr_sample_c422_420(src_cr, src_stride_c, x, y0, y1, maxv);
        }
        const int uv = y * dst_stride_uv + x;
        dst_uv[uv + 0] = (uint16_t)(cb << shift);
        dst_uv[uv + 1] = (uint16_t)(cr << shift);
    }
}

static cudaError_t cupr_launch_decode_422(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    int strategy,
    int chroma_format,
    cudaStream_t stream) {
    const auto *slices = reinterpret_cast<const SliceInfo *>(d_slices);
    const int stride_y = width;
    const int stride_c = (chroma_format == 3) ? width : (width / 2);
    const int is_444 = (chroma_format == 3) ? 1 : 0;

    dim3 block(32, 1, 1);
    if (strategy == CUPR_STRATEGY_WIDE) {
        dim3 grid(num_slices, 1, 1);
        prores_decode_slice<<<grid, block, 4096, stream>>>(
            d_compressed, slices, d_y, d_cb, d_cr,
            stride_y, stride_c, width, height, is_444, bit_depth, num_slices);
        return cudaGetLastError();
    }

    if (is_444) {
        if (strategy == CUPR_STRATEGY_LANE16) {
            dim3 grid((num_slices + 15) / 16, 1, 1);
            pr_decode_luma_lanes16<<<grid, block, 0, stream>>>(
                d_compressed, slices, d_y, stride_y, bit_depth, num_slices);
            auto err = cudaGetLastError();
            if (err != cudaSuccess) return err;
            pr_decode_chroma444_both_lanes16<<<grid, block, 0, stream>>>(
                d_compressed, slices, d_cb, d_cr, stride_c, bit_depth, num_slices);
            return cudaGetLastError();
        }
        if (strategy == CUPR_STRATEGY_DUAL) {
            dim3 grid((num_slices + 31) / 32, 1, 1);
            pr_decode_luma<<<grid, block, 0, stream>>>(
                d_compressed, slices, d_y, stride_y, bit_depth, num_slices);
            auto err = cudaGetLastError();
            if (err != cudaSuccess) return err;
            pr_decode_chroma444_both_lanes8<<<grid, block, 0, stream>>>(
                d_compressed, slices, d_cb, d_cr, stride_c, bit_depth, num_slices);
            return cudaGetLastError();
        }
        // LANE8 (default)
        dim3 grid((num_slices + 7) / 8, 1, 1);
        pr_decode_luma_lanes8<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_y, stride_y, bit_depth, num_slices);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        pr_decode_chroma444_both_lanes8<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_cb, d_cr, stride_c, bit_depth, num_slices);
        return cudaGetLastError();
    }

    if (strategy == CUPR_STRATEGY_LANE16) {
        dim3 grid((num_slices + 15) / 16, 1, 1);
        pr_decode_luma_lanes16<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_y, stride_y, bit_depth, num_slices);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        pr_decode_chroma422_both_lanes16<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_cb, d_cr, stride_c, bit_depth, num_slices);
        return cudaGetLastError();
    }

    if (strategy == CUPR_STRATEGY_DUAL) {
        dim3 grid((num_slices + 31) / 32, 1, 1);
        pr_decode_luma<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_y, stride_y, bit_depth, num_slices);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        pr_decode_chroma422_both<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_cb, d_cr, stride_c, bit_depth, num_slices);
        return cudaGetLastError();
    }

    dim3 grid((num_slices + 7) / 8, 1, 1);
    pr_decode_luma_lanes8<<<grid, block, 0, stream>>>(
        d_compressed, slices, d_y, stride_y, bit_depth, num_slices);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    pr_decode_chroma422_both_lanes8<<<grid, block, 0, stream>>>(
        d_compressed, slices, d_cb, d_cr, stride_c, bit_depth, num_slices);
    return cudaGetLastError();
}

cudaError_t cupr_decode_422_to_p210_async(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    int dst_pitch_y,
    int dst_pitch_uv,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    int strategy,
    int chroma_format,
    cudaStream_t stream) {
    auto err = cupr_launch_decode_422(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, strategy, chroma_format, stream);
    if (err != cudaSuccess) {
        return err;
    }
    const int stride_y = width;
    const int stride_c = (chroma_format == 3) ? width : (width / 2);

    dim3 packBlock(16, 16, 1);
    dim3 packGrid((width + packBlock.x - 1) / packBlock.x, (height + packBlock.y - 1) / packBlock.y, 1);
    cupr_pack_p210_kernel<<<packGrid, packBlock, 0, stream>>>(
        d_y, d_cb, d_cr,
        reinterpret_cast<uint16_t *>(dst_y),
        reinterpret_cast<uint16_t *>(dst_uv),
        stride_y, stride_c,
        dst_pitch_y / 2, dst_pitch_uv / 2,
        width, height, bit_depth, chroma_format);
    return cudaGetLastError();
}

cudaError_t cupr_decode_422_to_nv12_async(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    int dst_pitch_y,
    int dst_pitch_uv,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    int strategy,
    int chroma_format,
    cudaStream_t stream) {
    auto err = cupr_launch_decode_422(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, strategy, chroma_format, stream);
    if (err != cudaSuccess) {
        return err;
    }
    const int stride_y = width;
    const int stride_c = (chroma_format == 3) ? width : (width / 2);
    dim3 packBlock(16, 16, 1);
    dim3 packGrid((width + packBlock.x - 1) / packBlock.x, (height + packBlock.y - 1) / packBlock.y, 1);
    cupr_pack_nv12_kernel<<<packGrid, packBlock, 0, stream>>>(
        d_y, d_cb, d_cr,
        dst_y, dst_uv,
        stride_y, stride_c,
        dst_pitch_y, dst_pitch_uv,
        width, height, bit_depth, chroma_format);
    return cudaGetLastError();
}

cudaError_t cupr_decode_422_to_p010_async(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    int dst_pitch_y,
    int dst_pitch_uv,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    int strategy,
    int chroma_format,
    cudaStream_t stream) {
    auto err = cupr_launch_decode_422(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, strategy, chroma_format, stream);
    if (err != cudaSuccess) {
        return err;
    }
    const int stride_y = width;
    const int stride_c = (chroma_format == 3) ? width : (width / 2);
    dim3 packBlock(16, 16, 1);
    dim3 packGrid((width + packBlock.x - 1) / packBlock.x, (height + packBlock.y - 1) / packBlock.y, 1);
    cupr_pack_p010_kernel<<<packGrid, packBlock, 0, stream>>>(
        d_y, d_cb, d_cr,
        reinterpret_cast<uint16_t *>(dst_y),
        reinterpret_cast<uint16_t *>(dst_uv),
        stride_y, stride_c,
        dst_pitch_y / 2, dst_pitch_uv / 2,
        width, height, bit_depth, chroma_format);
    return cudaGetLastError();
}

static cudaError_t cupr_benchmark_one_strategy(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    int strategy,
    int chroma_format,
    cudaStream_t stream,
    float *ms) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    auto err = cudaEventCreateWithFlags(&start, cudaEventDefault);
    if (err == cudaSuccess) {
        err = cudaEventCreateWithFlags(&stop, cudaEventDefault);
    }
    if (err == cudaSuccess) {
        err = cudaEventRecord(start, stream);
    }
    if (err == cudaSuccess) {
        err = cupr_launch_decode_422(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, strategy, chroma_format, stream);
    }
    if (err == cudaSuccess) {
        err = cudaEventRecord(stop, stream);
    }
    if (err == cudaSuccess) {
        err = cudaEventSynchronize(stop);
    }
    if (err == cudaSuccess) {
        err = cudaEventElapsedTime(ms, start, stop);
    }
    if (start) {
        cudaEventDestroy(start);
    }
    if (stop) {
        cudaEventDestroy(stop);
    }
    return err;
}

cudaError_t cupr_benchmark_422_decode_async(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    cudaStream_t stream,
    CuprDecodeBenchmarkResult *result) {
    auto err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_LANE8, 2, stream, &result->lane8);
    if (err == cudaSuccess) {
        err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_LANE16, 2, stream, &result->lane16);
    }
    if (err == cudaSuccess) {
        err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_DUAL, 2, stream, &result->dual);
    }
    if (err == cudaSuccess) {
        err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_WIDE, 2, stream, &result->wide);
    }
    return err;
}

cudaError_t cupr_benchmark_444_decode_async(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    cudaStream_t stream,
    CuprDecodeBenchmarkResult *result) {
    auto err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_LANE8, 3, stream, &result->lane8);
    if (err == cudaSuccess) {
        err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_LANE16, 3, stream, &result->lane16);
    }
    if (err == cudaSuccess) {
        err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_DUAL, 3, stream, &result->dual);
    }
    if (err == cudaSuccess) {
        err = cupr_benchmark_one_strategy(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, CUPR_STRATEGY_WIDE, 3, stream, &result->wide);
    }
    return err;
}

static cudaError_t cupr_launch_decode_alpha444(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_alpha,
    int width,
    int bit_depth,
    int num_slices,
    int strategy,
    cudaStream_t stream) {
    const auto *slices = reinterpret_cast<const SliceInfo *>(d_slices);
    const int stride_a = width;
    dim3 block(32, 1, 1);
    if (strategy == CUPR_STRATEGY_LANE16) {
        dim3 grid((num_slices + 15) / 16, 1, 1);
        pr_decode_alpha444_lanes16<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_alpha, stride_a, bit_depth, num_slices);
    } else {
        dim3 grid((num_slices + 7) / 8, 1, 1);
        pr_decode_alpha444_lanes8<<<grid, block, 0, stream>>>(
            d_compressed, slices, d_alpha, stride_a, bit_depth, num_slices);
    }
    return cudaGetLastError();
}

cudaError_t cupr_decode_444_to_nv12a_async(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    int16_t *d_alpha,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    uint8_t *dst_alpha,
    int dst_pitch_y,
    int dst_pitch_uv,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    int strategy,
    cudaStream_t stream) {
    auto err = cupr_launch_decode_422(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, strategy, 3, stream);
    if (err != cudaSuccess) return err;
    err = cupr_launch_decode_alpha444(d_compressed, d_slices, d_alpha, width, bit_depth, num_slices, strategy, stream);
    if (err != cudaSuccess) return err;
    const int stride_y = width;
    const int stride_c = width;
    dim3 packBlock(16, 16, 1);
    dim3 packGrid((width + packBlock.x - 1) / packBlock.x, (height + packBlock.y - 1) / packBlock.y, 1);
    cupr_pack_nv12a_kernel<<<packGrid, packBlock, 0, stream>>>(
        d_y, d_cb, d_cr, d_alpha,
        dst_y, dst_uv, dst_alpha,
        stride_y, stride_c,
        dst_pitch_y, dst_pitch_uv,
        width, height, bit_depth, 3);
    return cudaGetLastError();
}

cudaError_t cupr_decode_444_to_p010a_async(
    const uint8_t *d_compressed,
    const CuprSliceInfo *d_slices,
    int16_t *d_y,
    int16_t *d_cb,
    int16_t *d_cr,
    int16_t *d_alpha,
    uint8_t *dst_y,
    uint8_t *dst_uv,
    uint8_t *dst_alpha,
    int dst_pitch_y,
    int dst_pitch_uv,
    int width,
    int height,
    int bit_depth,
    int num_slices,
    int strategy,
    cudaStream_t stream) {
    auto err = cupr_launch_decode_422(d_compressed, d_slices, d_y, d_cb, d_cr, width, height, bit_depth, num_slices, strategy, 3, stream);
    if (err != cudaSuccess) return err;
    err = cupr_launch_decode_alpha444(d_compressed, d_slices, d_alpha, width, bit_depth, num_slices, strategy, stream);
    if (err != cudaSuccess) return err;
    const int stride_y = width;
    const int stride_c = width;
    dim3 packBlock(16, 16, 1);
    dim3 packGrid((width + packBlock.x - 1) / packBlock.x, (height + packBlock.y - 1) / packBlock.y, 1);
    cupr_pack_p010a_kernel<<<packGrid, packBlock, 0, stream>>>(
        d_y, d_cb, d_cr, d_alpha,
        reinterpret_cast<uint16_t *>(dst_y),
        reinterpret_cast<uint16_t *>(dst_uv),
        reinterpret_cast<uint16_t *>(dst_alpha),
        stride_y, stride_c,
        dst_pitch_y / 2, dst_pitch_uv / 2,
        width, height, bit_depth, 3);
    return cudaGetLastError();
}
