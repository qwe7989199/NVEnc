#pragma once
#ifndef __RGY_INPUT_CUPR_KERNELS_H__
#define __RGY_INPUT_CUPR_KERNELS_H__

#include <stdint.h>
#include <cuda_runtime.h>

struct CuprSliceInfo {
    uint32_t offset;
    uint32_t size;
    uint16_t mb_x;
    uint16_t mb_y;
    uint16_t mb_count;
    uint8_t qindex;
    uint8_t reserved;
};

struct CuprDecodeBenchmarkResult {
    float lane8;
    float lane16;
    float dual;
    float wide;
};

cudaError_t cupr_upload_qmat_async(const uint8_t luma[64], const uint8_t chroma[64], cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    CuprDecodeBenchmarkResult *result);

#endif //__RGY_INPUT_CUPR_KERNELS_H__
