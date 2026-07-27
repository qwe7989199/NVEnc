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
    int chroma_format,
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
    int chroma_format,
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
    int chroma_format,
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
    CuprDecodeBenchmarkResult *result);

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
    int alpha_info,
    int num_slices,
    int strategy,
    cudaStream_t stream);

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
    int alpha_info,
    int num_slices,
    int strategy,
    cudaStream_t stream);

#endif //__RGY_INPUT_CUPR_KERNELS_H__
