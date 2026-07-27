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
#ifndef __RGY_INPUT_NVJ2K_KERNELS_H__
#define __RGY_INPUT_NVJ2K_KERNELS_H__

#include <stdint.h>
#include <cuda_runtime.h>
#include "convert_csp.h"

struct NvJ2kPlane {
    const uint16_t *ptr;
    size_t pitch;
    int width;
    int height;
    int precision;
};

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
    cudaStream_t stream);

#endif //__RGY_INPUT_NVJ2K_KERNELS_H__
