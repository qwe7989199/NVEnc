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
