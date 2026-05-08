// -----------------------------------------------------------------------------------------
// QSVEnc/NVEnc by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2026
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
#ifndef __RGY_INPUT_CUPR_H__
#define __RGY_INPUT_CUPR_H__

#include "rgy_input_avcodec.h"

#if ENABLE_AVSW_READER && ENCODER_NVENC

#include <cuda_runtime.h>
#include "rgy_cuda_util.h"

class RGYInputCuprPrm : public RGYInputAvcodecPrm {
public:
    RGYInputCuprPrm(RGYInputAvcodecPrm base);
    virtual ~RGYInputCuprPrm() {};
    RGY_CUPR_DECODE_STRATEGY strategy;
};

class RGYInputCupr : public RGYInputAvcodec {
public:
    struct ProResFrameInfo;

    RGYInputCupr();
    virtual ~RGYInputCupr();

    RGY_ERR LoadNextFrameDevice(CUFrameBuf *surface, cudaStream_t stream);

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm) override;
    virtual RGY_ERR LoadNextFrameInternal(RGYFrame *surface) override;

private:
    RGY_ERR decodePacketToSurface(const AVPacket *pkt, CUFrameBuf *surface, cudaStream_t stream);
    RGY_ERR updateAutoStrategy(const ProResFrameInfo& frame, cudaStream_t stream);
    RGY_ERR ensureDeviceBuffer(uint8_t **ptr, size_t *capacity, size_t required);
    void releaseDeviceBuffers();

    RGY_CSP m_outputCsp;
    RGY_CUPR_DECODE_STRATEGY m_strategy;
    RGY_CUPR_DECODE_STRATEGY m_selectedStrategy;
    int m_autoBenchmarkCount;
    float m_autoBenchmarkMs[4];
    uint8_t *m_dCompressed;
    uint8_t *m_dSlices;
    int16_t *m_dY;
    int16_t *m_dCb;
    int16_t *m_dCr;
    size_t m_dCompressedCapacity;
    size_t m_dSlicesCapacity;
    size_t m_dYCapacity;
    size_t m_dCbCapacity;
    size_t m_dCrCapacity;
};

#endif //ENABLE_AVSW_READER && ENCODER_NVENC

#endif //__RGY_INPUT_CUPR_H__
