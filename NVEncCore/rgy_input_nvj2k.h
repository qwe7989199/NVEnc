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
#ifndef __RGY_INPUT_NVJ2K_H__
#define __RGY_INPUT_NVJ2K_H__

#include "rgy_input_avcodec.h"

#if ENABLE_AVSW_READER && ENCODER_NVENC

#include <memory>
#include <cuda_runtime.h>
#include "rgy_cuda_util.h"

class RGYInputNvJ2kPrm : public RGYInputAvcodecPrm {
public:
    RGYInputNvJ2kPrm(RGYInputAvcodecPrm base);
    virtual ~RGYInputNvJ2kPrm() {};
};

class RGYInputNvJ2k : public RGYInputAvcodec {
public:
    struct NvJ2kFuncs;
    struct DevicePlane;
    struct ComponentInfo;

    RGYInputNvJ2k();
    virtual ~RGYInputNvJ2k();

    RGY_ERR LoadNextFrameDevice(CUFrameBuf *surface, cudaStream_t stream);

protected:
    virtual RGY_ERR Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm) override;
    virtual RGY_ERR LoadNextFrameInternal(RGYFrame *surface) override;

private:
    RGY_ERR initNvjpeg2k();
    void closeNvjpeg2k();
    RGY_ERR ensureDevicePlane(DevicePlane& plane, const ComponentInfo& comp);
    RGY_ERR decodePacketToSurface(const AVPacket *pkt, CUFrameBuf *surface, cudaStream_t stream);
    bool outputCspSupported(RGY_CSP csp) const;

    std::unique_ptr<NvJ2kFuncs> m_nvj;
    void *m_handle;
    void *m_decodeState;
    void *m_jpStream;
    RGY_CSP m_outputCsp;
    DevicePlane *m_planes;
};

#endif //ENABLE_AVSW_READER && ENCODER_NVENC

#endif //__RGY_INPUT_NVJ2K_H__
