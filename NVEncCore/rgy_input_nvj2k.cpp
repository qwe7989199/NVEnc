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

#include "rgy_input_nvj2k.h"

#if ENABLE_AVSW_READER && ENCODER_NVENC

#include <array>
#include <cstring>
#include "rgy_input_nvj2k_kernels.h"

using nvjpeg2kStatus_t = int;
static const nvjpeg2kStatus_t NVJPEG2K_STATUS_SUCCESS = 0;

struct nvjpeg2kImageInfo_t {
    uint32_t image_width;
    uint32_t image_height;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t num_tiles_x;
    uint32_t num_tiles_y;
    uint32_t num_components;
};

struct nvjpeg2kImageComponentInfo_t {
    uint32_t component_width;
    uint32_t component_height;
    uint8_t precision;
    uint8_t sgn;
};

struct nvjpeg2kImage_t {
    void **pixel_data;
    size_t *pitch_in_bytes;
    int pixel_type;
    uint32_t num_components;
};

static const int NVJPEG2K_UINT16 = 1;

static bool nvj2k_source_is_xyz(const AVCodecParameters *codecpar) {
    const auto pixfmt = (AVPixelFormat)codecpar->format;
    const auto pixdesc = av_pix_fmt_desc_get(pixfmt);
    return pixdesc && (pixdesc->flags & AV_PIX_FMT_FLAG_XYZ);
}

static bool nvj2k_source_is_rgb(const AVCodecParameters *codecpar) {
    if (nvj2k_source_is_xyz(codecpar)) {
        return false;
    }
    const auto pixfmt = (AVPixelFormat)codecpar->format;
    const auto pixdesc = av_pix_fmt_desc_get(pixfmt);
    return codecpar->color_space == AVCOL_SPC_RGB
        || (pixdesc && (pixdesc->flags & AV_PIX_FMT_FLAG_RGB));
}

static int nvj2k_source_depth(const AVCodecParameters *codecpar) {
    const auto pixfmt = (AVPixelFormat)codecpar->format;
    const auto pixdesc = av_pix_fmt_desc_get(pixfmt);
    return (pixdesc) ? pixdesc->comp[0].depth : 16;
}

static CspMatrix nvj2k_rgb_to_yuv_matrix(const VideoVUIInfo& vui, int height) {
    if (vui.transfer == RGY_TRANSFER_ST2084 || vui.transfer == RGY_TRANSFER_ARIB_B67
        || vui.colorprim == RGY_PRIM_BT2020 || vui.colorprim == RGY_PRIM_ST431_2 || vui.colorprim == RGY_PRIM_ST432_1) {
        return RGY_MATRIX_BT2020_NCL;
    }
    return (height >= 720) ? RGY_MATRIX_BT709 : RGY_MATRIX_ST170_M;
}

struct RGYInputNvJ2k::ComponentInfo {
    uint32_t width;
    uint32_t height;
    uint8_t precision;
    uint8_t sgn;
};

struct RGYInputNvJ2k::DevicePlane {
    uint16_t *ptr = nullptr;
    size_t pitch = 0;
    size_t widthBytes = 0;
    uint32_t height = 0;
};

struct RGYInputNvJ2k::NvJ2kFuncs {
    HMODULE module = nullptr;
    nvjpeg2kStatus_t (*CreateSimple)(void **handle) = nullptr;
    nvjpeg2kStatus_t (*Destroy)(void *handle) = nullptr;
    nvjpeg2kStatus_t (*DecodeStateCreate)(void *handle, void **decode_state) = nullptr;
    nvjpeg2kStatus_t (*DecodeStateDestroy)(void *decode_state) = nullptr;
    nvjpeg2kStatus_t (*StreamCreate)(void **stream_handle) = nullptr;
    nvjpeg2kStatus_t (*StreamDestroy)(void *stream_handle) = nullptr;
    nvjpeg2kStatus_t (*StreamParse)(void *handle, const uint8_t *data, size_t length, int save_metadata, int save_stream, void *stream_handle) = nullptr;
    nvjpeg2kStatus_t (*StreamGetImageInfo)(void *stream_handle, nvjpeg2kImageInfo_t *image_info) = nullptr;
    nvjpeg2kStatus_t (*StreamGetImageComponentInfo)(void *stream_handle, nvjpeg2kImageComponentInfo_t *component_info, uint32_t component_id) = nullptr;
    nvjpeg2kStatus_t (*Decode)(void *handle, void *decode_state, void *jpeg2k_stream, nvjpeg2kImage_t *decode_output, cudaStream_t stream) = nullptr;

    ~NvJ2kFuncs() {
        if (module) {
            RGY_FREE_LIBRARY(module);
            module = nullptr;
        }
    }
};

template<typename Func>
static bool nvj2k_load_proc(HMODULE module, Func& func, const char *name) {
    func = reinterpret_cast<Func>(RGY_GET_PROC_ADDRESS(module, name));
    return func != nullptr;
}

static std::unique_ptr<RGYInputNvJ2k::NvJ2kFuncs> nvj2k_load() {
#if defined(_WIN32) || defined(_WIN64)
    static const TCHAR *moduleNames[] = {
        _T("nvjpeg2k_0.dll"),
        _T("nvjpeg2k64_13.dll"),
        _T("nvjpeg2k64_12.dll"),
        _T("nvjpeg2k.dll")
    };
#else
    static const TCHAR *moduleNames[] = {
        _T("libnvjpeg2k.so.13"),
        _T("libnvjpeg2k.so.12"),
        _T("libnvjpeg2k.so.0"),
        _T("libnvjpeg2k.so")
    };
#endif
    HMODULE module = nullptr;
    for (const auto name : moduleNames) {
        module = RGY_LOAD_LIBRARY(name);
        if (module != nullptr) {
            break;
        }
    }
    if (module == nullptr) {
        return nullptr;
    }

    auto funcs = std::make_unique<RGYInputNvJ2k::NvJ2kFuncs>();
    funcs->module = module;
    bool ok = true;
    ok &= nvj2k_load_proc(module, funcs->CreateSimple, "nvjpeg2kCreateSimple");
    ok &= nvj2k_load_proc(module, funcs->Destroy, "nvjpeg2kDestroy");
    ok &= nvj2k_load_proc(module, funcs->DecodeStateCreate, "nvjpeg2kDecodeStateCreate");
    ok &= nvj2k_load_proc(module, funcs->DecodeStateDestroy, "nvjpeg2kDecodeStateDestroy");
    ok &= nvj2k_load_proc(module, funcs->StreamCreate, "nvjpeg2kStreamCreate");
    ok &= nvj2k_load_proc(module, funcs->StreamDestroy, "nvjpeg2kStreamDestroy");
    ok &= nvj2k_load_proc(module, funcs->StreamParse, "nvjpeg2kStreamParse");
    ok &= nvj2k_load_proc(module, funcs->StreamGetImageInfo, "nvjpeg2kStreamGetImageInfo");
    ok &= nvj2k_load_proc(module, funcs->StreamGetImageComponentInfo, "nvjpeg2kStreamGetImageComponentInfo");
    ok &= nvj2k_load_proc(module, funcs->Decode, "nvjpeg2kDecode");
    if (!ok) {
        return nullptr;
    }
    return funcs;
}

RGYInputNvJ2kPrm::RGYInputNvJ2kPrm(const RGYInputAvcodecPrm& base) : RGYInputAvcodecPrm(base) {
    readVideo = true;
}

RGYInputNvJ2k::RGYInputNvJ2k() :
    RGYInputAvcodec(),
    m_nvj(),
    m_handle(nullptr),
    m_decodeState(nullptr),
    m_jpStream(nullptr),
    m_outputCsp(RGY_CSP_P010),
    m_planes(new DevicePlane[3]),
    m_sourceXyz(false) {
    m_readerName = _T("nvj2k");
}

RGYInputNvJ2k::~RGYInputNvJ2k() {
    closeNvjpeg2k();
}

bool RGYInputNvJ2k::outputCspSupported(RGY_CSP csp) const {
    switch (csp) {
    case RGY_CSP_NV12:
    case RGY_CSP_P010:
    case RGY_CSP_NV16:
    case RGY_CSP_P210:
    case RGY_CSP_YUV444:
    case RGY_CSP_YUV444_10:
    case RGY_CSP_YUV444_16:
    case RGY_CSP_RGB:
    case RGY_CSP_RGB_16:
    case RGY_CSP_GBR:
    case RGY_CSP_GBR_16:
        return true;
    default:
        return false;
    }
}

RGY_ERR RGYInputNvJ2k::initNvjpeg2k() {
    if (m_nvj) {
        return RGY_ERR_NONE;
    }
    m_nvj = nvj2k_load();
    if (!m_nvj) {
        AddMessage(RGY_LOG_ERROR, _T("failed to load nvjpeg2k library.\n"));
        return RGY_ERR_UNSUPPORTED;
    }
    auto sts = m_nvj->CreateSimple(&m_handle);
    if (sts != NVJPEG2K_STATUS_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("nvjpeg2kCreateSimple failed: status=%d.\n"), sts);
        closeNvjpeg2k();
        return RGY_ERR_UNKNOWN;
    }
    sts = m_nvj->DecodeStateCreate(m_handle, &m_decodeState);
    if (sts != NVJPEG2K_STATUS_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("nvjpeg2kDecodeStateCreate failed: status=%d.\n"), sts);
        closeNvjpeg2k();
        return RGY_ERR_UNKNOWN;
    }
    sts = m_nvj->StreamCreate(&m_jpStream);
    if (sts != NVJPEG2K_STATUS_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("nvjpeg2kStreamCreate failed: status=%d.\n"), sts);
        closeNvjpeg2k();
        return RGY_ERR_UNKNOWN;
    }
    return RGY_ERR_NONE;
}

void RGYInputNvJ2k::closeNvjpeg2k() {
    if (m_planes) {
        for (int i = 0; i < 3; i++) {
            if (m_planes[i].ptr) {
                cudaFree(m_planes[i].ptr);
                m_planes[i] = DevicePlane();
            }
        }
    }
    if (m_nvj) {
        if (m_jpStream) {
            m_nvj->StreamDestroy(m_jpStream);
            m_jpStream = nullptr;
        }
        if (m_decodeState) {
            m_nvj->DecodeStateDestroy(m_decodeState);
            m_decodeState = nullptr;
        }
        if (m_handle) {
            m_nvj->Destroy(m_handle);
            m_handle = nullptr;
        }
        m_nvj.reset();
    }
}

RGY_ERR RGYInputNvJ2k::ensureDevicePlane(DevicePlane& plane, const ComponentInfo& comp) {
    const size_t widthBytes = (size_t)comp.width * sizeof(uint16_t);
    if (plane.ptr && plane.widthBytes >= widthBytes && plane.height >= comp.height) {
        return RGY_ERR_NONE;
    }
    if (plane.ptr) {
        cudaFree(plane.ptr);
        plane = DevicePlane();
    }
    void *devicePtr = nullptr;
    auto err = cudaMallocPitch(&devicePtr, &plane.pitch, widthBytes, comp.height);
    if (err != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("cudaMallocPitch for nvj2k component failed: %s.\n"), char_to_tstring(cudaGetErrorString(err)).c_str());
        return err_to_rgy(err);
    }
    plane.ptr = static_cast<uint16_t *>(devicePtr);
    plane.widthBytes = widthBytes;
    plane.height = comp.height;
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputNvJ2k::Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm) {
    auto nvj2kPrm = dynamic_cast<const RGYInputNvJ2kPrm *>(prm);
    if (nvj2kPrm == nullptr) {
        return RGY_ERR_INVALID_PARAM;
    }

    const auto requestedCsp = inputInfo->csp;

    VideoInfo avInitInfo = *inputInfo;
    avInitInfo.type = RGY_INPUT_FMT_AVANY;
    RGYInputAvcodecPrm avPrm(static_cast<const RGYInputAvcodecPrm&>(*nvj2kPrm));
    avPrm.avswDecoder.clear();
    avPrm.disableVideoDecode = true;
    auto err = RGYInputAvcodec::Init(strFileName, &avInitInfo, &avPrm);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    m_readerName = _T("nvj2k");
    if (!m_Demux.video.stream || m_Demux.video.stream->codecpar->codec_id != AV_CODEC_ID_JPEG2000) {
        AddMessage(RGY_LOG_ERROR, _T("nvj2k input requires a JPEG 2000 video stream.\n"));
        return RGY_ERR_INVALID_CODEC;
    }
    const auto sourceXyz = nvj2k_source_is_xyz(m_Demux.video.stream->codecpar);
    const auto sourceRgb = sourceXyz ? true : nvj2k_source_is_rgb(m_Demux.video.stream->codecpar);
    m_sourceXyz = sourceXyz;
    m_outputCsp = sourceRgb
        ? ((nvj2k_source_depth(m_Demux.video.stream->codecpar) > 8) ? RGY_CSP_RGB_16 : RGY_CSP_RGB)
        : (outputCspSupported(requestedCsp) ? requestedCsp : RGY_CSP_P010);
    if ((err = initNvjpeg2k()) != RGY_ERR_NONE) {
        return err;
    }

    CloseVideoDecoder();
    m_Demux.video.HWDecodeDeviceId.clear();
    m_inputVideoInfo = avInitInfo;
    m_inputVideoInfo.type = RGY_INPUT_FMT_NVJ2K;
    m_inputVideoInfo.codec = RGY_CODEC_UNKNOWN;
    m_inputVideoInfo.csp = m_outputCsp;
    m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_outputCsp];
    if (sourceXyz) {
        // Kernel converts XYZ(DCI) -> linear BT.709 -> BT.709 gamma-encoded RGB.
        // Downstream sees ordinary BT.709 RGB input.
        m_inputVideoInfo.vui.colorprim = RGY_PRIM_BT709;
        m_inputVideoInfo.vui.transfer = RGY_TRANSFER_BT709;
        m_inputVideoInfo.vui.matrix = RGY_MATRIX_BT709;
        m_inputVideoInfo.vui.colorrange = RGY_COLORRANGE_FULL;
        m_inputVideoInfo.vui.descriptpresent = 1;
    } else if (sourceRgb) {
        m_inputVideoInfo.vui.matrix = nvj2k_rgb_to_yuv_matrix(m_inputVideoInfo.vui, m_inputVideoInfo.srcHeight);
        m_inputVideoInfo.vui.colorrange = RGY_COLORRANGE_LIMITED;
    }
    m_readerName = _T("nvj2k");
    m_inputInfo = strsprintf(_T("nvj2k: nvJPEG2000 CUDA, %dx%d, %d/%d fps, %s%s"),
        m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcHeight, m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD,
        RGY_CSP_NAMES[m_outputCsp], sourceXyz ? _T(" (XYZ DCI->BT.709)") : _T(""));
    *inputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputNvJ2k::LoadNextFrameInternal(RGYFrame *surface) {
    if (surface == nullptr) {
        return RGYInputAvcodec::LoadNextFrameInternal(surface);
    }
    AddMessage(RGY_LOG_ERROR, _T("nvj2k input cannot be loaded through a host surface.\n"));
    return RGY_ERR_INVALID_CALL;
}

RGY_ERR RGYInputNvJ2k::decodePacketToSurface(const AVPacket *pkt, CUFrameBuf *surface, cudaStream_t stream) {
    auto sts = initNvjpeg2k();
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    auto nvsts = m_nvj->StreamParse(m_handle, pkt->data, pkt->size, 0, 0, m_jpStream);
    if (nvsts != NVJPEG2K_STATUS_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("nvjpeg2kStreamParse failed: status=%d.\n"), nvsts);
        return RGY_ERR_INVALID_DATA_TYPE;
    }

    nvjpeg2kImageInfo_t info = {};
    nvsts = m_nvj->StreamGetImageInfo(m_jpStream, &info);
    if (nvsts != NVJPEG2K_STATUS_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("nvjpeg2kStreamGetImageInfo failed: status=%d.\n"), nvsts);
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    if (info.num_components != 3) {
        AddMessage(RGY_LOG_ERROR, _T("nvj2k currently supports 3-component JPEG 2000 only (components=%u).\n"), info.num_components);
        return RGY_ERR_UNSUPPORTED;
    }
    if ((int)info.image_width != surface->width() || (int)info.image_height != surface->height() || surface->csp() != m_outputCsp) {
        AddMessage(RGY_LOG_ERROR, _T("nvj2k decoded frame does not match destination surface: %ux%u -> %dx%d %s.\n"),
            info.image_width, info.image_height, surface->width(), surface->height(), RGY_CSP_NAMES[surface->csp()]);
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }

    std::array<ComponentInfo, 3> comps;
    for (uint32_t i = 0; i < info.num_components; i++) {
        nvjpeg2kImageComponentInfo_t c = {};
        nvsts = m_nvj->StreamGetImageComponentInfo(m_jpStream, &c, i);
        if (nvsts != NVJPEG2K_STATUS_SUCCESS) {
            AddMessage(RGY_LOG_ERROR, _T("nvjpeg2kStreamGetImageComponentInfo[%u] failed: status=%d.\n"), i, nvsts);
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        if (c.sgn != 0 || c.precision == 0 || c.precision > 16) {
            AddMessage(RGY_LOG_ERROR, _T("nvj2k unsupported component[%u] precision/sign: precision=%u, signed=%u.\n"), i, c.precision, c.sgn);
            return RGY_ERR_UNSUPPORTED;
        }
        comps[i] = ComponentInfo{ c.component_width, c.component_height, c.precision, c.sgn };
        if (comps[i].width == 0 || comps[i].height == 0) {
            AddMessage(RGY_LOG_ERROR, _T("nvj2k component[%u] has invalid dimensions %ux%u.\n"), i, comps[i].width, comps[i].height);
            return RGY_ERR_INVALID_VIDEO_PARAM;
        }
        if ((sts = ensureDevicePlane(m_planes[i], comps[i])) != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (comps[0].width != info.image_width || comps[0].height != info.image_height) {
        AddMessage(RGY_LOG_ERROR, _T("nvj2k luma/component0 dimensions must match image dimensions.\n"));
        return RGY_ERR_UNSUPPORTED;
    }

    void *pixelData[3] = { m_planes[0].ptr, m_planes[1].ptr, m_planes[2].ptr };
    size_t pitch[3] = { m_planes[0].pitch, m_planes[1].pitch, m_planes[2].pitch };
    nvjpeg2kImage_t image = {};
    image.pixel_data = pixelData;
    image.pitch_in_bytes = pitch;
    image.pixel_type = NVJPEG2K_UINT16;
    image.num_components = info.num_components;
    nvsts = m_nvj->Decode(m_handle, m_decodeState, m_jpStream, &image, stream);
    if (nvsts != NVJPEG2K_STATUS_SUCCESS) {
        AddMessage(RGY_LOG_ERROR, _T("nvjpeg2kDecode failed: status=%d.\n"), nvsts);
        return RGY_ERR_UNKNOWN;
    }

    NvJ2kPlane src[3] = {};
    for (int i = 0; i < 3; i++) {
        src[i] = NvJ2kPlane{ m_planes[i].ptr, m_planes[i].pitch, (int)comps[i].width, (int)comps[i].height, (int)comps[i].precision };
    }

    uint8_t *dst0 = nullptr;
    uint8_t *dst1 = nullptr;
    uint8_t *dst2 = nullptr;
    int pitch0 = 0;
    int pitch1 = 0;
    int pitch2 = 0;
    if (m_outputCsp == RGY_CSP_RGB || m_outputCsp == RGY_CSP_RGB_16 || m_outputCsp == RGY_CSP_GBR || m_outputCsp == RGY_CSP_GBR_16) {
        dst0 = surface->ptrPlane(RGY_PLANE_R);
        dst1 = surface->ptrPlane(RGY_PLANE_G);
        dst2 = surface->ptrPlane(RGY_PLANE_B);
        pitch0 = surface->pitch(RGY_PLANE_R);
        pitch1 = surface->pitch(RGY_PLANE_G);
        pitch2 = surface->pitch(RGY_PLANE_B);
    } else {
        dst0 = surface->ptrY();
        dst1 = surface->ptrUV();
        dst2 = surface->ptrV();
        pitch0 = surface->pitch(RGY_PLANE_Y);
        pitch1 = surface->pitch(RGY_PLANE_C);
        pitch2 = surface->pitch(RGY_PLANE_V);
    }
    auto cuerr = nvj2k_convert_to_surface_async(src, dst0, dst1, dst2, pitch0, pitch1, pitch2,
        (int)info.image_width, (int)info.image_height, m_outputCsp, m_sourceXyz, stream);
    if (cuerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("CUDA nvj2k surface conversion failed: %s.\n"), char_to_tstring(cudaGetErrorString(cuerr)).c_str());
        return err_to_rgy(cuerr);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputNvJ2k::LoadNextFrameDevice(CUFrameBuf *surface, cudaStream_t stream) {
    if (!m_Demux.thread.thInput.joinable() && m_Demux.qVideoPkt.get_keep_length() > 0) {
        auto [ret, pkt] = getSample();
        if (ret == 0) {
            m_Demux.qVideoPkt.push(pkt.release());
        } else if (ret != AVERROR_EOF) {
            return RGY_ERR_UNKNOWN;
        }
    }

    bool gotPacket = false;
    AVPacket *pkt = nullptr;
    for (int i = 0; false == (gotPacket = m_Demux.qVideoPkt.front_copy_and_pop_no_lock(&pkt, (m_Demux.thread.queueInfo) ? &m_Demux.thread.queueInfo->usage_vid_in : nullptr)) && m_Demux.qVideoPkt.size() > 0; i++) {
        m_Demux.qVideoPkt.wait_for_push();
    }
    if (!gotPacket) {
        return (m_Demux.format.inputError != RGY_ERR_NONE) ? m_Demux.format.inputError : RGY_ERR_MORE_DATA;
    }

    auto sts = decodePacketToSurface(pkt, surface, stream);
    if (sts == RGY_ERR_NONE) {
        const auto pts = (0 == (m_Demux.frames.getStreamPtsStatus() & (~RGY_PTS_NORMAL))) ? pkt->pts : AV_NOPTS_VALUE;
        const auto findPos = m_Demux.frames.findpts(pts, &m_Demux.video.findPosLastIdx);
        auto flags = RGY_FRAME_FLAG_NONE;
        if (findPos.poc != FRAMEPOS_POC_INVALID && (findPos.pic_struct & RGY_PICSTRUCT_INTERLACED) == 0 && findPos.repeat_pict > 1) {
            flags |= RGY_FRAME_FLAG_RFF;
        }
        const auto picstruct = (findPos.poc != FRAMEPOS_POC_INVALID && findPos.pic_struct != RGY_PICSTRUCT_UNKNOWN) ? static_cast<RGY_PICSTRUCT>(findPos.pic_struct) : RGY_PICSTRUCT_FRAME;
        surface->setTimestamp(pts);
        surface->setDuration(pkt->duration);
        surface->setPicstruct((m_inputVideoInfo.picstruct == RGY_PICSTRUCT_AUTO) ? picstruct : m_inputVideoInfo.picstruct);
        surface->setFlags(flags);
        surface->clearDataList();
        {
            auto hdr10plus = std::shared_ptr<RGYFrameData>(getHDR10plusMetaData(pkt));
            if (hdr10plus) {
                surface->dataList().push_back(hdr10plus);
            }
        }
        {
            auto dovirpu = std::shared_ptr<RGYFrameData>(getDoviRpuMetaData(pkt));
            if (dovirpu) {
                surface->dataList().push_back(dovirpu);
            }
        }
        m_Demux.video.nSampleGetCount++;
        m_encSatusInfo->m_sData.frameIn++;
    }
    m_poolPkt->returnFree(&pkt);

    if (m_Demux.format.inputError != RGY_ERR_NONE) {
        return m_Demux.format.inputError;
    }
    if (sts != RGY_ERR_NONE) {
        return sts;
    }
    double progressPercent = 0.0;
    if (m_Demux.format.formatCtx->duration) {
        progressPercent = m_Demux.frames.duration() * (m_Demux.video.stream->time_base.num / (double)m_Demux.video.stream->time_base.den);
    }
    return m_encSatusInfo->UpdateDisplayByCurrentDuration(progressPercent);
}

#endif //ENABLE_AVSW_READER && ENCODER_NVENC
