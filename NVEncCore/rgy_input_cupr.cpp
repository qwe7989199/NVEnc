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

#include "rgy_input_cupr.h"

#if ENABLE_AVSW_READER && ENCODER_NVENC

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include "rgy_input_cupr_kernels.h"

static uint16_t cupr_be16(const uint8_t *p) {
    return (uint16_t)((p[0] << 8) | p[1]);
}

static size_t cupr_align4(size_t value) {
    return (value + 3) & ~size_t(3);
}

static void cupr_write_be16(std::vector<uint8_t>& dst, size_t off, size_t value) {
    dst[off + 0] = (uint8_t)((value >> 8) & 0xff);
    dst[off + 1] = (uint8_t)(value & 0xff);
}

static bool cupr_fits_be16(size_t value) {
    return value <= std::numeric_limits<uint16_t>::max();
}

struct RGYInputCupr::ProResFrameInfo {
    int width = 0;
    int height = 0;
    int bitDepth = 10;
    int chromaFormat = 2; // 2 = 4:2:2, 3 = 4:4:4
    bool hasAlpha = false;
    int alphaInfo = 0; // 1 = 8-bit alpha, 2 = 16-bit alpha
    uint8_t lumaQmat[64] = {};
    uint8_t chromaQmat[64] = {};
    std::vector<uint8_t> compressed;
    std::vector<CuprSliceInfo> slices;
};

static RGY_ERR cupr_parse_prores_packet(RGYInputCupr::ProResFrameInfo& frame, const uint8_t *packet, size_t packetSize, tstring& err) {
    if (packetSize < 28) {
        err = _T("packet too short");
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    const size_t hdrOffset = packetSize >= 8 && memcmp(packet + 4, "icpf", 4) == 0 ? 8 : 0;
    const uint8_t *hdr = packet + hdrOffset;
    const size_t hdrSize = cupr_be16(hdr);
    if (hdrSize < 20 || hdrOffset + hdrSize > packetSize) {
        err = _T("invalid ProRes frame header");
        return RGY_ERR_INVALID_DATA_TYPE;
    }

    frame = RGYInputCupr::ProResFrameInfo();
    frame.width = cupr_be16(hdr + 8);
    frame.height = cupr_be16(hdr + 10);
    if (frame.width <= 0 || frame.height <= 0 || (frame.width & 1)) {
        err = _T("invalid ProRes frame dimensions");
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    const int chroma = (hdr[12] >> 6) & 3;
    if (chroma != 2 && chroma != 3) {
        err = _T("cupr supports ProRes 4:2:2 or 4:4:4 input only");
        return RGY_ERR_UNSUPPORTED;
    }
    frame.chromaFormat = chroma;
    frame.alphaInfo = hdrSize > 17 ? (hdr[17] & 0x0f) : 0;
    if (frame.alphaInfo > 2) {
        err = _T("invalid ProRes alpha mode");
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    frame.hasAlpha = (chroma == 3) && frame.alphaInfo != 0;
    const int bitDepthIndex = (hdr[13] >> 6) & 3;
    frame.bitDepth = bitDepthIndex == 0 ? 10 : bitDepthIndex == 1 ? 12 : 0;
    if (chroma == 2 && frame.bitDepth != 10) {
        err = _T("cupr 4:2:2 path currently supports 10-bit ProRes only");
        return RGY_ERR_UNSUPPORTED;
    }
    if (chroma == 3 && frame.bitDepth != 10 && frame.bitDepth != 12) {
        err = _T("cupr 4:4:4 path supports 10-bit or 12-bit ProRes only");
        return RGY_ERR_UNSUPPORTED;
    }

    const uint8_t qmatFlags = hdrSize > 19 ? hdr[19] : 0;
    size_t pos = 20;
    if (qmatFlags & 0x02) {
        if (pos + 64 > hdrSize) {
            err = _T("luma quant matrix overflows header");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        memcpy(frame.lumaQmat, hdr + pos, 64);
        pos += 64;
    } else {
        std::fill(std::begin(frame.lumaQmat), std::end(frame.lumaQmat), 4);
    }
    if (qmatFlags & 0x01) {
        if (pos + 64 > hdrSize) {
            err = _T("chroma quant matrix overflows header");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        memcpy(frame.chromaQmat, hdr + pos, 64);
    } else {
        std::fill(std::begin(frame.chromaQmat), std::end(frame.chromaQmat), 4);
    }

    const size_t picStart = hdrOffset + hdrSize;
    if (picStart + 8 >= packetSize) {
        err = _T("missing ProRes picture header");
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    const uint8_t *pic = packet + picStart;
    const size_t picHeaderSize = pic[0] >> 3;
    if (picHeaderSize < 8) {
        err = _T("picture header too short");
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    const int numSlices = cupr_be16(pic + 5);
    if (numSlices <= 0) {
        err = _T("ProRes frame contains no slices");
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    const int log2SliceMbWidth = pic[7] >> 4;
    if (log2SliceMbWidth > 3) {
        err = _T("invalid log2_slice_mb_width in ProRes picture header");
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    const int mbWidth = (frame.width + 15) / 16;
    const size_t indexStart = picStart + picHeaderSize;
    if (indexStart + (size_t)numSlices * 2 > packetSize) {
        err = _T("slice index overflows packet");
        return RGY_ERR_INVALID_DATA_TYPE;
    }
    const size_t dataStart = indexStart + (size_t)numSlices * 2;

    size_t dataOffset = 0;
    int sliceMbCount = 1 << log2SliceMbWidth;
    int mbX = 0;
    int mbY = 0;
    frame.slices.reserve(numSlices);
    for (int i = 0; i < numSlices; i++) {
        const size_t srcSize = cupr_be16(packet + indexStart + i * 2);
        if (dataStart + dataOffset + srcSize > packetSize) {
            err = _T("slice data region too short");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        if (srcSize < 6) {
            err = _T("slice header too short");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        const uint8_t *src = packet + dataStart + dataOffset;
        dataOffset += srcSize;
        while (mbWidth - mbX < sliceMbCount) {
            sliceMbCount >>= 1;
        }
        if (sliceMbCount <= 0 || mbX >= mbWidth) {
            err = _T("invalid slice macroblock layout");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        const size_t srcHeaderSize = src[0] >> 3;
        if (srcHeaderSize < 6 || srcHeaderSize > srcSize) {
            err = _T("invalid slice header size");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        const size_t outHeaderSize = frame.hasAlpha ? 10 : 8;
        const size_t ySize = cupr_be16(src + 2);
        const size_t uSize = cupr_be16(src + 4);
        if (srcHeaderSize + ySize + uSize > srcSize) {
            err = _T("plane sizes overflow slice");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        const size_t vSize = srcHeaderSize > 7 ? cupr_be16(src + 6) : srcSize - srcHeaderSize - ySize - uSize;
        size_t aSize = 0;
        if (frame.hasAlpha) {
            if (srcHeaderSize > 9) {
                aSize = cupr_be16(src + 8);
            } else {
                const size_t used = srcHeaderSize + ySize + uSize + vSize;
                aSize = (used < srcSize) ? (srcSize - used) : 0;
            }
        }
        if (srcHeaderSize + ySize + uSize + vSize + aSize > srcSize) {
            err = _T("plane data overflows slice");
            return RGY_ERR_INVALID_DATA_TYPE;
        }

        const size_t yPadded = cupr_align4(ySize);
        const size_t uPadded = cupr_align4(uSize);
        const size_t vPadded = cupr_align4(vSize);
        const size_t aPadded = frame.hasAlpha ? cupr_align4(aSize) : 0;
        if (!cupr_fits_be16(yPadded) || !cupr_fits_be16(uPadded) || !cupr_fits_be16(vPadded) || !cupr_fits_be16(aPadded)) {
            err = _T("padded ProRes slice plane size exceeds 16-bit header field");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        const size_t sliceStart = cupr_align4(frame.compressed.size());
        if (sliceStart > std::numeric_limits<uint32_t>::max()) {
            err = _T("packed ProRes slice offset exceeds 32-bit device header field");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        frame.compressed.resize(sliceStart, 0);
        frame.compressed.resize(sliceStart + outHeaderSize, 0);
        frame.compressed[sliceStart] = (uint8_t)((outHeaderSize << 3) | (src[0] & 0x07));
        frame.compressed[sliceStart + 1] = src[1];
        cupr_write_be16(frame.compressed, sliceStart + 2, yPadded);
        cupr_write_be16(frame.compressed, sliceStart + 4, uPadded);
        cupr_write_be16(frame.compressed, sliceStart + 6, vPadded);
        if (frame.hasAlpha) {
            cupr_write_be16(frame.compressed, sliceStart + 8, aPadded);
        }

        const size_t yStart = srcHeaderSize;
        frame.compressed.insert(frame.compressed.end(), src + yStart, src + yStart + ySize);
        frame.compressed.resize(sliceStart + outHeaderSize + yPadded, 0);
        const size_t uStart = yStart + ySize;
        frame.compressed.insert(frame.compressed.end(), src + uStart, src + uStart + uSize);
        frame.compressed.resize(sliceStart + outHeaderSize + yPadded + uPadded, 0);
        const size_t vStart = uStart + uSize;
        frame.compressed.insert(frame.compressed.end(), src + vStart, src + vStart + vSize);
        frame.compressed.resize(sliceStart + outHeaderSize + yPadded + uPadded + vPadded, 0);
        if (frame.hasAlpha) {
            const size_t aStart = vStart + vSize;
            frame.compressed.insert(frame.compressed.end(), src + aStart, src + aStart + aSize);
            frame.compressed.resize(sliceStart + outHeaderSize + yPadded + uPadded + vPadded + aPadded, 0);
        }

        const size_t packedSliceSize = frame.compressed.size() - sliceStart;
        if (packedSliceSize > std::numeric_limits<uint32_t>::max() || mbX > std::numeric_limits<uint16_t>::max() || mbY > std::numeric_limits<uint16_t>::max() || sliceMbCount > std::numeric_limits<uint16_t>::max()) {
            err = _T("packed ProRes slice metadata exceeds device header field");
            return RGY_ERR_INVALID_DATA_TYPE;
        }
        CuprSliceInfo si = {};
        si.offset = (uint32_t)sliceStart;
        si.size = (uint32_t)packedSliceSize;
        si.mb_x = (uint16_t)mbX;
        si.mb_y = (uint16_t)mbY;
        si.mb_count = (uint16_t)sliceMbCount;
        frame.slices.push_back(si);

        mbX += sliceMbCount;
        if (mbX == mbWidth) {
            sliceMbCount = 1 << log2SliceMbWidth;
            mbX = 0;
            mbY++;
        }
    }
    return RGY_ERR_NONE;
}

RGYInputCuprPrm::RGYInputCuprPrm(const RGYInputAvcodecPrm& base) : RGYInputAvcodecPrm(base) {
    readVideo = true;
    strategy = RGY_CUPR_DECODE_STRATEGY_AUTO;
}

RGYInputCupr::RGYInputCupr() :
    RGYInputAvcodec(),
    m_outputCsp(RGY_CSP_P210),
    m_strategy(RGY_CUPR_DECODE_STRATEGY_AUTO),
    m_selectedStrategy(RGY_CUPR_DECODE_STRATEGY_LANE8),
    m_autoBenchmarkCount(0),
    m_autoBenchmarkMs(),
    m_srcInfoUpdated(false),
    m_dCompressed(nullptr),
    m_dSlices(nullptr),
    m_dY(nullptr),
    m_dCb(nullptr),
    m_dCr(nullptr),
    m_dAlpha(nullptr),
    m_dCompressedCapacity(0),
    m_dSlicesCapacity(0),
    m_dYCapacity(0),
    m_dCbCapacity(0),
    m_dCrCapacity(0),
    m_dAlphaCapacity(0) {
    m_readerName = _T("cupr");
}

RGYInputCupr::~RGYInputCupr() {
    releaseDeviceBuffers();
}

void RGYInputCupr::releaseDeviceBuffers() {
    if (m_dCompressed) cudaFree(m_dCompressed);
    if (m_dSlices) cudaFree(m_dSlices);
    if (m_dY) cudaFree(m_dY);
    if (m_dCb) cudaFree(m_dCb);
    if (m_dCr) cudaFree(m_dCr);
    if (m_dAlpha) cudaFree(m_dAlpha);
    m_dCompressed = nullptr;
    m_dSlices = nullptr;
    m_dY = nullptr;
    m_dCb = nullptr;
    m_dCr = nullptr;
    m_dAlpha = nullptr;
    m_dCompressedCapacity = 0;
    m_dSlicesCapacity = 0;
    m_dYCapacity = 0;
    m_dCbCapacity = 0;
    m_dCrCapacity = 0;
    m_dAlphaCapacity = 0;
}

template<typename T>
RGY_ERR RGYInputCupr::ensureDeviceBuffer(T **ptr, size_t *capacity, size_t required) {
    if (*capacity >= required) {
        return RGY_ERR_NONE;
    }
    if (*ptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
        *capacity = 0;
    }
    void *devicePtr = nullptr;
    auto err = cudaMalloc(&devicePtr, required);
    if (err != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("cudaMalloc failed: %s.\n"), char_to_tstring(cudaGetErrorString(err)).c_str());
        return err_to_rgy(err);
    }
    *ptr = static_cast<T *>(devicePtr);
    *capacity = required;
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputCupr::Init(const TCHAR *strFileName, VideoInfo *inputInfo, const RGYInputPrm *prm) {
    auto cuprPrm = dynamic_cast<const RGYInputCuprPrm *>(prm);
    if (cuprPrm == nullptr) {
        return RGY_ERR_INVALID_PARAM;
    }

    const auto requestedCsp = inputInfo->csp;
    switch (requestedCsp) {
    case RGY_CSP_NV12:
    case RGY_CSP_P010:
    case RGY_CSP_P210:
    case RGY_CSP_NV12A:
    case RGY_CSP_P010A:
        m_outputCsp = requestedCsp;
        break;
    default:
        m_outputCsp = RGY_CSP_P210;
        break;
    }
    m_strategy = cuprPrm->strategy;
    m_selectedStrategy = (m_strategy == RGY_CUPR_DECODE_STRATEGY_AUTO) ? RGY_CUPR_DECODE_STRATEGY_LANE8 : m_strategy;

    VideoInfo avInitInfo = *inputInfo;
    avInitInfo.type = RGY_INPUT_FMT_AVANY;
    RGYInputAvcodecPrm avPrm(static_cast<const RGYInputAvcodecPrm&>(*cuprPrm));
    avPrm.avswDecoder.clear();
    avPrm.disableVideoDecode = true;
    auto err = RGYInputAvcodec::Init(strFileName, &avInitInfo, &avPrm);
    if (err != RGY_ERR_NONE) {
        return err;
    }
    m_readerName = _T("cupr");
    if (!m_Demux.video.stream || m_Demux.video.stream->codecpar->codec_id != AV_CODEC_ID_PRORES) {
        AddMessage(RGY_LOG_ERROR, _T("cupr input requires a ProRes video stream.\n"));
        return RGY_ERR_INVALID_CODEC;
    }

    CloseVideoDecoder();
    m_Demux.video.HWDecodeDeviceId.clear();

    // When RGY_CSP_NA is requested (alpha output mode), auto-select NV12A if source has alpha.
    if (requestedCsp == RGY_CSP_NA) {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)m_Demux.video.stream->codecpar->format);
        if (desc && (desc->flags & AV_PIX_FMT_FLAG_ALPHA)) {
            m_outputCsp = RGY_CSP_NV12A; // alpha encoding only supports 8-bit YUV420
        }
    }

    m_inputVideoInfo = avInitInfo;
    m_inputVideoInfo.type = RGY_INPUT_FMT_CUPR;
    m_inputVideoInfo.codec = RGY_CODEC_UNKNOWN;
    m_inputVideoInfo.csp = m_outputCsp;
    m_inputVideoInfo.bitdepth = RGY_CSP_BIT_DEPTH[m_outputCsp];
    m_readerName = _T("cupr");

    // Infer source chroma/bitdepth from container pixel format for the info line.
    {
        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)m_Demux.video.stream->codecpar->format);
        int srcBitDepth = (desc && desc->comp[0].depth > 0) ? desc->comp[0].depth : 10;
        const TCHAR *chromaStr = _T("4:2:2");
        bool srcHasAlpha = false;
        if (desc) {
            if (desc->log2_chroma_w == 0) chromaStr = _T("4:4:4");
            srcHasAlpha = (desc->flags & AV_PIX_FMT_FLAG_ALPHA) != 0;
        }
        const TCHAR *alphaStr = srcHasAlpha ? _T("+A") : _T("");
        m_inputInfo = strsprintf(_T("ProRes CUDA, %s%s %dbit, %dx%d, %d/%d fps, fast decoded to %s, strategy %s"),
            chromaStr, alphaStr, srcBitDepth,
            m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcHeight, m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD,
            RGY_CSP_NAMES[m_outputCsp], get_chr_from_value(list_cupr_decode_strategy, (int)m_strategy));
        m_srcInfoUpdated = true;
    }
    *inputInfo = m_inputVideoInfo;
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputCupr::LoadNextFrameInternal(RGYFrame *surface) {
    if (surface == nullptr) {
        return RGYInputAvcodec::LoadNextFrameInternal(surface);
    }
    AddMessage(RGY_LOG_ERROR, _T("cupr input cannot be loaded through a host surface.\n"));
    return RGY_ERR_INVALID_CALL;
}

RGY_ERR RGYInputCupr::updateAutoStrategy(const ProResFrameInfo& frame, cudaStream_t stream) {
    if (m_strategy != RGY_CUPR_DECODE_STRATEGY_AUTO || m_autoBenchmarkCount >= 3) {
        return RGY_ERR_NONE;
    }

    CuprDecodeBenchmarkResult result = {};
    cudaError_t cuerr;
    if (frame.chromaFormat == 3) {
        cuerr = cupr_benchmark_444_decode_async(
            m_dCompressed, reinterpret_cast<const CuprSliceInfo *>(m_dSlices), m_dY, m_dCb, m_dCr,
            frame.width, frame.height, frame.bitDepth, (int)frame.slices.size(), stream, &result);
    } else {
        cuerr = cupr_benchmark_422_decode_async(
            m_dCompressed, reinterpret_cast<const CuprSliceInfo *>(m_dSlices), m_dY, m_dCb, m_dCr,
            frame.width, frame.height, frame.bitDepth, (int)frame.slices.size(), stream, &result);
    }
    if (cuerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("cupr auto strategy benchmark failed: %s.\n"), char_to_tstring(cudaGetErrorString(cuerr)).c_str());
        return err_to_rgy(cuerr);
    }

    m_autoBenchmarkMs[0] += result.lane8;
    m_autoBenchmarkMs[1] += result.lane16;
    m_autoBenchmarkMs[2] += result.dual;
    m_autoBenchmarkMs[3] += result.wide;
    m_autoBenchmarkCount++;

    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (m_autoBenchmarkMs[i] < m_autoBenchmarkMs[best]) {
            best = i;
        }
    }
    static const RGY_CUPR_DECODE_STRATEGY strategies[4] = {
        RGY_CUPR_DECODE_STRATEGY_LANE8,
        RGY_CUPR_DECODE_STRATEGY_LANE16,
        RGY_CUPR_DECODE_STRATEGY_DUAL,
        RGY_CUPR_DECODE_STRATEGY_WIDE
    };
    m_selectedStrategy = strategies[best];

    AddMessage(m_autoBenchmarkCount >= 3 ? RGY_LOG_INFO : RGY_LOG_DEBUG,
        _T("cupr auto strategy benchmark %d/3: lane8 %.3f ms, lane16 %.3f ms, dual %.3f ms, wide %.3f ms -> %s.\n"),
        m_autoBenchmarkCount,
        m_autoBenchmarkMs[0] / m_autoBenchmarkCount,
        m_autoBenchmarkMs[1] / m_autoBenchmarkCount,
        m_autoBenchmarkMs[2] / m_autoBenchmarkCount,
        m_autoBenchmarkMs[3] / m_autoBenchmarkCount,
        get_chr_from_value(list_cupr_decode_strategy, (int)m_selectedStrategy));
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputCupr::decodePacketToSurface(const AVPacket *pkt, CUFrameBuf *surface, cudaStream_t stream) {
    ProResFrameInfo frame;
    tstring parseErr;
    auto sts = cupr_parse_prores_packet(frame, pkt->data, pkt->size, parseErr);
    if (sts != RGY_ERR_NONE) {
        AddMessage(RGY_LOG_ERROR, _T("failed to parse ProRes packet: %s.\n"), parseErr.c_str());
        return sts;
    }
    if (!m_srcInfoUpdated) {
        m_srcInfoUpdated = true;
        const TCHAR *chromaStr = (frame.chromaFormat == 3) ? _T("4:4:4") : _T("4:2:2");
        const TCHAR *alphaStr = frame.hasAlpha ? _T("+A") : _T("");
        const TCHAR *outputCspName = RGY_CSP_NAMES[m_outputCsp];
        m_inputInfo = strsprintf(_T("ProRes CUDA, %s%s %dbit, %dx%d, %d/%d fps, fast decoded to %s, strategy %s"),
            chromaStr, alphaStr, frame.bitDepth,
            m_inputVideoInfo.srcWidth, m_inputVideoInfo.srcHeight, m_inputVideoInfo.fpsN, m_inputVideoInfo.fpsD,
            outputCspName, get_chr_from_value(list_cupr_decode_strategy, (int)m_strategy));
    }
    if (frame.width != surface->width() || frame.height != surface->height() || surface->csp() != m_outputCsp) {
        AddMessage(RGY_LOG_ERROR, _T("cupr decoded frame does not match destination surface: %dx%d -> %dx%d %s.\n"),
            frame.width, frame.height, surface->width(), surface->height(), RGY_CSP_NAMES[surface->csp()]);
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }
    if (rgy_csp_has_alpha(m_outputCsp) && !frame.hasAlpha) {
        AddMessage(RGY_LOG_ERROR, _T("cupr alpha output requires ProRes 4444 input with alpha channel.\n"));
        return RGY_ERR_INVALID_VIDEO_PARAM;
    }

    const int paddedHeight = ALIGN(frame.height, 16);
    const size_t ySamples = (size_t)frame.width * paddedHeight;
    const int chromaWidth = (frame.chromaFormat == 3) ? frame.width : (frame.width / 2);
    const size_t cSamples = (size_t)chromaWidth * paddedHeight;
    if ((sts = ensureDeviceBuffer(&m_dCompressed, &m_dCompressedCapacity, frame.compressed.size())) != RGY_ERR_NONE) return sts;
    if ((sts = ensureDeviceBuffer(&m_dSlices, &m_dSlicesCapacity, frame.slices.size() * sizeof(CuprSliceInfo))) != RGY_ERR_NONE) return sts;
    if ((sts = ensureDeviceBuffer(&m_dY, &m_dYCapacity, ySamples * sizeof(int16_t))) != RGY_ERR_NONE) return sts;
    if ((sts = ensureDeviceBuffer(&m_dCb, &m_dCbCapacity, cSamples * sizeof(int16_t))) != RGY_ERR_NONE) return sts;
    if ((sts = ensureDeviceBuffer(&m_dCr, &m_dCrCapacity, cSamples * sizeof(int16_t))) != RGY_ERR_NONE) return sts;
    if (m_outputCsp == RGY_CSP_NV12A || m_outputCsp == RGY_CSP_P010A) {
        if ((sts = ensureDeviceBuffer(&m_dAlpha, &m_dAlphaCapacity, ySamples * sizeof(int16_t))) != RGY_ERR_NONE) return sts;
    }

    auto cuerr = cudaMemcpyAsync(m_dCompressed, frame.compressed.data(), frame.compressed.size(), cudaMemcpyHostToDevice, stream);
    if (cuerr == cudaSuccess) {
        cuerr = cudaMemcpyAsync(m_dSlices, frame.slices.data(), frame.slices.size() * sizeof(CuprSliceInfo), cudaMemcpyHostToDevice, stream);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMemsetAsync(m_dY, 0, ySamples * sizeof(int16_t), stream);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMemsetAsync(m_dCb, 0, cSamples * sizeof(int16_t), stream);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cudaMemsetAsync(m_dCr, 0, cSamples * sizeof(int16_t), stream);
    }
    if (cuerr == cudaSuccess && m_dAlpha && (m_outputCsp == RGY_CSP_NV12A || m_outputCsp == RGY_CSP_P010A)) {
        cuerr = cudaMemsetAsync(m_dAlpha, 0, ySamples * sizeof(int16_t), stream);
    }
    if (cuerr == cudaSuccess) {
        cuerr = cupr_upload_qmat_async(frame.lumaQmat, frame.chromaQmat, stream);
    }
    if (cuerr == cudaSuccess) {
        if ((sts = updateAutoStrategy(frame, stream)) != RGY_ERR_NONE) {
            return sts;
        }
    }
    if (cuerr == cudaSuccess) {
        if (m_outputCsp == RGY_CSP_NV12) {
            cuerr = cupr_decode_422_to_nv12_async(
                m_dCompressed, reinterpret_cast<const CuprSliceInfo *>(m_dSlices), m_dY, m_dCb, m_dCr,
                surface->ptrY(), surface->ptrUV(),
                surface->pitch(RGY_PLANE_Y), surface->pitch(RGY_PLANE_C),
                frame.width, frame.height, frame.bitDepth, (int)frame.slices.size(), (int)m_selectedStrategy, frame.chromaFormat, stream);
        } else if (m_outputCsp == RGY_CSP_P010) {
            cuerr = cupr_decode_422_to_p010_async(
                m_dCompressed, reinterpret_cast<const CuprSliceInfo *>(m_dSlices), m_dY, m_dCb, m_dCr,
                surface->ptrY(), surface->ptrUV(),
                surface->pitch(RGY_PLANE_Y), surface->pitch(RGY_PLANE_C),
                frame.width, frame.height, frame.bitDepth, (int)frame.slices.size(), (int)m_selectedStrategy, frame.chromaFormat, stream);
        } else if (m_outputCsp == RGY_CSP_NV12A) {
            cuerr = cupr_decode_444_to_nv12a_async(
                m_dCompressed, reinterpret_cast<const CuprSliceInfo *>(m_dSlices), m_dY, m_dCb, m_dCr, m_dAlpha,
                surface->ptrY(), surface->ptrUV(), surface->ptrPlane(RGY_PLANE_A),
                surface->pitch(RGY_PLANE_Y), surface->pitch(RGY_PLANE_C),
                frame.width, frame.height, frame.bitDepth, frame.alphaInfo, (int)frame.slices.size(), (int)m_selectedStrategy, stream);
        } else if (m_outputCsp == RGY_CSP_P010A) {
            cuerr = cupr_decode_444_to_p010a_async(
                m_dCompressed, reinterpret_cast<const CuprSliceInfo *>(m_dSlices), m_dY, m_dCb, m_dCr, m_dAlpha,
                surface->ptrY(), surface->ptrUV(), surface->ptrPlane(RGY_PLANE_A),
                surface->pitch(RGY_PLANE_Y), surface->pitch(RGY_PLANE_C),
                frame.width, frame.height, frame.bitDepth, frame.alphaInfo, (int)frame.slices.size(), (int)m_selectedStrategy, stream);
        } else {
            cuerr = cupr_decode_422_to_p210_async(
                m_dCompressed, reinterpret_cast<const CuprSliceInfo *>(m_dSlices), m_dY, m_dCb, m_dCr,
                surface->ptrY(), surface->ptrUV(),
                surface->pitch(RGY_PLANE_Y), surface->pitch(RGY_PLANE_C),
                frame.width, frame.height, frame.bitDepth, (int)frame.slices.size(), (int)m_selectedStrategy, frame.chromaFormat, stream);
        }
    }
    if (cuerr != cudaSuccess) {
        AddMessage(RGY_LOG_ERROR, _T("CUDA ProRes decode failed: %s.\n"), char_to_tstring(cudaGetErrorString(cuerr)).c_str());
        return err_to_rgy(cuerr);
    }
    return RGY_ERR_NONE;
}

RGY_ERR RGYInputCupr::LoadNextFrameDevice(CUFrameBuf *surface, cudaStream_t stream) {
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
