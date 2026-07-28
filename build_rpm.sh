#!/bin/sh

set -e

PACKAGE_NAME=nvencc
PACKAGE_BIN=nvencc
PACKAGE_OS=
PACKAGE_MAINTAINER=rigaya
PACKAGE_DEPENDS="glibc"
PACKAGE_DESCRIPTION=
PACKAGE_ROOT=.rpmpkg
PACKAGE_VERSION=`git describe --tags | cut -f 1 --delim="-" | sed -e 's/^v//'`
PACKAGE_ARCH=`uname -m`
PACKAGE_LICENSE="MIT AND LicenseRef-NVIDIA-nvJPEG2000"

if [ -e /etc/os-release ]; then
    PACKAGE_OS_NAME=`cat /etc/os-release | grep ID | cut -f 2 --delim="=" | head -n 1 `
    PACKAGE_OS_VER=`cat /etc/os-release | grep VERSION_ID | cut -f 2 --delim="=" | head -n 1 `
    PACKAGE_OS="${PACKAGE_OS_NAME}${PACKAGE_OS_VER}"
fi

NVJPEG2K_LIB_DIR=${NVJPEG2K_ROOT:?NVJPEG2K_ROOT is not set}/lib/${NVJPEG2K_CUDA_MAJOR:?NVJPEG2K_CUDA_MAJOR is not set}
if [ ! -f "${NVJPEG2K_LIB_DIR}/libnvjpeg2k.so.0.9.0.43" ] || [ ! -L "${NVJPEG2K_LIB_DIR}/libnvjpeg2k.so" ] || [ ! -f "${NVJPEG2K_ROOT}/LICENSE" ]; then
    echo "nvJPEG2000 runtime or license is missing under ${NVJPEG2K_ROOT}!"
    exit 1
fi

rm -rf ${PACKAGE_ROOT}
mkdir -p ${PACKAGE_ROOT}/usr/bin
cp ${PACKAGE_BIN} ${PACKAGE_ROOT}/usr/bin/
chmod +x ${PACKAGE_ROOT}/usr/bin/${PACKAGE_BIN}

mkdir -p ${PACKAGE_ROOT}/usr/lib64
cp -a ${NVJPEG2K_LIB_DIR}/libnvjpeg2k.so* ${PACKAGE_ROOT}/usr/lib64/

mkdir -p ${PACKAGE_ROOT}/usr/share/licenses/${PACKAGE_NAME}
cp ${NVJPEG2K_ROOT}/LICENSE ${PACKAGE_ROOT}/usr/share/licenses/${PACKAGE_NAME}/nvjpeg2000-LICENSE

rm -rf rpmbuild
RPMBUILD_DIR=${PWD}/rpmbuild
PACKAGE_SOURCE_DIR=${RPMBUILD_DIR}/SOURCES
PACKAGE_SPEC_DIR=${RPMBUILD_DIR}/SPECS

mkdir -p ${PACKAGE_SOURCE_DIR}
mkdir -p ${PACKAGE_SPEC_DIR}

WORK_DIR=.tmpwork
rm -rf ${WORK_DIR}
mkdir ${WORK_DIR}

build_pkg/replace.py \
    --rpm \
    -i packaging/nvencc.spec \
    -o ${WORK_DIR}/${PACKAGE_NAME}.spec \
    --pkg-name ${PACKAGE_NAME} \
    --pkg-bin ${PACKAGE_BIN} \
    --pkg-version ${PACKAGE_VERSION} \
    --pkg-arch ${PACKAGE_ARCH} \
    --pkg-maintainer ${PACKAGE_MAINTAINER} \
    --pkg-depends ${PACKAGE_DEPENDS} \
    --pkg-desc ${PACKAGE_DESCRIPTION} \
    --pkg-license "${PACKAGE_LICENSE}"

cp -rp "${PACKAGE_ROOT}" "${WORK_DIR}/${PACKAGE_NAME}"
cd ${WORK_DIR}
tar czf tmp.tar.gz "${PACKAGE_NAME}/"
mv tmp.tar.gz ${PACKAGE_SOURCE_DIR}/
cd ..
cp -p ${WORK_DIR}/${PACKAGE_NAME}.spec "${PACKAGE_SPEC_DIR}/"
rm -rf ${WORK_DIR}

rpmbuild --define "_topdir ${RPMBUILD_DIR}" -ba "${PACKAGE_SPEC_DIR}/${PACKAGE_NAME}.spec"

cp ${RPMBUILD_DIR}/RPMS/${PACKAGE_ARCH}/${PACKAGE_NAME}*.rpm .
