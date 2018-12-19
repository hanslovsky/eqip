FROM tensorflow/tensorflow:1.3.0-gpu
LABEL maintainer hanslovskyp@janelia.hhmi.org

# basic ubuntu packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
	libmlpack-dev \
	python-numpy \
	python-setuptools \
	libboost-all-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*

ENV PIP=pip
ENV PYTHON=python

RUN ${PIP} install scikit-image

ENV PREFIX=$HOME

#
# malis
#
ENV MALIS_ROOT=${PREFIX}/src/malis
ENV MALIS_REPOSITORY=https://github.com/TuragaLab/malis.git
ENV MALIS_REVISION=beb4ee965acee89ab00a20a70205f51177003c69

RUN python --version
RUN ${PIP} install cython
WORKDIR ${MALIS_ROOT}
RUN git clone ${MALIS_REPOSITORY} . && \
    git checkout ${MALIS_REVISION}
RUN ${PYTHON} setup.py build_ext --inplace
ENV PYTHONPATH ${MALIS_ROOT}:$PYTHONPATH

#
# augment
#
ENV AUGMENT_ROOT=${PREFIX}/src/augment
ENV AUGMENT_REPOSITORY=https://github.com/funkey/augment.git
ENV AUGMENT_REVISION=4a42b01ccad7607b47a1096e904220729dbcb80a

WORKDIR ${AUGMENT_ROOT}
RUN git clone ${AUGMENT_REPOSITORY} . && \
    git checkout ${AUGMENT_REVISION}
RUN ${PIP} install -r requirements.txt
ENV PYTHONPATH ${AUGMENT_ROOT}:$PYTHONPATH

#
# gunpowder
#
ENV GUNPOWDER_ROOT=${PREFIX}/src/gunpowder
ENV GUNPOWDER_REPOSITORY=https://github.com/funkey/gunpowder.git
ENV GUNPOWDER_REVISION=f3edcca3f292d4fe8ff3b691793847124681ce19

WORKDIR ${GUNPOWDER_ROOT}
RUN git clone ${GUNPOWDER_REPOSITORY} . && \
    git checkout ${GUNPOWDER_REVISION}
RUN ${PIP} install -r requirements.txt
RUN ${PYTHON} setup.py build_ext --inplace
ENV PYTHONPATH ${GUNPOWDER_ROOT}:$PYTHONPATH

#
# gunpowder-nodes
#
ENV GPN_ROOT=${PREFIX}/src/gpn
ENV GPN_REPOSITORY=https://github.com/hanslovsky/gunpowder-nodes.git
ENV GPN_REVISION=121744e736573408162db2839f30fc3064f630b2

WORKDIR ${GPN_ROOT}
RUN git clone ${GPN_REPOSITORY} . && \
    git checkout ${GPN_REVISION}
ENV PYTHONPATH ${GPN_ROOT}:$PYTHONPATH

#
# CNNectome
#
ENV CNNECTOME_ROOT=${PREFIX}/src/CNNectome
ENV CNNECTOME_REPOSITORY=https://github.com/hanslovsky/CNNectome.git
ENV CNNECTOME_REVISION=01c2700

WORKDIR ${CNNECTOME_ROOT}
RUN git clone ${CNNECTOME_REPOSITORY} . && \
    git checkout ${CNNECTOME_REVISION}
ENV PYTHONPATH ${CNNECTOME_ROOT}:$PYTHONPATH

ARG EQIP_REVISION=master

#
# eqip
#
ENV EQIP_ROOT=${PREFIX}/src/eqip
ENV EQIP_REPOSITORY=https://github.com/hanslovsky/eqip.git

WORKDIR ${EQIP_ROOT}
RUN git clone ${EQIP_REPOSITORY} . && \
    git checkout ${EQIP_REVISION} && \
    ${PIP} install .
