# This file has been adapted from https://github.com/demartis/matlab_runtime_docker

FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="me@alexanderwilhelmi.de"

RUN apt update && \
	apt install -y libgl1 python3-pip wget unzip libxt6

# ToDo Do we really need all of them?
RUN cd /tmp && \
	wget https://ssd.mathworks.com/supportfiles/downloads/R2023b/Release/4/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2023b_Update_4_glnxa64.zip && \
	unzip MATLAB_Runtime_R2023b_Update_4_glnxa64.zip && \
	./install -agreeToLicense yes && \
    rm MATLAB_Runtime_R2023b_Update_4_glnxa64.zip

RUN mkdir -p /app && \
    mkdir -p /app/SemSeg && \ 
    mkdir -p /app/results && \
    mkdir -p /app/results/csvFiles

COPY SemSegpythonPackage /app/SemSeg

RUN cd /app/SemSeg && \
	python3 /app/SemSeg/setup.py install

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/MATLAB/MATLAB_Runtime/R2023b/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023b/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023b/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023b/extern/bin/glnxa64

ENV XAPPLRESDIR=/etc/X11/app-defaults

# ToDo check if this is really needed
ENV MCR_MASTER_PATH=/usr/local/MATLAB/MATLAB_Runtime/R2023b

WORKDIR /app

RUN pip3 install "arkitekt-next[all]==0.8.37" tifffile pyarrow fastparquet && \
    pip3 uninstall kraph

COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
COPY input.tif /app/input.tif
