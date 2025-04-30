FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS build-env

RUN apt update && \
    apt install -y --no-install-recommends \
      git python3 python3-pip python3.12-venv ffmpeg \
      libcudnn9-dev-cuda-12 && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY requirements.txt ./

# create venv in /opt/venv so it’s obvious and outside your app tree
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# copy your code after installing deps
COPY *.py ./
COPY logging_config.json ./

# add the venv’s bin dir to PATH so that 'python3' and 'pip' point to your venv
ENV PATH="/opt/venv/bin:$PATH"

# now you can just invoke python without activating
CMD ["python3", "main.py"]
