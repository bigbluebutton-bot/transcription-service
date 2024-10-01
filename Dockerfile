FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS build-env

RUN apt update && apt install -y git python3 python3-pip ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt install -y --no-install-recommends \
    libcudnn8 \
    libcudnn8-dev

WORKDIR /app
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY *.py ./

COPY logging_config.json ./

CMD ["python3", "main.py"]
