FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git python3 python3-pip ffmpeg && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./

# create venv in /opt/venv so it’s obvious and outside your app tree
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# copy your code after installing deps
COPY *.py ./
COPY logging_config.json ./

# now you can just invoke python without activating
CMD ["python3", "main.py"]
