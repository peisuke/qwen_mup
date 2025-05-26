FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN mkdir -p /workspace/scripts

WORKDIR /workspace

RUN git clone https://github.com/microsoft/mutransformers.git && \
    cd mutransformers && \
    pip install -r requirements.txt && \
    pip install -e .

RUN git clone https://github.com/microsoft/mup.git && \
    cp -r mup/examples/Transformer/data scripts/ && \
    rm -rf mup

COPY scripts/*.py scripts/
COPY scripts/setting.yml scripts/
