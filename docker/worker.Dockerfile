FROM continuumio/miniconda3:24.1.2-0

ARG ENV_FILE

WORKDIR /app

COPY ${ENV_FILE} /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

COPY api /app/api
COPY environments /app/environments
COPY restore_photo.py /app/restore_photo.py

RUN mkdir -p /app/api/temp /app/api/static/results /app/outputs /app/experiments/pretrained_models

ENV PYTHONUNBUFFERED=1
ENV ALLOW_MODEL_DOWNLOADS=false

CMD ["bash"]
