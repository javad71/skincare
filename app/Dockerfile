# pull official base image
FROM python:3.7

# set work directory
RUN mkdir /usr/src/app
RUN mkdir /usr/src/app/static
RUN mkdir /usr/src/app/media
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install opencv and dlib dependencies
RUN apt-get update -y && \
    apt-get install build-essential cmake pkg-config -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install torchvision==0.11.3

COPY . /usr/src/app
COPY ./process/core/acne/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /root/.cache/torch/hub/checkpoints/
COPY ./process/core/beauty/seresnext50_32x4d-0451-52029a7f.pth /root/.torch/models/
COPY ./process/core/age_race/age_model_weights.h5 /root/.deepface/weights/
COPY ./process/core/age_race/race_model_single_batch.h5 /root/.deepface/weights/

# copy project
COPY . .

 
