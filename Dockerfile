FROM python:3.6

# Common requirements
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    python3 \
    ffmpeg \
    libsndfile1-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN mkdir /code
COPY . /code
WORKDIR /code
ENTRYPOINT [ "python3", "/code/quad_pred.py" ]
