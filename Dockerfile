FROM python:3.6

# Common requirements
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    python3 \
    ffmpeg \
    libsndfile1 \
    python3-pip 

COPY ./requirements.txt /tmp/requirements.txt
# RUN pip3 install SoundFile==0.10.2 librosa==0.6.1 scipy==1.1.0 ffmpeg-python==0.1.17
RUN pip3 install --upgrade pip && pip3 install -r /tmp/requirements.txt
COPY . /tmp/
WORKDIR /tmp
ENTRYPOINT [ "python3", "/tmp/quad_pred.py" ]
