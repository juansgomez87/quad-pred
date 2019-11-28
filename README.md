# Arousal-Valence Quadrant Prediction

## Content
This script loads an audio file and makes predictions of the perceived emotion using the Russell circumplex model of emotion as a classifier (four classes). Quadrant 1 relates to positive arousal - positive valence (e.g., happy), Quadrant 2 relates to positive arousal - negative valence (e.g., angry), Quadrant 3 relates to negative arousal - negative valence (e.g., sad), and Quadrant 4 relates to negative arousal - positive valence (e.g., relaxed). The models have been previously trained with speech in English (Librispeech) and Mandarin (AISHELL) and transfer learning has been performed to fine-tune on music in English (4Q-Emotion) and Mandarin (CH-818).


## Prerequisites
Install Docker, for Ubuntu [go here].
[https://docs.docker.com/install/linux/docker-ce/ubuntu/]

## Run

Clone this repository and build the container with all corresponding installations. This might take a while since it will install Tensorflow from scratch:

```
git clone https://github.com/juansgomez87/quad-pred.git
cd quad-pred
docker build -t quadpred .
docker run -it --rm -v /abspath/quad-pred/audio/bitter_1.mp3:/audio.mp3 -v /abspath/quad-pred/:/outdir quadpred -s e -m e -i /audio.mp3 -o /outdir/result.npy
```

Otherwise, you can also install dependencies using:
```
pip3 install -r requirements.txt
python3 quad_pred.py -s e -m e -i audio/anger_1.mp3 -o audio/results.npy
```

You can also change the flags in the contructor method to output a taggram or print the mean probability of the classifier over the whole clip.

You can use the `-help` flag to see the complete list of information. 
```
docker run --rm quadpred -h
usage: quad_pred.py [-h] -s SPEECH -m MUSIC -i INPUT

optional arguments:
  -h, --help            show this help message and exit
  -s SPEECH, --speech SPEECH
                        Select from pretrained models on speech in english (e)
                        or mandarin (m)
  -m MUSIC, --music MUSIC
                        Select music of data for transfer learning: english
                        (e) or mandarin (m)
  -i INPUT, --input INPUT
                        Select filename to make predictions
```

### Tag-gram
![alt text](https://github.com/juansgomez87/quad-pred/blob/master/audio/anger_1.png)

### Approximation
```
*************
Calculating output for file: audio/anger_1.mp3
Using model: ./models/speech_eng_2_music_eng/model_over_8.spec.it_1.feat_ext.json 
*************
*************
Mean predictions for file: audio/anger_1.npy
Quadrant 1 (positive arousal, positive valence): 0.20719571
Quadrant 2 (positive arousal, negative valence): 0.7511331
Quadrant 3 (negative arousal, negative valence): 0.021125803
Quadrant 4 (negative arousal, positive valence): 0.020545341
```
