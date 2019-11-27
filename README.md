# Arousal-Valence Quadrant Prediction

## Content
This script loads an audio file and makes predictions of the perceived emotion using the Russell circumplex model of emotion as a classifier (four classes). Quadrant 1 relates to positive arousal - positive valence (e.g., happy), Quadrant 2 relates to positive arousal - negative valence (e.g., angry), Quadrant 3 relates to negative arousal - negative valence (e.g., sad), and Quadrant 4 relates to negative arousal - positive valence (e.g., happy). The models have been previously trained with speech in English (Librispeech) and Mandarin (AISHELL) and transfer learning has been performed to fine-tune on music in English (4Q-Emotion) and Mandarin (CH-818).


# Prerequisites
```
pip install keras, matplotlib, numpy, librosa, h5py
```

## Run 

Just run the file with the corresponding model selector and the input file to process. 
```
python3 quad_pred.py -s e -m e -i audio/anger_1.mp3
```

You can also change the flags in the contructor method to output a taggram or print the mean probability of the classifier over the whole clip.

![alt text](https://github.com/juansgomez87/quad-pred/blob/master/audio/anger_1.png)
