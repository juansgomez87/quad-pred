import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import argparse
from keras.models import model_from_json, load_model

import pdb


class QuadPredictor():
    def __init__(self, case_str, input_file):
        """Constructor method
        """
        # initial configuration
        self.max_output = False
        self.print_approx = True
        self.plot_taggram = True
        
        self.ind_to_label = {0: 'Q1 (A+V+)', 1: 'Q2 (A+V-)', 2: 'Q3 (A-V-)', 3: 'Q4 (A-V+)'}
        self.mod_type = 'feat_ext'  # 'feat_ext' or 'unmix'
        self.model_name = 'model_over_8'
        self.sampling_rate = 16000
        path_model_load = os.path.join('./models', case_str)

        # load models
        j_f, w_f = self.model_selector(path_model_load)
        model = self.load_pretrained_model(j_f, w_f)
        
        # extract spectrogram
        spec_array = self.create_spectrogram(input_file)
        print('*************\nCalculating output for file:', input_file)
        print('Using model:', j_f, '\n*************')
        
        # predict!
        self.format_input = input_file.split('.')[-1]
        out_file = input_file.replace(self.format_input, 'npy')
        self.predict_and_save(model, spec_array, out_file)
       
      
    def model_selector(self, path):
        """ This method selects the weights and structure of the network
        """
        sel_txt = 'it_1'
        files = [os.path.join(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) 
                                                                     and f.find(sel_txt) > 0 
                                                                     and f.find(self.model_name) == 0 
                                                                     and f.find('spec') >= 0
                                                                     and f.find(self.mod_type) >= 0)]
        weights_filename = [_ for _ in files if _.endswith('.hdf5')][0]
        json_filename = [_ for _ in files if _.endswith('.json')][0]
        return json_filename, weights_filename


    def load_pretrained_model(self, json_file, weight_file):
        """ This method loads the pretrained models, loads the 
        weights and adds the new layers"""
        # load model
        j_f = open(json_file, 'r')
        loaded_model = j_f.read()
        j_f.close()
        model = model_from_json(loaded_model)
        # load weights
        model = load_model(weight_file)
        return model

    def create_spectrogram(self, in_f, sr=16000, win_length=1024, hop_length=512, num_mel=128):
        """This method creates a melspectrogram from an audio file using librosa
        audio processing library. Parameters are default from Han et al.
        
        :param filename: wav filename to process.
        :param sr: sampling rate in Hz (default: 16000).
        :param win_length: window length for STFT (default: 1024).
        :param hop_length: hop length for STFT (default: 512).
        :param num_mel: number of mel bands (default:128).
        :type filename: str
        :type sr: int
        :type win_length: int
        :type hop_length: int
        :type num_mel: int
        
        :returns: **ln_S** *(np.array)* - melspectrogram of the complete audio file with logarithmic compression with dimensionality [mel bands x time frames].
        """
        # minimum float32 representation epsilon in python
        eps = np.finfo(np.float32).eps
        assert os.path.exists(in_f), "filename %r does not exist" % in_f

        data, sr = librosa.load(in_f, sr=sr, mono=True)
        duration = int(np.floor(sr / hop_length))
        try:
            # normalize data
            data /= np.max(np.abs(data))
        except Warning:
            print(filename, ' is empty')

        # time-frequency representation Short Time Fourier Transform
        D = np.abs(librosa.stft(data, win_length=win_length, hop_length=hop_length, center=True))
        # mel frequency representation
        S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=num_mel)
        # apply natural logarithm
        ln_S = np.log(S + eps)
        spec_list = []
        for idx in range(0, ln_S.shape[1] - duration + 1, duration):
            # append chunk of spectrogram to dataset
            spec_list.append(ln_S[:, idx:(idx + duration)])
        spec_array = np.expand_dims(spec_list, axis=1)
        return spec_array

    def predict_and_save(self, model, spec_array, out_file):
        """ This method makes predictions and saves the output in several forms 
        depending on the initial config. 
        """
        y_pred = model.predict(spec_array)
        if self.max_output:
            y_pred_max = np.zeros(y_pred.shape)
            for i in range(y_pred_max.shape[0]):
                y_pred_max[i, np.argmax(y_pred, axis=1)[i]] = np.max(y_pred, axis=1)[i]
            y_pred = y_pred_max
        
        if self.print_approx:
            mean_pred = np.mean(y_pred, axis=0)
            print('*************\nMean predictions for file:', out_file)
            print('Quadrant 1 (positive arousal, positive valence):', mean_pred[0])
            print('Quadrant 2 (positive arousal, negative valence):', mean_pred[1])
            print('Quadrant 3 (negative arousal, negative valence):', mean_pred[2])
            print('Quadrant 4 (negative arousal, positive valence):', mean_pred[3])
            print('*************')
        
        if self.plot_taggram:
            plt.imshow(y_pred.T, aspect='auto', interpolation='nearest')
            plt.xlabel('Time [s]')
            plt.yticks(np.arange(len(self.ind_to_label)), self.ind_to_label.values())
            plt.ylabel('Quadrants')
            plt.colorbar()
            plt.title('Quadrant prediction')
            plt.tight_layout()
            png_file = out_file.replace('npy', 'png')
            plt.savefig(png_file)
        # save predictions
        np.save(out_file, y_pred)
        

if __name__ == "__main__":
    # Usage python3 quad_pred.py --speech e/m --music e/m --input input_filename
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--speech',
                        help='Select from pretrained models on speech in english (e) or mandarin (m)',
                        action='store',
                        required=True,
                        dest='speech')
    parser.add_argument('-m',
                        '--music',
                        help='Select music of data for transfer learning: english (e) or mandarin (m)',
                        action='store',
                        required=True,
                        dest='music')
    parser.add_argument('-i',
                        '--input',
                        help='Select filename to make predictions',
                        action='store',
                        required=True,
                        dest='input')
    args = parser.parse_args()

    if args.speech == 'e' and args.music == 'e':
        case_str = 'speech_eng_2_music_eng'
        print('******\n INTRALINGUISTIC CASE: eng 2 eng\n')
    elif args.speech == 'e' and args.music == 'm':
        case_str = 'speech_eng_2_music_man'
        print('******\n CROSSLINGUISTIC CASE: eng 2 man\n')
    elif args.speech == 'm' and args.music == 'm':
        case_str = 'speech_man_2_music_man'
        print('******\n INTRALINGUISTIC CASE: man 2 man\n')
    elif args.speech == 'm' and args.music == 'e':
        case_str = 'speech_man_2_music_eng'
        print('******\n CROSSLINGUISTIC CASE: man 2 eng\n')
    
    # instanciate predictor
    q_pred = QuadPredictor(case_str, args.input)
