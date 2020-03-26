import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import argparse
from keras.models import model_from_json, load_model

import pdb


class QuadPredictor():
    def __init__(self, case_str, input_file, out_dir):
        """Constructor method
        """
        # initial configuration
        self.max_output = False
        self.print_approx = True
        self.plot_taggram = True
        
        self.ind_to_label_quad = {0: 'Q1 (A+V+)', 1: 'Q2 (A+V-)', 2: 'Q3 (A-V-)', 3: 'Q4 (A-V+)'}
        self.ind_to_label_arou = {0: 'A-', 1: 'A+'}
        self.ind_to_label_vale = {0: 'V-', 1: 'V+'}
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

        out_file = os.path.join(out_dir, input_file.split('/')[-1].split('.')[0])

        self.predict_and_save(model, spec_array, out_file)
       
      
    def model_selector(self, path):
        """ This method selects the weights and structure of the network
        """
        sel_txt = 'it_3'
        files = [os.path.join(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) 
                                                                     and f.find(sel_txt) > 0)]
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
            y_pred_max_quad = np.zeros(y_pred[0].shape)
            y_pred_max_arou = np.zeros(y_pred[1].shape)
            y_pred_max_vale = np.zeros(y_pred[2].shape)
            for i in range(y_pred_max_quad.shape[0]):
                y_pred_max_quad[i, np.argmax(y_pred[0], axis=1)[i]] = np.max(y_pred[0], axis=1)[i]
                y_pred_max_arou[i, np.argmax(y_pred[1], axis=1)[i]] = np.max(y_pred[1], axis=1)[i]
                y_pred_max_vale[i, np.argmax(y_pred[2], axis=1)[i]] = np.max(y_pred[2], axis=1)[i]
            y_pred_quad = y_pred_max_quad
            y_pred_arou = y_pred_max_arou
            y_pred_vale = y_pred_max_vale
        else:
            y_pred_quad = y_pred[0]
            y_pred_arou = y_pred[1]
            y_pred_vale = y_pred[2]            
   
        if self.print_approx:
            mean_pred_quad = np.mean(y_pred_quad, axis=0)
            mean_pred_arou = np.mean(y_pred_arou, axis=0)
            mean_pred_vale = np.mean(y_pred_vale, axis=0)
            print('*************\nMean predictions for file:', out_file)
            print('Quadrant 1 (positive arousal, positive valence):', mean_pred_quad[0])
            print('Quadrant 2 (positive arousal, negative valence):', mean_pred_quad[1])
            print('Quadrant 3 (negative arousal, negative valence):', mean_pred_quad[2])
            print('Quadrant 4 (negative arousal, positive valence):', mean_pred_quad[3])
            print('*************')
            print('Negative arousal:', mean_pred_arou[0])
            print('Positive arousal:', mean_pred_arou[1])
            print('*************')
            print('Negative valence:', mean_pred_vale[0])
            print('Positive valence:', mean_pred_vale[1])
            print('*************')

        
        if self.plot_taggram:
            fig, ax = plt.subplots(3, 1, figsize=(5, 10))
            ax[0].imshow(y_pred_quad.T, aspect='auto', interpolation='nearest')
            ax[0].set_yticks(np.arange(len(self.ind_to_label_quad)))
            ax[0].set_yticklabels(self.ind_to_label_quad.values())
            ax[0].set_ylabel('Quadrants')
            ax[1].imshow(y_pred_arou.T, aspect='auto', interpolation='nearest')
            ax[1].set_yticks(np.arange(len(self.ind_to_label_arou)))
            ax[1].set_yticklabels(self.ind_to_label_arou.values())
            ax[1].set_ylabel('Arousal')
            ax[2].imshow(y_pred_vale.T, aspect='auto', interpolation='nearest')
            ax[2].set_xlabel('Time [s]')
            ax[2].set_yticks(np.arange(len(self.ind_to_label_vale)))
            ax[2].set_yticklabels(self.ind_to_label_vale.values())
            ax[2].set_ylabel('Valence')
 
            plt.tight_layout()
            png_file = out_file.replace('npy', 'png')
            plt.savefig(png_file)
        # save predictions
        np.save(out_file, y_pred_quad)


if __name__ == "__main__":
    # Usage python3 quad_pred.py --speech e/m --music e/m --input input_filename --output output_dir
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
    parser.add_argument('-o',
                        '--output',
                        help='Select output directory to save results',
                        action='store',
                        required=True,
                        dest='output')
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
    q_pred = QuadPredictor(case_str, args.input, args.output)
