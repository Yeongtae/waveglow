# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import argparse
import json
import torch
import torch.utils.data
import sys
import numpy as np
import random
from scipy.io.wavfile import read
import librosa

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

# For debugging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
def plot_data(data, figsize=(16, 4), filename='test.png'):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')
    plt.savefig(filename)

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def audiopaths_and_melpaths(filename):
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip().split('|') for f in files] # [[audiopath, melpath], ... ]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, load_mel_from_disk=False):
        self.load_mel_from_disk = load_mel_from_disk
        self.hop_length = hop_length
        self.audio_files = audiopaths_and_melpaths(training_files) if self.load_mel_from_disk else files_to_list(training_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_mel_from_file(self, mel_path):
        melspec = np.load(mel_path)
        melspec = torch.autograd.Variable(melspec, requires_grad=False)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename[0]) if self.load_mel_from_disk \
                        else load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        if(self.load_mel_from_disk):
            # Take segment
            mel = np.load(filename[1])
            assert self.segment_length%self.hop_length == 0, 'self.segment_length must be n times of self.hop_length'
            max_mel_length = int(self.segment_length/self.hop_length)
            audio_ = audio.data.cpu().numpy()
            if(mel.shape[1] > len(audio_)/self.hop_length): #handling error
                diff = int(mel.shape[1] - len(audio_)/self.hop_length)
                mel=mel[:,:-diff]
            if(mel.shape[1] < len(audio_)/self.hop_length):
                print(filename, mel.shape, len(audio))
            if audio.size(0) >= self.segment_length:
                max_mel_start = int((audio.size(0)-self.segment_length)/self.hop_length) # audio.size(0)%self.hop_length is the remainder
                mel_start = random.randint(0, max_mel_start)
                audio_start = mel_start*self.hop_length
                audio = audio[audio_start : audio_start + self.segment_length]
                mel = mel[:,mel_start:mel_start + max_mel_length]
            else:
                len_pad = int((self.segment_length/ self.hop_length) - mel.shape[1])
                pad = np.ones((80, len_pad), dtype=np.float32) * -11.512925
                mel =  np.append(mel, pad, axis=1)
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

            mel = torch.from_numpy(mel).float()
            audio = audio / MAX_WAV_VALUE
            # if(mel.shape[1] != int(self.segment_length/self.hop_length)):
            #     print()
        else:
            # Take segment
            if audio.size(0) >= self.segment_length:
                max_audio_start = audio.size(0) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start + self.segment_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

            mel = self.get_mel(audio)  #
            audio = audio / MAX_WAV_VALUE
        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    with open("config.json") as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    # original mel_audio loader
    #mel2samp = Mel2Samp(**data_config)

    # GTA mel_audio loader
    data_config['training_files'] = 'audio_mel_train.txt'
    data_config['load_mel_from_disk'] = True
    data_config['segment_length'] = 16128
    mel2samp2 = Mel2Samp(**data_config)

    for i in range(5):
        data, sr = load_wav_to_torch('000000{}.wav'.format(i))
        mel_gta = np.load('000000{}.mel.npy'.format(i))
        mel_org = mel2samp2.get_mel(data)
        plot_data([mel_org, mel_gta], filename='mel_match{}.png'.format(i))

        mel1, audio1_norm = mel2samp2.__getitem__(i)
        mel1_org = mel2samp2.get_mel(audio1_norm*MAX_WAV_VALUE)
        mel1, mel1_org = mel1.data.cpu().numpy(), mel1_org.data.cpu().numpy()
        print(len(audio1_norm), mel1_org.shape, mel1_org.max(), mel1_org.min(), mel1.shape, mel1.max(), mel1.min())
        plot_data([mel1_org, mel1], filename='mel_index{}.png'.format(i))

    pass