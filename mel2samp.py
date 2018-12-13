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
            assert self.segment_length%self.hop_length == 0, 'self.segment_length must be n times of self.hop_length'
            max_mel_length = int(self.segment_length/self.hop_length)

            if audio.size(0) >= self.segment_length:
                max_audio_start = int(audio.size(0)/self.hop_length) # audio.size(0)%self.hop_length is remainder
                audio_start = random.randint(0, max_audio_start)*self.hop_length
                mel_start = int(audio_start/self.hop_length)
                audio = audio[audio_start : audio_start + self.segment_length]
                mel = np.load(filename[1])[:,mel_start:mel_start + max_mel_length]
            else:
                mel = np.load(filename[1])
                if self.segment_length - audio.size(0) > self.hop_length:
                    pad = np.ones((80, int((self.segment_length - audio.size(0)) / self.hop_length)), dtype=np.float32) * -11.512925
                    mel =  np.append(mel, pad, axis=1)
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

            mel = torch.from_numpy(mel).float()
            audio = audio / MAX_WAV_VALUE
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
    mel2samp = Mel2Samp(**data_config)

    # GTA mel_audio loader
    data_config['training_files'] = 'audio_mel_train.txt'
    data_config['load_mel_from_disk'] = True
    data_config['segment_length'] = 16128
    mel2samp2 = Mel2Samp(**data_config)

    # for testing orginal mel_audio loader
    mel1 , audio1_norm = mel2samp.__getitem__(0)
    mel1_org = mel2samp.get_mel(audio1_norm*MAX_WAV_VALUE)
    mel1, mel1_org = mel1.data.cpu().numpy(), mel1_org.data.cpu().numpy()
    print(len(audio1_norm), mel1_org.shape, mel1_org.max(), mel1_org.min(), mel1.shape, mel1.max(), mel1.min())
    plot_data([mel1_org, mel1], filename='mel1.png')

    # for testing GTA mel_audio loader
    mel2, audio2_norm = mel2samp2.__getitem__(1)
    mel2, audio2_norm = mel2samp2.__getitem__(1)
    mel2_org = mel2samp2.get_mel(audio2_norm * MAX_WAV_VALUE)
    mel2, mel2_org =  mel2.data.cpu().numpy(), mel2_org.data.cpu().numpy()
    print(len(audio2_norm), mel2_org.shape, mel2_org.max(), mel2_org.min(), mel2.shape, mel2.max(), mel2.min())
    plot_data([mel2_org, mel2], filename='mel2.png')

    # for testing GTA mel_audio loader and padding
    mel3, audio3_norm = mel2samp2.__getitem__(0)
    mel3_org = mel2samp2.get_mel(audio3_norm * MAX_WAV_VALUE)
    mel3, mel3_org = mel3.data.cpu().numpy(), mel3_org.data.cpu().numpy()
    print(len(audio3_norm), mel3_org.shape, mel3_org.max(), mel3_org.min(), mel3.shape, mel3.max(), mel3.min())
    plot_data([mel3_org, mel3], filename='mel3.png')

    pass

    # with open("config.json") as f:
    #     data = f.read()
    # data_config = json.loads(data)["data_config"]
    # mel2samp = Mel2Samp(**data_config)
    #
    # data_config['training_files'] = 'train_files.txt'
    # mel2samp2 = Mel2Samp(**data_config)
    #
    # filepaths = ['LJ050-0271.wav', 'LJ050-0278.wav']
    #
    # for filepath in filepaths:
    #     audio, sr = load_wav_to_torch(filepath)
    #     mel1_org = mel2samp.get_mel(audio).data.cpu().numpy()
    #     mel1_gta = np.load(filepath.replace('wav', 'mel') + '.npy')
    #     #print(mel1_org[:,675]) #check empty melspectrogram
    #     basename = os.path.basename(filepath)
    #     plot_data([mel1_org, mel1_gta], filename=basename + '.png')
    #     print(len(audio), mel1_org.shape, mel1_org.max(), mel1_org.min(), mel1_gta.shape, mel1_gta.max(), mel1_gta.min())

    ## Slicing debug
    # slice_index = 30
    # mel1 = mel2samp.get_mel(audio[slice_index*256:])
    # gta_melsepctrogram = gta_melsepctrogram[:,slice_index:]
    # basename = os.path.basename(filepath)
    # print(mel1.data.cpu().numpy().shape)
    # print(gta_melsepctrogram.shape)
    # plot_data([mel1, gta_melsepctrogram], filename=basename + '_slice.png')

    #assert n >= 0.0, 'Data should only contain positive values'