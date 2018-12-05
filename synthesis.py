import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import sys
import time
import numpy as np
from scipy.io.wavfile import write
sys.path.insert(0, 'tacotron2')

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from mel2samp import MAX_WAV_VALUE

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')
    plt.savefig('test.png')

def run(sigma, taco_cp_path = "", wg_cp_path ="", cleaner=['english_cleaners'], is_fp16=True):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    # set 80 if u use korean_cleaners. set 149 if u use english_cleaners
    hparams.n_symbols = 80 if cleaner is ['korean_cleaners'] else 149

    model = load_model(hparams)
    model.load_state_dict(torch.load(taco_cp_path)['state_dict'])
    _ = model.eval()
    waveglow = torch.load(wg_cp_path)['model']
    waveglow.remove_weightnorm()
    waveglow.cuda().eval()
    if is_fp16:
        waveglow.half()
        for k in waveglow.convinv:
            k.float()

    text = "Scientists at the CERN laboratory say they have discovered a new particle."
    start = time.perf_counter()
    sequence = np.array(text_to_sequence(text, cleaner))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    duration = time.perf_counter() - start
    print("Tacotron2 inference time {:.2f}s/it".format(duration))

    # save figure
    plot_data((mel_outputs.data.cpu().numpy()[0],
               mel_outputs_postnet.data.cpu().numpy()[0],
               alignments.data.cpu().numpy()[0].T))

    mel = mel_outputs_postnet[-1, :, :]
    mel = torch.autograd.Variable(mel.cuda())
    mel = torch.unsqueeze(mel, 0)
    mel = mel.half() if is_fp16 else mel
    #mel = mel.data
    start = time.perf_counter()
    with torch.no_grad():
        audio = MAX_WAV_VALUE * waveglow.infer(mel, sigma=sigma)[0]
        #audio = waveglow.infer(mel, sigma=sigma)[0]
    duration = time.perf_counter() - start
    print("Waveglow inference time {:.2f}s/it".format(duration))

    audio = audio.data.cpu().numpy().astype('int16')
    print(audio.max(), audio.min(), audio.shape, audio.dtype)
    write('test.wav', hparams.sampling_rate, audio)

if __name__ == "__main__":
    import argparse

    #sigma , taco_cp_path = "tacotron2/tacotron2_statedict.pt", wg_cp_path ="waveglow_old.pt", text = '', cleaner=['english_cleaners'], is_fp16=True
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--taco_cp_path",
                        help='Path to tacotron decoder checkpoint with model', required=True)
    parser.add_argument('-w', '--wg_cp_path',
                        help='Path to waveglow decoder checkpoint with model', required=True)
    parser.add_argument('-c', "--cleaner")
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--is_fp16", action="store_true")

    args = parser.parse_args()

    run(args.sigma, args.taco_cp_path, args.wg_cp_path, [args.cleaner], args.is_fp16)