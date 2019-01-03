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
from model_old import Tacotron2_old
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from mel2samp import MAX_WAV_VALUE

def plot_data(data, index, output_dir="", figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')
    plt.savefig(os.path.join(output_dir, 'sentence_{}.png'.format(index)))

def generate_mels(taco2, sentences, cleaner, output_dir=""):
    output_mels = []
    for i, s in enumerate(sentences):
        sequence = np.array(text_to_sequence(s, [cleaner]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        stime = time.time()
        _, mel_outputs_postnet, _, alignments = taco2.inference(sequence)
        # plot_data((mel_outputs_postnet.data.cpu().numpy()[0],
        #            alignments.data.cpu().numpy()[0].T), i, output_dir)
        inf_time = time.time() - stime
        str = "{}th sentence, tacotron Infenrece time: {:.2f}s, len_mel: {}".format(i, inf_time, mel_outputs_postnet.size(2))
        print(str)
        output_mels.append(mel_outputs_postnet[:,:,:-3])

    return output_mels

def mels_to_wavs_WG(waveglow, mels, sigma, is_fp16, hparams, output_dir=""):
    for i, mel in enumerate(mels):
        mel = mel[-1, :, :]
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        #mel = mel.data
        stime = time.time()
        with torch.no_grad():
            audio = MAX_WAV_VALUE * waveglow.infer(mel, sigma=sigma)[0]
        inf_time = time.time() - stime
        audio = audio.data.cpu().numpy().astype('int16')
        len_audio = float(len(audio)) / float(hparams.sampling_rate)
        str = "{}th sentence, audio length: {:.2f} sec,  waveglow inf time: {:.2f}".format(i, len_audio, inf_time)
        print(str)
        write(os.path.join(output_dir, "sentence_{}.wav".format(i)), hparams.sampling_rate, audio)

def run(sigma, sentence_path, taco_cp_path, wg_cp_path, cleaner='english_cleaners', output_dir='', is_fp16=True):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    # set 80 if u use korean_cleaners. set 149 if u use english_cleaners
    hparams.n_symbols = 80 if cleaner == 'korean_cleaners' else 148

    f = open(sentence_path,'r')
    sentences = [x.strip() for x in f.readlines()]
    f.close()

    model = load_model(hparams)
    model.load_state_dict(torch.load(taco_cp_path)['state_dict'])
    _ = model.cuda().eval()
    waveglow = torch.load(wg_cp_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()

    if is_fp16:
        model.half()
        waveglow.half()
        for k in waveglow.convinv:
            k.float()

    mel_outputs = generate_mels(model, sentences, cleaner, output_dir)
    mels_to_wavs_WG(waveglow, mel_outputs, sigma, is_fp16, hparams, output_dir)


if __name__ == "__main__":
    import argparse
    """
    usage
    python synthesis.py -t=tacotron2/nam-h-ep8/checkpoint_80000 -w=checkpoints/waveglow_40000 -s=kor_test.txt -c=korean_cleaners --is_fp16
    python synthesis.py -t=tacotron2/tacotron2_statedict.pt -w=waveglow_old.pt -s=eng_test.txt -c=english_cleaners --is_fp16
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default='',
                        help='directory to save wave and fig')
    parser.add_argument('-t', "--taco_cp_path",
                        help='Path to tacotron decoder checkpoint with model', required=True)
    parser.add_argument('-w', '--wg_cp_path',
                        help='Path to waveglow decoder checkpoint with model', required=True)
    parser.add_argument('-c', "--cleaner")
    parser.add_argument('-s','--sentence_path', type=str, default=None,
                        required=True, help='sentence path')
    parser.add_argument('--silence_mel_padding', type=int, default=0,
                        help='silence audio size is hop_length * silence mel padding')
    parser.add_argument("--sigma", default=1.0, type=float)
    parser.add_argument("--is_fp16", action="store_true")

    args = parser.parse_args()

    run(args.sigma, args.sentence_path, args.taco_cp_path, args.wg_cp_path, args.cleaner, args.output_directory, args.is_fp16)

