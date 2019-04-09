"""
Author: Zhengyang Chen
Email: chenzhengyang117@gmail.com
Description: Extract mfcc feature. The input and output are all in kaldi format.
"""

import kaldi_io
import numpy as np
import librosa
from tqdm import tqdm
from fire import Fire
import soundfile as sf
import scipy




def getlines(file):
    '''
    :param file: kaldi wav.scp format, every line's format: key wav_file_path
    :return: the list contain (key,wav_file_path) pair
    '''
    out_list = []
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if(len(line) != 2):
                raise Exception("Wrong data format")
            out_list.append((line[0],line[1]))
    return out_list

def imfcc(filepath: str, win_length: int = 25, hop_length: int  = 10,n_fft: int = 512,n_mels:int = 40):
    '''
    :param filepath: wav file path
    :param win_length: ms(int)
    :param hop_length: ms(int)
    :param n_fft: fft bins number
    :param n_mels: the mel filter bank number
    :return: mfcc feature
    '''
    y, file_sr = sf.read(filepath, dtype='float32')

    cur_win_length = file_sr * win_length // 1000
    cur_hop_length = file_sr * hop_length // 1000

    S = np.abs(librosa.core.stft(
        y, n_fft=n_fft, win_length=cur_win_length, hop_length=cur_hop_length))

    # Build a Mel filter
    mel_basis = librosa.filters.mel(
        file_sr, n_fft, n_mels = n_mels)
    inverse_mel_basis = np.flip(mel_basis, axis=1)

    mel_fbank = np.dot(inverse_mel_basis, S)

    S = librosa.core.power_to_db(mel_fbank)
    cc = scipy.fftpack.dct(S, axis=0,  type=2, norm='ortho').transpose()

    return cc


def extract_imfcc(scp_file: str, output_ark: str, ceps: int = 13,win_length: int = 25, hop_length: int = 10, n_fft: int = 512,
            n_mels: int = 40):
    file_key_path = getlines(scp_file)
    with open(output_ark, 'wb') as wp:
        for line in tqdm(file_key_path, ncols=100):
            feat = imfcc(line[1], win_length=win_length, hop_length=hop_length,
                         n_fft=n_fft,n_mels=n_mels)[:,:ceps]
            kaldi_io.write_mat(wp, feat, key=line[0])

    return


if __name__ == '__main__':
    Fire(extract_imfcc)
