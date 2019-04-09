"""
Author: Zhengyang Chen
Email: chenzhengyang117@gmail.com
Description: Extract lfcc feature. The input and output are all in kaldi format.
"""
import kaldi_io
import numpy as np
import librosa
from tqdm import tqdm
from fire import Fire
import soundfile as sf
import scipy


def filter_banks(sr, n_fft, n_banks=40, fmin=0.0, fmax=None):
    """Create a Filterbank matrix to combine FFT bins into Linear-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_banks    : int > 0 [scalar]
        number of filter banks to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`


    Returns
    -------
    M         : np.ndarray [shape=(n_banks, 1 + n_fft/2)]
        Filter bank transform matrix

    """

    if fmax is None:
        fmax = float(sr) / 2


    # Initialize the weights
    n_banks = int(n_banks)
    weights = np.zeros((n_banks, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = np.linspace(0,
                float(sr) / 2,
                int(1 + n_fft // 2),
                endpoint=True)
    # 'Center freqs' of mel bands - uniformly spaced between limits

    banks_loc = np.linspace(fmin, fmax, n_banks + 2)

    fdiff = np.diff(banks_loc)
    ramps = np.subtract.outer(banks_loc, fftfreqs)

    for i in range(n_banks):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return weights


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

def lfcc(filepath: str, win_length: int = 25, hop_length: int  = 10,n_fft: int = 512,n_mels:int = 40,fmin: float = 0.0, fmax: float = None):
    '''
    :param filepath: wav file path
    :param win_length: ms(int)
    :param hop_length: ms(int)
    :param n_fft: fft bins number
    :param n_mels: the mel filter bank number
    :param fmin: the start frequency of mel filter bank
    :param fmax: the end of the mel filter bank
    :return: mfcc feature
    '''
    y, file_sr = sf.read(filepath, dtype='float32')

    cur_win_length = file_sr * win_length // 1000
    cur_hop_length = file_sr * hop_length // 1000

    S = np.abs(librosa.core.stft(
        y, n_fft=n_fft, win_length=cur_win_length, hop_length=cur_hop_length))

    # Build a Mel filter
    linear_basis = filter_banks(
        file_sr, n_fft, n_banks = n_mels, fmin=fmin, fmax=fmax)

    linear_fbank = np.dot(linear_basis, S)

    S = librosa.core.power_to_db(linear_fbank)
    cc = scipy.fftpack.dct(S, axis=0,  type=2, norm='ortho').transpose()

    return cc



def extract_lfcc(scp_file: str, output_ark: str, ceps: int = 13,win_length: int = 25, hop_length: int = 10, n_fft: int = 512,
            n_mels: int = 40, fmin: float = 0.0, fmax: float = None):
    file_key_path = getlines(scp_file)
    with open(output_ark, 'wb') as wp:
        for line in tqdm(file_key_path, ncols=100):
            feat = lfcc(line[1], win_length=win_length, hop_length=hop_length,n_fft=n_fft,
                        n_mels=n_mels,fmin=fmin, fmax=fmax)[:,:ceps]
            kaldi_io.write_mat(wp, feat, key=line[0])

    return


if __name__ == '__main__':
    # feat = lfcc('example.wav')
    # print(feat.shape)
    Fire(extract_lfcc)
