"""
File: extract_raw.py
Author: 丁翰林
Email: heini89@gmail.com
Description: Extracts raw waveform features from KALDI format scp file
"""
import kaldi_io
import numpy as np
import librosa
from tqdm import tqdm
from fire import Fire
import mmap
import scipy

# Just for pretty tqdm


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def group_delay(phase):
    unwrapped_phase = np.unwrap(phase)
    # omega(t, w) - omega(t, w -1)
    unwrapped_phase[:, :-1] -= unwrapped_phase[:, 1:]
    # set the highest frequency to 0
    unwrapped_phase[:, -1] = 0
    return unwrapped_phase


def extract_imfcc_gd(scp_file: str, output_ark: str, sr: int = None, win_length: int = 25, n_fft: int = 512, hop_length: int = 10, fmin: int = 0.0, fmax: int = None):
    # Win length and hop_length are given in ms
    # scpfile is formatted as KEY VALUE where key is name and value is .wav filepath
    with open(scp_file, 'r') as rp, open(output_ark, 'wb') as wp:
        for line in tqdm(rp, total=get_num_lines(scp_file)):
            key, fpath = line.split()[:2]
            y, file_sr = librosa.load(fpath, sr=sr)
            # Adjust window length
            cur_win_length = file_sr * win_length // 1000
            cur_hop_length = file_sr * hop_length // 1000
            S = librosa.core.stft(
                y, n_fft=n_fft, hop_length=cur_hop_length, win_length=cur_win_length)
            phase = np.angle(S)
            gd = group_delay(phase)
            # Build a Mel filter
            mel_basis = librosa.filters.mel(
                file_sr, n_fft, fmin=fmin, fmax=fmax)
            inverse_mel_basis = np.flip(mel_basis, axis=1)

            lms = np.dot(inverse_mel_basis, gd)
            S = librosa.core.power_to_db(lms)
            feature = scipy.fftpack.dct(
                S, axis=0,  type=2, norm='ortho')[:30].transpose()
            kaldi_io.write_mat(wp, feature, key=key)


if __name__ == '__main__':
    Fire(extract_imfcc_gd)
