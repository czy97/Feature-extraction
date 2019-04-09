"""
File: extract_raw.py
Author: 丁翰林
Email: heini89@gmail.com
Description: Extracts raw waveform features from KALDI format scp file
"""
import kaldi_io
import numpy as np
import librosa
from gammatone.gtgram import gtgram
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


def extract_gammatone(scp_file: str, output_ark: str, sr: int = None, win_length: int = 25, hop_length: int = 10):
    # Win length and hop_length are given in ms
    # scpfile is formatted as KEY VALUE where key is name and value is .wav filepath
    with open(scp_file, 'r') as rp, open(output_ark, 'wb') as wp:
        for line in tqdm(rp, total=get_num_lines(scp_file)):
            key, fpath = line.split()[:2]
            y, file_sr = librosa.load(fpath, sr=sr)
            # Adjust window length
            cur_win_length = win_length / 1000
            cur_hop_length = hop_length / 1000
            feature = gtgram(wave=y, fs=file_sr, f_min=0, channels=257,
                             window_time=cur_win_length, hop_time=cur_hop_length).transpose()
            kaldi_io.write_mat(wp, feature, key=key)


if __name__ == '__main__':
    Fire(extract_gammatone)
