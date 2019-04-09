"""
Author: 丁翰林
Email: heini89@gmail.com
"""
import kaldi_io
import numpy as np
import librosa
from tqdm import tqdm
from fire import Fire
import mmap
import scipy
import soundfile as sf

# Just for pretty tqdm


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def extract(scp_file: str, output_ark: str, sr: int = None, win_length: int = 25, hop_length: int = 10, n_mels: int = 64, ftype: str = 'angle'):
    # Win length and hop_length are given in ms
    def postprocess(x): return x
    if ftype == 'sin':
        postprocess = np.sin
    elif ftype == 'cos':
        postprocess = np.cos
    elif ftype == 'sincos':
        def postprocess(x): return np.concatenate((np.sin(x), np.cos(x)))
    # scpfile is formatted as KEY VALUE where key is name and value is .wav filepath
    with open(scp_file, 'r') as rp, open(output_ark, 'wb') as wp:
        for line in tqdm(rp, total=get_num_lines(scp_file)):
            key, fpath = line.split()[:2]
            y, file_sr = sf.read(fpath, dtype='float32')
            # Adjust window length
            cur_win_length = file_sr * win_length // 1000
            cur_hop_length = file_sr * hop_length // 1000
            n_fft = 512
            S = librosa.core.stft(
                y, n_fft=n_fft, hop_length=cur_hop_length, win_length=cur_win_length)
            # Get angle
            S = np.angle(S)
            S = postprocess(S)
            # Build a Mel filter
            mel_basis = librosa.filters.mel(file_sr, n_fft, n_mels=n_mels)

            inv_lms = np.dot(mel_basis, S).transpose()
            kaldi_io.write_mat(wp, inv_lms, key=key)


if __name__ == '__main__':
    Fire(extract)
