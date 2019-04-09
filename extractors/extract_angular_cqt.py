"""
Author: 丁翰林
Email: heini89@gmail.com
Description: Extracts harmonics + percussive
"""
import kaldi_io
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from fire import Fire
import mmap

# Just for pretty tqdm


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def extract(scp_file: str, output_ark: str, sr: int = None, win_length: int = 25, hop_length: int = 8, n_fft: int = 512, ftype:str='angle'):
    # Default postprocess, just extract angle 0 - 2Pi
    postprocess = lambda x: x
    if ftype == 'sin':
        postprocess = np.sin
    elif ftype == 'cos':
        postprocess = np.cos
    elif ftype == 'sincos':
        postprocess = lambda x: np.concatenate((np.sin(x), np.cos(x)))
    
    # Win length and hop_length are given in ms
    # scpfile is formatted as KEY VALUE where key is name and value is .wav filepath
    with open(scp_file, 'r') as rp, open(output_ark, 'wb') as wp:
        for line in tqdm(rp, total=get_num_lines(scp_file)):
            key, fpath = line.split()[:2]
            y, file_sr = sf.read(fpath,dtype='float32')
            # y, file_sr = librosa.load(fpath, sr=sr)
            # Adjust window length
            cur_win_length = file_sr * win_length // 1000
            cur_hop_length = file_sr * hop_length // 1000
            # Extract cqt
            S = librosa.core.cqt(y, file_sr, hop_length=cur_hop_length)
            feature = np.angle(S)
            feature = postprocess(feature).transpose()
            #  components and acitivations
            kaldi_io.write_mat(wp, feature, key=key)


if __name__ == '__main__':
    Fire(extract)
