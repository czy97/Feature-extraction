"""
File: extract_log_cqt.py
Author: 丁翰林
Email: heini89@gmail.com
Description: Extracts log_cqt features from KALDI format scp file
"""
import kaldi_io
import numpy as np
import librosa
from tqdm import tqdm
from fire import Fire
import soundfile as sf
import mmap

# Just for pretty tqdm


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def extract_feature(scp_file: str, output_ark: str, sr: int = None, hop_length: int = 8, eps: float = 1e-10):
    # scpfile is formatted as KEY VALUE where key is name and value is .wav filepath
    with open(scp_file, 'r') as rp, open(output_ark, 'wb') as wp:
        for line in tqdm(rp, total=get_num_lines(scp_file)):
            key, fpath = line.split()[:2]
            # y, file_sr = librosa.load(fpath, sr=sr)
            y, file_sr = sf.read(fpath, dtype='float32')
            # Adjust window length
            cur_hop_length = file_sr * hop_length // 1000
            feature = librosa.core.cqt(y, file_sr, hop_length=cur_hop_length)
            # Extract log cqt
            feature = np.log(np.abs(feature)**2 + eps).transpose()
            kaldi_io.write_mat(wp, feature, key=key)


if __name__ == '__main__':
    Fire(extract_feature)
