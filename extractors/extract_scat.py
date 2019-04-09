"""
File: extract_log_cqt.py
Author: 丁翰林
Email: heini89@gmail.com
Description: Extracts log_cqt features from KALDI format scp file
"""
import kaldi_io
import numpy as np
import librosa
import torch
from tqdm import tqdm
from fire import Fire
import mmap
from kymatio import Scattering1D

# Just for pretty tqdm


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


CUDA_AVAIL = torch.cuda.is_available()
device = torch.device('cuda' if CUDA_AVAIL else 'cpu')


def extract_feature(scp_file: str, output_ark: str, sr: int = None, hop_length: int = 8, eps: float = 1e-10, Q: int = 12, J: int = 5, batch_size: int = 64):
    raw_audio = []
    # scpfile is formatted as KEY VALUE where key is name and value is .wav filepath
    with open(scp_file, 'r') as rp, open(output_ark, 'wb') as wp:
        for n_line, line in enumerate(tqdm(rp, total=get_num_lines(scp_file))):
            key, fpath = line.split()[:2]
            y, file_sr = librosa.load(fpath, sr=sr)
            raw_audio.append((key, y))
            if (n_line + 1) % batch_size == 0:
                maxlen = max([len(x[1]) for x in raw_audio])
                X = torch.zeros(len(raw_audio), maxlen)
                for i, (key, y) in enumerate(raw_audio):
                    X[i, :y.shape[-1]] = torch.from_numpy(y)
                X = X.to(device).float()
                scattering = Scattering1D(J, maxlen, Q)
                if CUDA_AVAIL:
                    scattering.cuda()
                # First component is constant
                feature = scattering(X)[:, 1:, :]
                # Feature is shape T, DIM, BATCH
                feature = torch.log(torch.abs(feature) +
                                    eps).cpu().numpy()
                for i, (key, _) in enumerate(raw_audio):
                    kaldi_io.write_mat(wp, feature[i, :, :].transpose(), key=key)
                # Reset buffer
                raw_audio = []


if __name__ == '__main__':
    Fire(extract_feature)
