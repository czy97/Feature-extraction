"""
Author: Zhengyang Chen
Email: chenzhengyang117@gmail.com
Description: Extract spec feature. The input and output are all in kaldi format.
"""
import kaldi_io
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from fire import Fire



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

def spec(filepath: str, win_length: int = 25, hop_length: int  = 10,n_fft: int = 512):
    '''
    :param filepath: wav file path
    :param win_length: ms(int)
    :param hop_length: ms(int)
    :param n_fft: fft bins number
    :return: spec feature
    '''
    y, file_sr = sf.read(filepath, dtype='float32')

    cur_win_length = file_sr * win_length // 1000
    cur_hop_length = file_sr * hop_length // 1000

    S = librosa.core.stft(
        y, n_fft=n_fft, win_length=cur_win_length, hop_length=cur_hop_length)
    S = librosa.core.power_to_db(np.abs(S)**2)

    return S.transpose()



def extract(scp_file: str, output_ark: str, win_length: int = 25, hop_length: int = 10, n_fft: int = 512):

    file_key_path = getlines(scp_file)
    with open(output_ark, 'wb') as wp:
        for line in tqdm(file_key_path,ncols=100):
            feat = spec(line[1],win_length = win_length,hop_length = hop_length,n_fft = n_fft)
            kaldi_io.write_mat(wp, feat, key=line[0])

    return



if __name__ == '__main__':
    Fire(extract)
    # feat = spec('example.wav')
    # print(feat.shape)
