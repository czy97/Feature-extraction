'''
Created on Jan 19,2019

@author: Zhengyang Chen

Some functions in this .py are refered from librosa.MFCC
LFCC pipeline: Preemphasis --> stft --> linear filter bank --> log --> dct
'''

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import signal
import scipy.fftpack as fft
import scipy.io.wavfile


# Constrain STFT block sizes to 256 KB
MAX_MEM_BLOCK = 2**8 * 2**10

def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hamming',
         center=True, dtype=np.complex64, pad_mode='reflect'):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number audio of frames between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        .. see also:: `filters.get_window`

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype       : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix



    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))


    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def frame(y, frame_length=2048, hop_length=512):
    '''Slice a time series into overlapping frames.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        Time series to frame. Must be one-dimensional and contiguous
        in memory.

    frame_length : int > 0 [scalar]
        Length of the frame in samples

    hop_length : int > 0 [scalar]
        Number of samples to hop between frames

    Returns
    -------
    y_frames : np.ndarray [shape=(frame_length, N_FRAMES)]
        An array of frames sampled from `y`:
        `y_frames[i, j] == y[j * hop_length + i]`

    '''

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames

def get_window(windowType='hamming',Nx = 2048,fftbins=True):
    return signal.get_window(windowType, Nx, fftbins)

def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`
    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    return np.pad(data, lengths, **kwargs)

def filter_banks(sr, n_fft, n_banks=40, fmin=0.0, fmax=None):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

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
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

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

def fft_frequencies(sr=22050, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreq`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size


    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`
    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

def linearSpectrogram(y=None, sr=22050,  n_fft=512, win_length = 512,hop_length=256,power=2.0,f_min = 20.0,f_max = None):
    '''
    Get the spectrogram of y,and filter the spectrogram using the filterbank
    '''
    S, n_fft = spectrogram(y=y, n_fft=n_fft, win_length = win_length ,hop_length=hop_length,power=power)

    # Build a Mel filter
    filterbank = filter_banks(sr, n_fft,fmin=f_min, fmax=f_max)

    return np.dot(filterbank, S)

def spectrogram(y=None, n_fft=512, win_length = 512,hop_length=256, power=1):
    '''Helper function to retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.


    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If `S` is provided as input, then `S_out == S`
        - Else, `S_out = |stft(y, n_fft=n_fft, hop_length=hop_length)|**power`

    n_fft : int > 0
        - If `S` is provided, then `n_fft` is inferred from `S`
        - Else, copied from input
    '''


    S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length ,win_length = win_length))**power

    return S, n_fft

def lfcc(y=None, sr=16000, n_ceps=20,win_length = 512,n_fft = None ,f_min = 20.0,f_max = None):
    '''
    :param y: np.ndarray [shape=(n,)] audio time series
    :param sr: sample rate
    :param n_ceps: the ceps you want to get per frame
    :param win_length: window length of a frame
    :param n_fft: the number of fft bins
    :param f_min: the min frequency of filter banks
    :param f_max: the max frequency of filter banks
    :return: lfcc feature,np.ndarray [shape=(n_ceps,frames_num)]
    '''
    if(n_fft is None):
        n_fft = win_length
    hop_length = int(win_length/2)

    magnitude =  linearSpectrogram(y = y,sr =sr,n_fft = n_fft,win_length = win_length,hop_length =hop_length,f_min = f_min,f_max = f_max)
    log_spec = 10.0 * np.log10(np.maximum(1e-10, magnitude))
    return fft.dct(log_spec, axis=0, type=2, norm='ortho')[:n_ceps]

def get_lfcc_feature(wavpath, n_ceps=20,win_length = 512,n_fft = None ,f_min = 20.0,f_max = None):
    '''
    :param wavpath: the path of the .wav
    :param n_ceps: the ceps you want to get per frame
    :param win_length: window length of a frame
    :param n_fft: the number of fft bins
    :param f_min: the min frequency of filter banks
    :param f_max: the max frequency of filter banks
    :return: lfcc feature,np.ndarray [shape=(n_ceps,frames_num)]
    '''
    samplerate, data = scipy.io.wavfile.read(wavpath)

    data = preemphasis(data)
    return lfcc(y=data, sr=samplerate, n_ceps=n_ceps,win_length = win_length,n_fft = n_fft ,f_min = f_min,f_max = f_max)



if __name__ == '__main__':
    get_lfcc_feature('example.wav', n_ceps=20, win_length=512, n_fft=None, f_min=20.0, f_max=None)