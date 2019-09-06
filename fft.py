import os
from os.path import isdir, join
from pathlib import Path

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
from pydub import AudioSegment
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, vals

song = AudioSegment.from_wav("Dung28082019.wav")
# song = song + 10
song.export("origin.wav", format="wav")
# new = song.high_pass_filter(40)
# new.export("lowpass.wav", format="wav")
# new = AudioSegment.from_wav("lowpass.wav")
# new = new.low_pass_filter(50)
# new.export("lowpass.wav", format="wav")

# sample_rate0, samples0 = wavfile.read("origin.wav")
# sample_rate1, samples1 = wavfile.read("lowpass.wav")
# xf, vals = custom_fft(samples0, sample_rate0)

# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(311)
# ax1.set_title('FFT of recording sampled with')
# ax1.set_xlabel('Frequency')
# plt.plot(xf, vals)
# # plt.ylim(-100, 100)
# plt.grid()

# ax2 = fig.add_subplot(312)
# ax2.set_title('Raw wave of origin')
# ax2.set_ylabel('Amplitude')
# plt.ylim(-2000, 2000)
# ax2.plot(np.linspace(0, sample_rate0/len(samples1), len(samples0)), samples0)

# ax3 = fig.add_subplot(313)
# ax3.set_title('Raw wave of origin')
# ax3.set_ylabel('Amplitude')
# plt.ylim(-2000, 2000)
# ax3.plot(np.linspace(0, sample_rate1/len(samples1), len(samples1)), samples1)

# plt.show()