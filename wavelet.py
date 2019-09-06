import sys
import math
import pywt
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
import math

sampleRate = 44100
frequency0 = 440
frequency1 = 35
frequency2 = 10
length = 5

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
sample_rate, y = wavfile.read("origin.wav")
y.astype(float)
y = y / 32767
# t = np.linspace(0, length, sampleRate * length)  #  Produces a 5 second Audio-File
# y0 = np.sin(frequency0 * 2 * np.pi * t)  #  Has frequency of 440Hz
# y1 = np.sin(frequency1 * 2 * np.pi * t)
# y1 = np.sin(frequency1 * 2 * np.pi * t)
# y = y0 + y1

w = pywt.Wavelet('sym4')
central_frequency = pywt.central_frequency(w, precision=8)
print('central_frequency ', central_frequency)
maxlev = pywt.dwt_max_level(len(y), w.dec_len)
# maxlev = 2 # Override if desired
print("maximum level is " + str(maxlev))
threshold = 0.04 # Threshold for filtering

# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(y, 'sym4', level=18)

# cA = pywt.threshold(cA, threshold*max(cA))
plt.figure()
for i in range(1, len(coeffs)):
    plt.subplot(maxlev, 1, i)
    plt.plot(coeffs[i])
    frequency = pywt.scale2frequency(w, [i])
    print('frequency ', 44100 / i)
    if i < 3 or i > 10:
        coeffs[i] = np.zeros(len(coeffs[i]))
    else:
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))


datarec = pywt.waverec(coeffs, 'sym4')
wavfile.write("lowpass.wav", sample_rate, datarec)
xf, vals = custom_fft(datarec, sample_rate)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(311)
ax1.set_title('FFT of recording sampled with')
ax1.set_xlabel('Frequency')
plt.plot(xf, vals)
plt.xlim(0, 300)
plt.grid()


mintime = 1000
maxtime = mintime + 2000
index = []
for i in range(0, len(y)):
    index.append(i)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index, y)
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(index, datarec)
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()
