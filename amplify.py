import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment

# rate, data = wavfile.read('Dung28082019.wav')
# shifted = data * (2 ** 32 - 1)   # Data ranges from -1.0 to 1.0
# ints = shifted.astype(np.int32)
# wavfile.write("export.wav", rate, ints)

song = AudioSegment.from_wav("Dung28082019.wav")
song = song + 30
song.export("test0.wav", format="wav")

