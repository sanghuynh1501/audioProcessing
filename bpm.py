import heartpy as hp
from scipy.io import wavfile

sample_rate, samples = wavfile.read("heartbeat.wav")
working_data, measures = hp.process(samples, sample_rate, report_time=True)

print(measures['bpm']) #returns BPM value
print(measures['rmssd']) # returns RMSSD HRV measure