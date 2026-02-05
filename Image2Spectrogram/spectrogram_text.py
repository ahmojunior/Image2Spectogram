import numpy as np
from scipy.io.wavfile import write
from PIL import Image

img_path = "message.png"
sample_rate = 44100
duration_per_column = 0.05
img = Image.open(img_path).convert('L')
img = img.resize((64, 16))
img = np.array(img)
img_norm = 1 - img / 255.0
img_norm = np.flipud(img_norm)

freq_start = 300  
freq_end = 8000
num_freq_bins = img_norm.shape[0]

samples_per_col = int(sample_rate * duration_per_column)
total_samples = samples_per_col * img_norm.shape[1]

signal = np.zeros(total_samples)

for col in range(img_norm.shape[1]):
    t = np.linspace(0, duration_per_column, samples_per_col, False)
    column_signal = np.zeros(samples_per_col)
    for row in range(num_freq_bins):
        intensity = img_norm[row, col]
        if intensity > 0.5:
            freq = freq_start + (freq_end - freq_start) * (row / (num_freq_bins - 1))
            column_signal += np.sin(2 * np.pi * freq * t)
    start = col * samples_per_col
    signal[start:start + samples_per_col] = column_signal


signal /= np.max(np.abs(signal))

write("output.wav", sample_rate, (signal * 32767).astype(np.int16))
print("output.wav created, now look at the spectrogram")
