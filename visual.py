import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

# -----------------------------
# 读取音频
# -----------------------------
audio_path = "E:/Awork/data/shipsEar/train/6_4.wav"
sr, data = wavfile.read(audio_path)

# 若为双声道，取一个通道
if len(data.shape) > 1:
    data = data[:, 0]

# 转 float 型
data = data.astype(np.float32)

# -----------------------------
# 时间轴
# -----------------------------
t = np.arange(len(data)) / sr

# -----------------------------
# FFT 幅度谱
# -----------------------------
N = len(data)
fft_data = np.fft.rfft(data)
fft_freq = np.fft.rfftfreq(N, d=1/sr)
fft_amp = np.abs(fft_data) / N   # 振幅
# 单位：V（不再画 dB）

# -----------------------------
# Welch PSD
# -----------------------------
f_psd, Pxx = welch(data, fs=sr, nperseg=2048)

# -----------------------------
# 绘图（3 子图）
# -----------------------------
plt.figure(figsize=(14, 10))

# ----- (1) 原始波形 -----
plt.subplot(3, 1, 1)
plt.plot(t, data)
plt.title("Original Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.grid(True)

# ----- (2) FFT 幅度谱 -----
plt.subplot(3, 1, 2)
plt.plot(fft_freq, fft_amp)
plt.title("FFT Amplitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (V)")
plt.xlim([0, 5000])   # 水声常用频段
plt.grid(True)

# ----- (3) PSD（Welch） -----
plt.subplot(3, 1, 3)
plt.semilogy(f_psd, Pxx)
plt.title("Power Spectral Density (Welch Method)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V²/Hz)")
plt.xlim([0, 5000])
plt.grid(True)

plt.tight_layout()
plt.show()
