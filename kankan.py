import os
import pandas as pd
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from nauta.dataset.preprocessing import define_mel_spectrogram  # 你的函数
from nauta.dataset.preprocessing import pad  # 如果你有 pad 函数

# CSV 路径
csv_path = "H:/data/qiandao/train/back.csv"
# 音频文件所在文件夹
audio_dir = "H:/data/qiandao/train"
# 保存频谱图的文件夹
save_dir = "H:/data/qiandao/mel_spectrograms_train"
os.makedirs(save_dir, exist_ok=True)

# 读取 CSV
df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    file_name = row[0]        # 第一列: 文件名
    # label = row[1]            # 第二列: label
    wav_path = os.path.join(audio_dir, file_name)

    # 1. 读音频
    try:
        audio, sr = sf.read(wav_path)
    except RuntimeError as e:
        print(f"Error reading {wav_path}: {e}")
        continue

    if audio.ndim > 1:
        audio = audio[:, 0]

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    # 2. Mel 频谱
    mel_spec_func = define_mel_spectrogram(sr)
    mel = mel_spec_func(audio_tensor)
    mel = pad(mel)
    mel = mel.squeeze(0).detach().numpy()

    # 3. 保存频谱图
    plt.figure(figsize=(10, 6))
    plt.imshow(10 * np.log10(mel + 1e-6), aspect='auto', origin='lower')
    plt.colorbar(label="dB")
    plt.title(f"{file_name} ")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")

    save_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved {save_path}")
