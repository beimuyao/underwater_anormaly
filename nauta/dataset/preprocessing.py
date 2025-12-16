from nnAudio import Spectrogram
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import torch.nn.functional as F

FREQ_BINS = 95 # This number was based on the CQT, which have 95 freq bins for 4186hz
HOP_LENGTH = 256 # Used to generate an output of 128 on x axis
N_FFT = 2048 # This value is UNUSED because of the freq bins is mandatory
FMAX = 4186 # Correspond to a C8 note (Most High on a piano) (empirical)
FMIN = 18.0 # Minimun accepted value on CQT for audios of 1s

def define_mel_spectrogram(sample_rate):
    """Returns a MelSpectrogram transforms object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        torchaudio.transforms: The MelSpectrogram object initialized.
    """
    mel_spectrogram = Spectrogram.MelSpectrogram(
        sr=sample_rate, n_fft=N_FFT, n_mels=128, hop_length=HOP_LENGTH,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=0, fmax=sample_rate/2, norm=1,
        trainable_mel=False, trainable_STFT=False, verbose=False
    )
    return mel_spectrogram


def define_gamma_spectrogram(sample_rate):
    """Returns a Gammatonegram object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        Spectrogram: The Gammatonegram object initialized.
    """
    gamma_spectrogram = Spectrogram.Gammatonegram(
        sr=sample_rate, n_fft=N_FFT, n_bins=FREQ_BINS, hop_length=HOP_LENGTH,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=FMIN, fmax=FMAX, norm=1,
        trainable_bins=False, trainable_STFT=False, verbose=False
    )
    return gamma_spectrogram


def define_cqt_spectrogram(sample_rate):
    """Returns a CQT object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        Spectrogram: The CQT object initialized.
    """
    cqt_spectrogram = Spectrogram.CQT(
        sr=sample_rate, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX,
        n_bins=FREQ_BINS, bins_per_octave=12, filter_scale=1, norm=1,
        window='hann', center=True, pad_mode='reflect', trainable=False,
        output_format='Magnitude', verbose=False
    )
    return cqt_spectrogram


_pre_processing_layers = {
    "mel": define_mel_spectrogram,
    "gammatone": define_gamma_spectrogram,
    "cqt": define_cqt_spectrogram,
}


def get_preprocessing_layer(pre_processing_type, sample_rate):
    return _pre_processing_layers[pre_processing_type](sample_rate)

def resize_spec(spec, target_size=(128, 128)):
    # spec shape: [1, H, W]
    spec = spec.unsqueeze(0)  # -> [1,1,H,W]
    spec = F.interpolate(spec, size=target_size, mode="bilinear", align_corners=False)
    return spec.squeeze(0)  # -> [1,256,256]

# 假设 mel, lofar, demon 形状如下：
# mel = [1, 95, 305]
# lofar = [1, 512, 400]
# demon = [1, 256, 300]



def main():
    # 1. 读入音频
    wav_path = "E:/Awork/data/shipsEar/train_back/91_6.wav"   # 改成你的文件
    audio, sr = sf.read(wav_path)

    # 如果是双通道，取一个通道
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 转成 Tensor (是给 torchaudio / torchlibrosa 用的)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, T]

    # 2. 获取 mel 对象
    mel_spec_func = define_mel_spectrogram(sr)

    # 3. 计算 Mel 频谱图
    mel = mel_spec_func(audio_tensor)  # 输出形状: [1, n_mels, time]
    mel_r   = resize_spec(mel)
    mel = mel_r.squeeze(0).detach().numpy()   # 转成 numpy


    # 4. 显示 Mel 图
    plt.figure(figsize=(10, 6))
    plt.imshow(10 * np.log10(mel + 1e-6), aspect='auto', origin='lower')
    plt.colorbar(label="dB")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.show()

if __name__ == "__main__":
    main()
