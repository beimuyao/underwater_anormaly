from nnAudio.features import MelSpectrogram, Gammatonegram, CQT
import torch
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import torch.nn.functional as F
import torchaudio
# import spec_augment_pytorch
import librosa

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
    mel_spectrogram = MelSpectrogram(
        sr=sample_rate, n_fft=2048, n_mels=128, hop_length=640,
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=0, fmax=2000, norm=1,
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
    gamma_spectrogram = Gammatonegram(
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
    cqt_spectrogram = CQT(
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
def pad(mel, target_width=128):
    if mel.dim() == 3:
        # 单张 Mel: [C, F, T] -> 增加 batch 维度
        mel = mel.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    _, _, _, time_frames = mel.shape
    pad_amount = target_width - time_frames
    if pad_amount <= 0:
        return mel[:, :, :, :target_width]  # 截断

    # pad=(left, right, top, bottom)，这里在右侧 pad
    mel_padded = F.pad(mel, (0, pad_amount, 0, 0), "constant", 0.0)

    if squeeze:
        mel_padded = mel_padded.squeeze(0)
    return mel_padded
# 假设 mel, lofar, demon 形状如下：
# mel = [1, 95, 305]
# lofar = [1, 512, 400]
# demon = [1, 256, 300]



def main():
    # 1. 读入音频
    wav_path = "H:/data/test/15_18.wav"   # 改成你的文件
    audio, sr = sf.read(wav_path)

    # 如果是双通道，取一个通道
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 如果是双通道，取一个通道
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 转成 Tensor (是给 torchaudio / torchlibrosa 用的)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, T]


    mel_pro = define_mel_spectrogram(sr)
    mel = mel_pro(audio_tensor)
    mel_pad = F.pad(mel, (0,2), mode='replicate')
    # mel_db = librosa.power_to_db(mel)
    # delta1 = librosa.feature.delta(mel_db, order=1)
    # delta2 = librosa.feature.delta(mel_db, order=2)
    # X = np.stack([mel_db, delta1, delta2], axis=0)
    
    mel_db = 10 * torch.log(mel_pad + 1e-6)
    # spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=mel_db,
    #                                                   title="Raw Mel Spectrogram")
    # warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_db,time_warping_para=0)
    # spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=warped_masked_spectrogram,
    #                                                   title="pytorch Warped & Masked Mel Spectrogram")
    # delta1 = mel_db[:, :, 1:] - mel_db[:, :, :-1]   # [1,128,127]
    # delta1 = F.pad(delta1, (0,1),mode='replicate')                   # [1,128,128]

    # delta2 = delta1[:, :, 1:] - delta1[:, :, :-1]  # [1,128,127]
    # delta2 = F.pad(delta2, (0,1),mode='replicate')                   # [1,128,128]
    # X = torch.cat([mel_db, delta1, delta2], dim=0)
    # print(X.shape)


    mel = mel_db.squeeze(0).detach().numpy()   # 转成 numpy
    # warped=warped_masked_spectrogram.squeeze(0).detach().numpy()
    mel_freqs = librosa.mel_frequencies(
        n_mels=128,
        fmin=0,
        fmax=2000
    )   

    # 4. 显示 Mel 图
    plt.figure(figsize=(10, 6))
    plt.imshow(mel, aspect='auto', origin='lower')
    idx = np.linspace(0, 128-1, 6, dtype=int)
    plt.yticks(
        idx,
        [f"{int(mel_freqs[i])}" for i in idx]
    )

    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time frame")
    plt.colorbar(label="Log-Mel Energy")
    plt.title("Log-Mel Spectrogram (nnAudio)")
    plt.show()
    

if __name__ == "__main__":
    main()
