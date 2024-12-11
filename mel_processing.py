import os
import pickle as pkl
import argparse
import glob
from tqdm import tqdm
import librosa
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T
from torchaudio.functional import compute_deltas

def get_3dmelspec_from_file(filepath, max_sec, pre_emph, fft_sec, hop_sec, window_sec, n_mels, device=None):
    log_mel_spec = get_melspec_from_file(filepath, max_sec, pre_emph, fft_sec, hop_sec, window_sec, n_mels, device)
    return calc_deltas(log_mel_spec)

def calc_deltas(mel_spec):
    delta1 = compute_deltas(mel_spec, win_length=5, mode='replicate')
    delta2 = compute_deltas(delta1, win_length=5, mode='replicate')
    mel_specs_3d = torch.cat((mel_spec, delta1, delta2), dim=0)
    return mel_specs_3d

def get_melspec_from_file(filepath, max_sec, pre_emph, fft_sec, hop_sec, window_sec, n_mels, device=None):
    # 音声の読み込み
    waveform, sr = torchaudio.load(filepath)
    if device is not None:
        waveform = waveform.to(device)
        # # DEBUG
        # # torch.tensor.to()はnn.Module.to()とは異なり，
        # # 返り値がtensorでこれを受けないとgpuに転送したtensorにならない
        # print(f'{device=}')
        # print(f'{waveform.device=}')
        # exit(0)
    
    # 長さをmax_secへ揃える
    max_sample = int(sr * max_sec)
    pad_size = int(max_sample - waveform.size(1))
    if pad_size <= 0:
        waveform = waveform[:,:max_sample]

    else:
        waveform = F.pad(waveform, (0,pad_size), mode='constant', value=0)
        
    # プリエンファシス（高域強調）
    if pre_emph:
        waveform = pre_emphasis(waveform)
    hop_length = int(sr * hop_sec)
    window_n_sample = int(sr * window_sec)
    n_fft = int(sr * fft_sec)

    transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        f_min=20.0,
        f_max=sr // 2.0,
        win_length=window_n_sample,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )
    if device is not None:
        transform.to(device)
        
        # # DEBUG
        # print(f'{waveform.size()=}')
        # print(f'{type(transform)=}')
        # print(f'{device=}')
        # print(f'{waveform.device=}')
        # exit(0)

    # 1e-8は melspectrogram を計算するときにsqrt(0)を計算し，NaNにならないように
    mel_spec = transform(waveform + 1e-8)

    # # STFTで振幅スペクトログラムを得る
    
    # spec = torch.abs(torch.stft(
    #     input=waveform, 
    #     n_fft=n_fft,
    #     hop_length=hop_length,
    #     window=torch.hamming_window(window_n_sample),
    #     return_complex=True
    # ))
    
    # mel_filters = F.melscale_fbanks(
    #     int(n_fft // 2 + 1),
    #     n_mels=n_mels,
    #     f_min=20.0,
    #     f_max=sr // 2.0,
    #     sample_rate=sr,
    #     norm="slaney",
    # )
    # # メルフィルタバンクの準備
    # mel_filterbank = T.MelScale(
    #     n_mels=n_mels, 
    #     f_min=20.0, 
    #     f_max=sr // 2.0, 
    #     sample_rate=sr
    # )

    # # メルスペクトログラムをlogスケールに
    # mel_spec = mel_filterbank(spec)

    log_mel_spec = torch.log(mel_spec)
    # DEBUG
    # print(log_mel_spec.size())
    return log_mel_spec

def pre_emphasis(y, alpha=0.97):
    y_preemphasized = torch.cat((y[:, 0:1], y[:, 1:] - alpha * y[:, :-1]), dim=1)
    return y_preemphasized