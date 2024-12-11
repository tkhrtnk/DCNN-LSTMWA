import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle as pkl
from preprocess import extract_resized_segments_from_file
import soundfile as sf
from utils import pad_and_trim_sequence

class EmotionDataset(Dataset):
    def __init__(self, train, hparams, device, transform=None):
        if train == 'train':
            filelist_path = hparams.data.training_files
        elif train == 'test':
            filelist_path = hparams.data.test_files
        elif train == 'val':    
            filelist_path = hparams.data.val_files
        else:
            assert False, 'tr_name is invalid'
        df = pd.read_csv(filelist_path, sep='|', names=['wavpath', 'speaker_id', 'transcript', 'emotion', 'language'])            
        self.wavpaths_emotions = list(zip(df.wavpath.to_list(), df.emotion.to_list()))
        self.segments_dict = dict()
        self.device = device
        self.transform = transform
        self.max_sec = hparams.data.max_sec

    def __len__(self):
        return len(self.wavpaths_emotions)
    
    def __getitem__(self, idx):
        wavpath, emo_label = self.wavpaths_emotions[idx]
        # 初回のみ処理し，以降はデータが保存されている辞書を参照する
        # if wavpath not in self.segments_dict.keys():
        #     self.segments_dict[wavpath] = extract_resized_segments_from_file(wavpath, device=self.device)
        # segments = self.segments_dict[wavpath]
        # ↑の処理だと辞書に保存していくうちにメモリが足りなくなるので逐次変換することにした
        segments = extract_resized_segments_from_file(wavpath, device=self.device, normalizer=self.transform, max_sec=self.max_sec)
        return segments.to(self.device), torch.tensor(emo_label, device=self.device), wavpath

class EmotionDatasetRaw(Dataset):
    def __init__(self, train, hparams, device, transform=None):
        if train == 'train':
            filelist_path = hparams.data.training_files
        elif train == 'test':
            filelist_path = hparams.data.test_files
        elif train == 'val':    
            filelist_path = hparams.data.val_files
        else:
            assert False, 'tr_name is invalid'
        df = pd.read_csv(filelist_path, sep='|', names=['wavpath', 'speaker_id', 'transcript', 'emotion', 'language'])            
        self.wavpaths_emotions = list(zip(df.wavpath.to_list(), df.emotion.to_list()))
        self.segments_dict = dict()
        self.device = device
        self.transform = transform
        self.max_sec = hparams.data.max_sec

    def __len__(self):
        return len(self.wavpaths_emotions)
    
    def __getitem__(self, idx):
        wavpath, emo_label = self.wavpaths_emotions[idx]
        wav, sr = sf.read(wavpath)
        wav = pad_and_trim_sequence(torch.from_numpy(wav).unsqueeze(0).to(self.device), self.max_sec * sr + 1)
        return wav, torch.tensor(emo_label, device=self.device), wavpath