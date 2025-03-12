import os
import pickle as pkl
import argparse
import glob
from tqdm import tqdm
import librosa
import torch
import numpy as np
from mel_processing import get_3dmelspec_from_file
import torchvision.transforms.functional as F
import torchvision.transforms as T

def main():
    # DEBUG
    # filepath = 'dataset/esd/0012/Neutral/0012_000002.wav'
    # mel_spec_3d = get_3dmelspec_from_file(filepath=filepath, max_sec=3.0, pre_emph=True, fft_sec=25e-3, hop_sec=10e-3, window_sec=25e-3, n_mels=64)
    # print(f'{mel_spec_3d.size()=}')
    # segments = segmentation(mel_spec_3d, 64, 32)
    # print(f"{segments.size()=}")
    # resized_segments = resize_segments(segments, (227, 227))
    # print(f'{resized_segments.size()=}')

    # emo_clf2直下のみで実行可能
    assert os.path.basename(os.getcwd()) == 'emo_clf2'
    parser = argparse.ArgumentParser()
    parser.add_argument('filelist_dir')
    args = parser.parse_args()

    filelist_paths = glob.glob(args.filelist_dir + '/*')
    for filelist_path in filelist_paths:
        print(filelist_path)
        dict = extract_wavpath_segments_from_filelist(filelist_path)

        # DEBUG
        # print(dict['dataset/esd/0012/Neutral/0012_000001.wav'].size())
        # exit(0)

        savepath = os.path.join('dump/segments', os.path.basename(args.filelist_dir), f"{os.path.basename(filelist_path)}.pkl")
        save_dict(dict, savepath)


def extract_wavpath_segments_from_filelist(filelistpath):
    # key:wavpath, value:segmentsの辞書を返す
    with open(filelistpath, 'r') as f:
        lines = f.readlines()
    dict = {}
    for line in tqdm(lines):
        wavpath = line.split(sep='|')[0]
        segments = extract_resized_segments_from_file(wavpath)
        dict[wavpath] = segments
    return dict

def load_dict(filepath):
    with open(filepath, mode='rb') as f:
        dict = pkl.load(f)
    return dict

def save_dict(dict, savepath):
    with open(savepath, mode='wb') as f:
        pkl.dump(dict, f)
    print('saved')

def extract_resized_segments_from_file(filepath, device=None, normalizer=None, max_sec=3.0, spec_aug=False):
    # extract melspecs -> normalize 0-1 -> custom normalize (if needed) -> segmentation -> resize each segment
    mel_spec_3d = get_3dmelspec_from_file(filepath=filepath, max_sec=max_sec, pre_emph=True, fft_sec=25e-3, hop_sec=10e-3, window_sec=25e-3, n_mels=64, device=device, spec_aug=spec_aug)
    mel_spec_3d = normalize_image(mel_spec_3d)
    # use custom normalize method
    if normalizer is not None:
        mel_spec_3d = normalizer(mel_spec_3d)
    segments = segmentation(mel_spec_3d, 64, 32)
    resized_segments = resize_segments(segments, (227, 227))
    return resized_segments

def segmentation(mel_spec_3d, frame_size=64, hop_size=32):
    # mel_spec_3d -> tensor([3, 64, 301]) 
    # "3" = static, delta1, delta2
    # "64" =  n_mels
    # "301" = means total frame_size of utterance
    # devide the dim of value 301 into segment 3x64xframe_size
    start = 0
    segments = []
    # DEBUG
    # print(mel_spec_3d.size())
    hop = int((mel_spec_3d.size(2) - frame_size) / hop_size + 1)
    for i in range(hop):
        segment = mel_spec_3d[:,:,start:start + frame_size]
        if segment.size(2) < frame_size:
            break
        # # 後部サイレント部分は非サイレント部分を連続させる
        # if torch.all(segment[2] == segment[2][0, 0]):
        #     segment = segments[-1]
        segments.append(segment)
        start += hop_size
    return torch.stack(segments)

def normalize_image(image):
    # C x H x W (tensor -> tensor)
    # convert pixel value 0-1 range
    normalized_image = torch.empty_like(image)
    # DEBUG
    # normalize
    for c in range(image.size(0)):
        min = image[c].min()
        max = image[c].max()
        normalized_image[c] = (image[c] - min) / (max - min)
    return normalized_image

def resize_segments(segments, size):
    # size is [..., H, W], at least requires 2-dim (H and W)
    # default の bilinear 補完で拡大
    return F.resize(img=segments, size=size, antialias=True)

if __name__ == '__main__':
    main()