import torch
import numpy as np
import soundfile as sf
from preprocess import extract_resized_segments_from_file
import torchvision.transforms as T
from torch import nn
import argparse
import pickle
import os
from tqdm import tqdm
from inference import inference
'''
JTES:DCNN-BLSTMwAのモデルでfilelistの音声ファイルから抽出した4次元serベクトルをOUT_DIRに保存
その他:感情カテゴリのone-hotをOUT_DIRに保存
'''

OUT_DIR = 'dump/servector/studies-calls-jtes'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filelist') # inference対象の音声ファイルのリスト
    parser.add_argument('-m', '--modelpath', default='/work/abelab5/t_tana/emo_clf2/logs/exp3/jtes/cv10/model_241211-190716_final.pth') # inferenceに使用するモデルのパス
    args = parser.parse_args()

    save_dir = os.path.join(OUT_DIR, os.path.basename(args.filelist))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = torch.load(args.modelpath, device)
    wavpath, spk, text, emotion = read_filelists(args.filelist)
    cnt = 0
    for wavpath, spk, text, emotion in tqdm(zip(wavpath, spk, text, emotion)):
        if 'jtes' in wavpath:
            embed, _ = inference(wavpath, model, device, 4.0) # JTESの学習時に使用したデータは4秒でそろえたため
        else:
            embed = torch.nn.functional.one_hot(torch.tensor(emotion), num_classes=4) # 感情カテゴリ数4
        save_path = os.path.join(save_dir, f'{os.path.basename(wavpath)}.pkl')
        save_servector(save_path, embed)
        cnt += 1
        if cnt == 5850:
            print(embed)

def read_filelists(filelist):
    wavpath = []
    spk = []
    text = []
    emotion = []
    with open(filelist, 'r') as f:
        lines = f.readlines()
    for line in lines:
        split = line.split('|')
        wavpath.append(split[0])
        spk.append(int(split[1]))
        text.append(split[2])
        emotion.append(int(split[3]))

    return wavpath, spk, text, emotion

def save_servector(save_path, vector):
    with open(save_path, 'wb') as f:
        pickle.dump(vector, f)

if __name__ == '__main__':
    main()