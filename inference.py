import torch
from torch import nn
import argparse
from preprocess import extract_resized_segments_from_file
import os
import torchvision.transforms as T
from utils import save_result_figure
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
import re

OUT_DIR = 'dump/servector'
# JTES: /work/abelab5/t_tana/emo_clf2/logs/exp3/jtes/cv10/model_241211-190716_final.pth
# ESD: /work/abelab5/t_tana/emo_clf2/logs/exp3/esd/loso0019/model_241211-201617_final.pth
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath")
    parser.add_argument("-f", "--filelist")
    parser.add_argument("-t", "--fig_title", default='sample')
    args = parser.parse_args()
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    torch.set_default_device(device)
    id2emo = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Angry'}
    if 'esd' in args.modelpath:
        max_sec = 3.0
    elif 'jtes' in args.modelpath or 'studies' in args.modelpath:
        max_sec = 4.0
    else:
        print('max sec is 3.0')
        max_sec = 3.0

    if args.filelist is not None:
        assert os.path.exists(args.filelist)
        print(f'Load filelist: {args.filelist}')
        pred = filelist_mode(args.modelpath, device, max_sec, id2emo, args.filelist, args.fig_title)
    else:
        print('Load filelist: None')
        dialogue_mode(args.modelpath, device, max_sec, id2emo)

# filelistを読み込み入力とするモード
def filelist_mode(modelpath, device, max_sec, id2emo, filelist, fig_title):
    # モデルのロード
    model = torch.load(modelpath, device)
    print(f'Load model: {modelpath}')
    with open(filelist, 'r') as f:
        lines = f.readlines()
    pred_labels, true_labels = [], []

    vector_dict = dict()
    for line in tqdm(lines):
        wavpath = line.split(sep='|')[0]
        true_label = int(line.split(sep='|')[3])
        if 'STUDIES' in wavpath or 'CALLS' in wavpath:
            pred = nn.functional.one_hot(torch.tensor(true_label), num_classes=4)
            label = true_label
        else:
            pred, label = inference(wavpath, model, device, max_sec)
            true_labels.append(true_label)
            pred_labels.append(label)
            pred = pred.squeeze(0)
        vector_dict[wavpath] = pred
    save_servector_dict(os.path.join(OUT_DIR, re.search(r'filelists/(.*)/', filelist), f'{os.path.basename(filelist)}.pkl'), vector_dict)
    assert len(true_labels) == len(pred_labels)
    acc = sum([int(pred_label == true_label) for pred_label, true_label in zip(pred_labels, true_labels)]) / len(pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)
    save_result_figure(0, acc, cm, modelpath, fig_title, id2emo)

def save_servector_dict(save_path, dict):
    with open(save_path, 'wb') as f:
        pickle.dump(dict, f)
    print(f'save: {save_path}')

def save_servector(save_path, vector):
    with open(save_path, 'wb') as f:
        pickle.dump(vector, f)

# 逐次入力するモード
def dialogue_mode(modelpath, device, max_sec, id2emo):
    # モデルのロード
    model = torch.load(modelpath, device)
    print(f'Load model: {modelpath}')
    print(f'model structure: \n{model}')
    print("Input wavpath or 'q' to quit.")
    while True:
        wavpath = input("> ")
        if wavpath == 'q':
            break
        pred, label = inference(wavpath, model, device, 8)
        print(f'prediction:\n {pred}')
        print(f'label(pred): {id2emo[label]}')


def inference(wavpath, model, device, max_sec):
    X = extract_resized_segments_from_file(wavpath, device=device, max_sec=max_sec, 
                                           normalizer=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    # X.unsqueeze(0).size() -> [batch_size(1), seq_len, C, H, W]
    with torch.no_grad():
        pred = model(X.unsqueeze(0))
        prob = nn.Softmax(dim=1)(pred)
    return prob, int(prob.argmax(1))

if __name__ == "__main__":
    main()