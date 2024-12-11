import json
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle as pkl

def pad_and_trim_sequence(sequences, max_len):
    """
    シーケンスデータを指定した長さに切り取り、短い場合はゼロパディングする関数。

    Args:
        sequences (list of torch.Tensor): シーケンスデータのリスト
        max_len (int): 指定した長さ

    Returns:
        torch.Tensor: パディングおよび切り取り後のシーケンスデータ
    """
    # シーケンスを指定した長さに切り取り、短い場合はゼロパディング
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seq = seq[:max_len]
        else:
            padded_seq = torch.cat([seq, torch.zeros(max_len - len(seq))])
        padded_sequences.append(padded_seq)
    
    # パディングされたシーケンスをテンソルに変換
    padded_sequences = torch.stack(padded_sequences)
    return padded_sequences

def load_result_pkl(filepath):
    with open(filepath, 'rb') as f:
       loss, acc, cm = pkl.load(f)
    print(f'Load: {filepath}')
    return loss, acc, cm

def normalize_cm(cm, normalize):
    if normalize == 'pred':
        # 混同行列をpred_labelで正規化
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    elif normalize == 'true':
        # 混同行列をtrue_labelで正規化
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm

def save_result_figure(loss, acc, cm, modelpath, figtitle='result_figure_example'):
    """
    混同行列を画像として保存し、キャプションとして平均ロス、平均精度、モデルの保存パスを追加する関数。

    Args:
        loss (float): 平均ロス
        acc (float): 平均精度
        cm (numpy.ndarray): 正規化前の混同行列
        modelpath (str): 実験したモデルの保存パス
        figtitle (str): 画像の保存ファイルタイトル（デフォルトは 'result_figure_example'）

    Returns:
        None
    """


    # ラベルIDと意味の関係 
    labels = {0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry'}
    label_names = [labels[i] for i in range(len(labels))]
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalize_cm(cm, 'true'), annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # キャプションを追加
    caption = f'Avg_loss: {loss:.4f}, Avg_acc: {acc*100:.2f}%, Model_path: {modelpath}'
    plt.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=12)

    # 画像として保存
    savepath = os.path.join('resources', f'{figtitle}.png')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    print(f"Save: \n {savepath}")


def save_result_pkl(loss, acc, cm, savetitle='loss_acc_cm_example'):
    savepath = os.path.join('dump', f'{savetitle}.pkl')
    with open(savepath, "wb") as f:
        pkl.dump((loss, acc, cm), f)
        print(f'Save: \n {savepath}')


def save_model(save_dir, model, time, checkpoint='final'):
    path = os.path.join(save_dir, f'model_{time}_{checkpoint}.pth')
    torch.save(model, path)
    if checkpoint != 'final':
        print(f'Save[checkpoint(Epoch {checkpoint})]: \n {path}\n')
    else:
        print(f'Save[final]: \n {path}\n')
    return path

def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)
  hparams = HParams(**config)
  return hparams

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()