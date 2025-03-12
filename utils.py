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

def _normalize_cm(cm, normalize):
    if normalize == 'true':
        # 混同行列をpred_labelで正規化
        # 行ごとに総和を計算 
        row_sums = cm.sum(axis=1) 
        # 総和がゼロの行に対する処理
        row_sums[row_sums == 0] = 1  
        # 例えば総和がゼロの行に1を代入してゼロ除算を防ぐ
        cm = cm.astype('float') / row_sums[:, np.newaxis]
    elif normalize == 'pred':
        # 混同行列をtrue_labelで正規化
        # 行ごとに総和を計算 
        row_sums = cm.sum(axis=0) 
        # 総和がゼロの行に対する処理
        row_sums[row_sums == 0] = 1 
        # 例えば総和がゼロの行に1を代入してゼロ除算を防ぐ
        cm = cm.astype('float') / row_sums[:, np.newaxis]
    return cm

def save_result_figure(loss, acc, cm, modelpath, figtitle='result_figure_example', id2emo={0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Angry'}):
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

    label_names = [id2emo[i] for i in range(len(id2emo))]
    plt.figure(figsize=(10, 8))
    sns.heatmap(_normalize_cm(cm, 'true'), annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
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
  
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss