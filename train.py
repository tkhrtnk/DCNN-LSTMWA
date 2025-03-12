import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from data_utils import EmotionDataset
from torch.optim import lr_scheduler
from utils import get_hparams_from_file, save_result_figure, save_result_pkl, save_model
from datetime import datetime
from models import LSTMEmoClf
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
from utils import EarlyStopping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath")
    args = parser.parse_args()
    hparams = get_hparams_from_file(args.config_filepath)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    torch.set_default_device(device)

    writer = SummaryWriter(log_dir=hparams.train.save_dir)
    generator = torch.Generator(device=device).manual_seed(hparams.train.randseed)
    random.seed(hparams.train.randseed)
    # 訓練データ
    training_data = EmotionDataset(train='train', hparams=hparams, device=device, 
                               transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    train_dataloader = DataLoader(training_data, batch_size=hparams.train.batch_size, generator=generator, shuffle=True, drop_last=False)
    # 検証データ
    validation_data = EmotionDataset(train='val', hparams=hparams, device=device, 
                           transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    val_dataloader = DataLoader(validation_data, batch_size=hparams.train.batch_size, generator=generator, shuffle=True, drop_last=False)
    # テストデータ
    if hparams.data.test_files == 0: # 0のときはvalidationデータをtestに使用する
        test_data = EmotionDataset(train='val', hparams=hparams, device=device, 
                           transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    else:
        test_data = EmotionDataset(train='test', hparams=hparams, device=device, 
                            transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    test_dataloader = DataLoader(test_data, batch_size=hparams.train.batch_size, generator=generator, shuffle=True, drop_last=False)

    print(f'config: {args.config_filepath}')
    print(hparams)
    seq_len = next(iter(train_dataloader))[0].size(1)
    # モデルの構築
    # 構築済みモデルを読み込む場合
    try:
        print(f'{hparams.model.model_path=}')
        model = torch.load(hparams.model.model_path)
        if model.seq_len != seq_len:
            model.lstm_reinit(seq_len)
    # 学習済みのextractorを使用して新しくモデルを作成する場合
    except AttributeError as e:
        print(f'{hparams.model.extractor_path=}')
        model = LSTMEmoClf(num_classes=hparams.data.num_classes, seq_len=seq_len, dropout=hparams.model.dropout, extractor_path=hparams.model.extractor_path)
    model.to(device)

    # 確認(summaryはRNNだと使用不可)
    # summary(model, input_size=(seq_len, 4096))
    print(f'input_size: ({model.bilstm_stack.num_layers}, {model.bilstm_stack.input_size})')
    print(f'output_size: {model.linear_relu_stack[5].out_features}')
    print(f'model structure: \n{model}')
    # print(f'{list(model.parameters())[0]=}')
    sys.stdout.flush()

    # # DEBUG
    # exit(0)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.train.learning_rate)
    #LAMBDA LR
    lambda1 = lambda epoch: hparams.train.lr_decay** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    acc_record = []
    loss_record =[]

    checkpoint = hparams.train.epochs / 2
    checkpoint_list = []
    while checkpoint < hparams.train.epochs:
        checkpoint_list.append(int(checkpoint))
        checkpoint += hparams.train.epochs / 2

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    modelpath_list = []

    # 学習・検証フェーズ
    for epoch in range(hparams.train.epochs):
        print(f'Epoch {epoch+1}\n----------------------------------------------------------------')
        train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, writer, epoch)
        val_acc, val_loss = val_loop(val_dataloader, model, loss_fn, writer, epoch)
        acc_record.append(val_acc)
        loss_record.append(val_loss)
        if epoch in checkpoint_list:
            # モデルの保存+テスト用モデルの保存パスのリストに追加
            modelpath_list.append(save_model(hparams.train.save_dir, model, start_time, epoch+1))
        sys.stdout.flush()
    modelpath_list.append(save_model(hparams.train.save_dir, model, start_time))
    sys.stdout.flush()

    # テストフェーズ
    for i, test_modelpath in enumerate(modelpath_list):
        print(f'Test {test_modelpath}\n----------------------------------------------------------------')
        # モデルのロード
        model = torch.load(test_modelpath)
        loss, correct, cm = test_loop(test_dataloader, model, loss_fn)
        # 結果を画像として保存
        save_result_figure(loss, correct, cm, test_modelpath, f'{os.path.basename(args.config_filepath).split(sep=".")[0]}_{i}')
        save_result_pkl(loss, correct, cm, f'loss_acc_cm_{os.path.basename(args.config_filepath).split(sep=".")[0]}_{i}')

    writer.close()

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, writer, epoch):
    size = len(dataloader.dataset)
    for batch, (X, y, _) in enumerate(dataloader):
        # X.size() -> [batch_size, seq_len, C, H, W]
        # pred.size() -> [batch_size, num_classes]
        # y.size() -> [batch_size]
        pred = model(X)
        loss = loss_fn(pred, y)
        # ミニバッチ1つ分の平均
        loss_batch = loss.sum().item() / X.size(0)
        # バックプロパゲーションステップ
        # モデルパラメータの勾配をリセット
        optimizer.zero_grad()
        # バックプロパゲーション実行
        loss.backward()
        # 各パラメータの勾配を使用してパラメータの値を調整
        optimizer.step()
        # 記録
        writer.add_scalar('Loss/train', loss_batch, epoch * len(dataloader) + batch)
        
        # 1 epoch の学習の中で loss の log を表示する回数
        divide = 3
        if batch + 1 in [int(len(dataloader.dataset) / dataloader.batch_size * (i + 1) / divide) for i in range(divide)]: 
            current = (batch + 1) * dataloader.batch_size if len(X) == dataloader.batch_size else size
            print(f"loss: {loss_batch:>7f} [{current:>5d}/{size:>5d}(files)]")
    # 学習率スケジューラ更新
    scheduler.step()

def val_loop(dataloader, model, loss_fn, writer, epoch):
    loss_all, correct_all = 0, 0
    size = len(dataloader.dataset)
    with torch.no_grad():
        for batch, (X, y, _) in enumerate(dataloader):
            loss_batch, correct_batch = 0, 0        
            pred = model(X)
            loss_batch = loss_fn(pred, y).sum().item() / X.size(0)
            loss_all += loss_batch
            correct_batch = (pred.argmax(1) == y).type(torch.float).sum().item() / X.size(0)
            correct_all += correct_batch
            # 記録
            writer.add_scalar('Loss/val', loss_batch, epoch * len(dataloader) + batch)
            writer.add_scalar('Acc/val', correct_batch, epoch * len(dataloader) + batch)
        loss_all /= len(dataloader) # 損失平均
        correct_all /= len(dataloader) # 正解率
        print(f'Val({size=}): \n Accuracy: {(100*correct_all):>0.1f}%, Avg loss: {loss_all:>8f} \n')
        return correct_all, loss_all

def test_loop(dataloader, model, loss_fn):
    loss_all, correct_all = 0, 0
    size = len(dataloader.dataset)
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch, (X, y, _) in enumerate(dataloader):        
            pred = model(X)
            loss_all += loss_fn(pred, y).sum().item() / X.size(0)
            correct_all += (pred.argmax(1) == y).type(torch.float).sum().item() / X.size(0)
            y_pred += pred.argmax(1).tolist()
            y_true += y.tolist()

        loss_all /= len(dataloader) # 損失平均
        correct_all /= len(dataloader) # 正解率
        cm = confusion_matrix(y_ture, y_pred) # 混同行列
        # DEBUG
        print(f'{y_pred=}')
        print(f'{y_true=}')
        print(f'{cm=}\n')
        print(f'Test({size=}): \n Accuracy: {(100*correct_all):>0.1f}%, Avg loss: {loss_all:>8f} \n')
        return loss_all, correct_all, cm 

if __name__ == "__main__":
    main()