import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from data_utils import EmotionDatasetRaw
from torch.optim import lr_scheduler
from utils import get_hparams_from_file, save_result_figure, save_result_pkl, save_model
from datetime import datetime
from models import DNN
from modules import ExtractXvector
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

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
    # 訓練データ
    training_data = EmotionDatasetRaw(train='train', hparams=hparams, device=device, 
                               transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    train_dataloader = DataLoader(training_data, batch_size=hparams.train.batch_size, generator=generator, shuffle=True, drop_last=False)
    # 検証データ
    validation_data = EmotionDatasetRaw(train='val', hparams=hparams, device=device, 
                           transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    val_dataloader = DataLoader(validation_data, batch_size=hparams.train.batch_size, generator=generator, shuffle=True, drop_last=False)
    # テストデータ
    if hparams.data.test_files == 0: # 0のときはvalidationデータをtestに使用する
        test_data = EmotionDatasetRaw(train='val', hparams=hparams, device=device, 
                           transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    else:
        test_data = EmotionDatasetRaw(train='test', hparams=hparams, device=device, 
                            transform=T.Normalize(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    test_dataloader = DataLoader(test_data, batch_size=hparams.train.batch_size, generator=generator, shuffle=True, drop_last=False)

    print(hparams)

    # 特徴抽出器をロードする
    extractor = ExtractXvector()
    
    # モデルの構築
    model = DNN(num_classes=hparams.data.num_classes, dropout=hparams.model.dropout)
    model.to(device)

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

    checkpoint = hparams.train.epochs / 4
    checkpoint_list = []
    while checkpoint < hparams.train.epochs:
        checkpoint_list.append(int(checkpoint))
        checkpoint += hparams.train.epochs / 4

    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    modelpath_list = []

    # 学習・検証フェーズ
    for epoch in range(hparams.train.epochs):
        print(f'Epoch {epoch+1}\n----------------------------------------------------------------')
        train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, writer, epoch, extractor, hparams.data.sample_rate)
        val_acc, val_loss = val_loop(val_dataloader, model, loss_fn, writer, epoch, extractor, hparams.data.sample_rate)
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
        loss, correct, cm = test_loop(test_dataloader, model, loss_fn, extractor, hparams.data.sample_rate)
        # 結果を画像として保存
        save_result_figure(loss, correct, cm, test_modelpath, f'{os.path.split(os.path.basename(args.config_filepath))[0]}_{i}')
        save_result_pkl(loss, correct, cm, f'loss_acc_cm_{os.path.split(os.path.basename(args.config_filepath))[0]}_{i}')

    writer.close()

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, writer, epoch, extractor, sr):
    size = len(dataloader.dataset)
    for batch, (X, y, _) in enumerate(dataloader):
        # X.size() -> [batch_size, seq_len, C, H, W]
        # extractor(X).size() -> [batch_size, seq_len, segment_level_feature_dim]
        # pred.size() -> [batch_size, num_classes]
        # y.size() -> [batch_size]
        for idx in X.size(0):
            feature[idx] = extractor(X[idx].cpu(), sr)
        
        pred = modelf(feature)
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

def val_loop(dataloader, model, loss_fn, writer, epoch, extractor, sr):
    loss_all, correct_all = 0, 0
    size = len(dataloader.dataset)
    with torch.no_grad():
        for batch, (X, y, _) in enumerate(dataloader):
            loss_batch, correct_batch = 0, 0        
            pred = model(extractor(X, sr))
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

def test_loop(dataloader, model, loss_fn, extractor, sr):
    loss_all, correct_all = 0, 0
    size = len(dataloader.dataset)
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch, (X, y, _) in enumerate(dataloader):        
            pred = model(extractor(X, sr))
            loss_all += loss_fn(pred, y).sum().item() / X.size(0)
            correct_all += (pred.argmax(1) == y).type(torch.float).sum().item() / X.size(0)
            y_pred += pred.argmax(1).tolist()
            y_true += y.tolist()
        loss_all /= len(dataloader) # 損失平均
        correct_all /= len(dataloader) # 正解率
        cm = confusion_matrix(y_pred, y_true) # 混同行列
        print(f'Test({size=}): \n Accuracy: {(100*correct_all):>0.1f}%, Avg loss: {loss_all:>8f} \n')
        return loss_all, correct_all, cm 

if __name__ == "__main__":
    main()