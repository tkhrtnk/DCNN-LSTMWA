import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import argparse
from data_utils import EmotionDataset
from utils import get_hparams_from_file, save_result_figure, save_model, save_result_pkl
from datetime import datetime
import pickle as pkl
from models import DCNN
from torchsummary import summary
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from hparams_changer import change_hparam
from sklearn.metrics import confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filepath")
    # autowrite: 学習したモデルのうち，最後のモデルのパスを自動でwrite_config_filepathのjsonファイルに書き込む
    parser.add_argument("-a", "--autowrite", action='store_true')
    parser.add_argument('-w', '--write_config_filepath', default=None)
    args = parser.parse_args()
    hparams = get_hparams_from_file(args.config_filepath)
    if args.autowrite:
        assert os.path.exists(args.write_config_filepath)
    
    start_time = datetime.now().strftime("%y%m%d-%H%M%S")
    print(f'{start_time=}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    torch.set_default_device(device)

    writer = SummaryWriter(log_dir=hparams.train.save_dir)
    generator = torch.Generator(device=device).manual_seed(hparams.train.randseed)
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

    print(hparams)

    # ImageNet で事前学習した AlexNet の重みをロード(DCNN)
    model = DCNN(hparams.data.num_classes, hparams.model.dropout, hparams.model.pretrained)
    model.to(device)

    # 確認
    summary(model, (3, 227, 227))
    # print(f'{list(model.parameters())[0]=}')

    # # DEBUG
    # print(model.classifier)
    # exit(0)

    sys.stdout.flush()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.train.learning_rate)
    # LAMBDA LR
    lambda1 = lambda epoch: hparams.train.lr_decay** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    acc_record = []
    loss_record =[]

    checkpoint = hparams.train.epochs / 2
    checkpoint_list = []
    while checkpoint < hparams.train.epochs:
        checkpoint_list.append(int(checkpoint))
        checkpoint += hparams.train.epochs / 2

    modelpath_list = []

    # 注意）バッチサイズの単位はファイルの数であるから，ここから音声ファイルをseq_len個のセグメントに分けるため，
    # batch_size * seq_len 個の画像で1つのミニバッチになり，この単位で重みの更新が行われる．

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
    path = os.path.join(hparams.train.save_dir, f'acc_loss_{start_time}.out')
    with open(path, "wb") as f:
        pkl.dump((acc_record, loss_record), f)
    print(f'Save: \n {path}')
    if args.autowrite:
        change_hparam(name='extractor_path', value=modelpath_list[-1], config=args.write_config_filepath)
    sys.stdout.flush()

    # テストフェーズ
    for i, test_modelpath in enumerate(modelpath_list):
        print(f'Test {test_modelpath}\n----------------------------------------------------------------')
        # モデルのロード
        model = torch.load(test_modelpath)
        model.to(device)
        loss, correct, cm = test_loop(test_dataloader, model, loss_fn)
        # 結果を画像として保存
        save_result_figure(loss, correct, cm, test_modelpath, f'{os.path.basename(args.config_filepath).split(sep=".")[0]}_{i}')
        save_result_pkl(loss, correct, cm, f'loss_acc_cm_{os.path.basename(args.config_filepath).split(sep=".")[0]}_{i}')
    writer.close()

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, writer, epoch):
    size = len(dataloader.dataset) * next(iter(dataloader))[0].size(1)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'{current_lr=}')
    for batch, (X, y, _) in enumerate(dataloader):        
        # Xは1発話から取り出したN個のセグメントの集合
        # X.size() -> [batch_size, N, 3, 277, 277]
        # 1発話のセグメントをN回連続で学習
        loss_batch = torch.zeros(X.size(0))
        seq_len = X.size(1)
        for segment_idx in range(seq_len):
            X_segment = X[:, segment_idx, :, :]
            pred = model(X_segment)
            loss = loss_fn(pred, y)
            loss_batch += loss
            # バックプロパゲーションステップ
            # モデルパラメータの勾配をリセット
            optimizer.zero_grad()
            # バックプロパゲーション実行
            loss.backward()
            # 各パラメータの勾配を使用してパラメータの値を調整
            optimizer.step()
        # 1 seqでの平均
        loss_batch /= seq_len
        loss_batch = loss_batch.sum().item() / X.size(0)
        writer.add_scalar('Loss/train_finetune', loss_batch, (epoch * len(dataloader) + batch))
        
        # 1 epoch の学習の中で loss の log を表示する回数
        divide = 3
        if batch + 1 in [int(len(dataloader.dataset) / dataloader.batch_size * (i + 1) / divide) - 1 for i in range(divide)]: 
            current = (batch + 1) * dataloader.batch_size * seq_len if len(X) == dataloader.batch_size else size
            print(f"loss: {loss_batch:>7f} [{current:>5d}/{size:>5d}(segments)]")
    # 学習率スケジューラを更新
    scheduler.step()

def val_loop(dataloader, model, loss_fn, writer, epoch):
    size = len(dataloader.dataset) * next(iter(dataloader))[0].size(1)
    loss_all = 0
    correct_all = 0
    with torch.no_grad():
        for batch, (X, y, _) in enumerate(dataloader):
            loss_batch = torch.zeros(X.size(0))
            correct_batch = torch.zeros(X.size(0))

            seq_len = X.size(1)
            for segment_idx in range(seq_len):
                X_segment = X[:, segment_idx, :, :]
                pred = model(X_segment)
                loss_batch += loss_fn(pred, y)
                correct_batch += (pred.argmax(1) == y).type(torch.float)
            loss_batch /= seq_len
            loss_batch = loss_batch.sum().item() / X.size(0)
            loss_all += loss_batch
            correct_batch /= seq_len
            correct_batch = correct_batch.sum().item() / X.size(0)
            correct_all += correct_batch
            # 1データあたりの平均
            writer.add_scalar('Loss/val_finetune', loss_batch, (epoch * len(dataloader) + batch))
            writer.add_scalar('Acc/val_finetune', correct_batch, (epoch * len(dataloader) + batch))
        loss_all /= len(dataloader)
        correct_all /= len(dataloader)
        # ミニバッチ1つ分の平均
        print(f'Val({size=}): \n Accuracy: {(100*correct_all):>0.1f}%, Avg loss: {loss_all:>8f} \n')
        return correct_all, loss_all

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset) * next(iter(dataloader))[0].size(1)
    loss_all = 0
    correct_all = 0
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch, (X, y, _) in enumerate(dataloader):
            loss_batch = torch.zeros(X.size(0))
            correct_batch = torch.zeros(X.size(0))

            seq_len = X.size(1)
            for segment_idx in range(seq_len):
                X_segment = X[:, segment_idx, :, :]
                pred = model(X_segment)
                loss_batch += loss_fn(pred, y)
                correct_batch += (pred.argmax(1) == y).type(torch.float)
                y_pred += pred.argmax(1).tolist()
            y_true += torch.repeat_interleave(y, seq_len).tolist()
            loss_batch /= seq_len
            loss_batch = loss_batch.sum().item() / X.size(0)
            loss_all += loss_batch
            correct_batch /= seq_len
            correct_batch = correct_batch.sum().item() / X.size(0)
            correct_all += correct_batch
        loss_all /= len(dataloader)
        correct_all /= len(dataloader)
        cm = confusion_matrix(y_pred, y_true) # 混同行列
        # ミニバッチ1つ分の平均
        print(f'Test({size=}): \n Accuracy: {(100*correct_all):>0.1f}%, Avg loss: {loss_all:>8f} \n')
        return loss_all, correct_all, cm

if __name__ == "__main__":
    main()