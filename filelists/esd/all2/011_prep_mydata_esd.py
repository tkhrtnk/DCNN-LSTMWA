import os
import glob
import sys
from typing import Iterable, List, Optional, Union
import random
from tqdm import tqdm

def main():
    ############################################
    # ここを変える
    dst_dirname = 'filelists/esd/all2'        # 出力ディレクトリ
    esd_dirname = 'esd'
    dst_filename = 'esd_audio_sid_text'
    speakers_esd = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
    speaker2id_esd = {'0011':11, '0012':12, '0013':13, '0014':14, '0015':15, '0016':16, '0017':17, '0018':18, '0019':19, '0020':20}
    lang2id = {'ja':0, 'en':1}
    emotions =['Neutral', 'Happy', 'Sad', 'Angry'] # 取り出す感情のリスト（Surpriseは抜いている）
    emo2id = {'Neutral':0, 'Happy':1, 'Sad':2, 'Angry':3, 'Surprise':4} # 感情IDへ変換
    train_test_val_rate = [0.80, 0.10, 0.10] # 合計が1.0になるように配分
    ############################################

    random.seed(1234)
    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'
    assert sum(train_test_val_rate) == 1

    basedir = 'dataset'
    esd_dir = os.path.join(basedir, esd_dirname)
    assert os.path.isdir(esd_dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # 出力するファイルリスト
    filelist = []
    filelist.append([])
    filelist.append([])
    filelist.append([])
            
    # esd に関する処理 ------------------------------------------------------------------------------
    # 感情ごとにファイルを分けたい
    wavpath_list = glob.glob(os.path.join(esd_dir, '**', '*.wav'), recursive=True)
    tr_name = {0:'train', 1:'test', 2:'val'}
    for emotion in emotions:
        # 特定の感情のリスト
        emolist = [f'{wavpath}|None|None|{emo2id[emotion]}|None\n' for wavpath in wavpath_list if emotion in wavpath]
        random.shuffle(emolist)
        start = 0
        for i in range(3): # train, test, val
            filelist[i] += emolist[start:start+int(train_test_val_rate[i]*len(emolist))]
            start += int(train_test_val_rate[i]*len(emolist))
    for i in range(3): # train, test, val
        print(f'{tr_name[i]}: {len(filelist[i])}')
        savename = os.path.join(dst_dirname, f'{dst_filename}_{tr_name[i]}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[i])
        print(f'{savename} Saved')
                
                

if __name__ == '__main__':
    main()
