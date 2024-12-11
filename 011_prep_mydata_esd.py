import os
import glob
import sys
from typing import Iterable, List, Optional, Union
from tqdm import tqdm

def main():
    ############################################
    # ここを変える
    dst_dirname = 'filelists/esd/loso0020'        # 出力ディレクトリ
    esd_dirname = 'esd'
    dst_filename = 'esd_audio_sid_text'
    TEST_DIRS_ESD = []
    VAL_DIRS_ESD = ['0020']
    TRAIN_DIRS_ESD = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019']
    speakers_esd = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
    speaker2id_esd = {'0011':11, '0012':12, '0013':13, '0014':14, '0015':15, '0016':16, '0017':17, '0018':18, '0019':19, '0020':20}
    lang2id = {'ja':0, 'en':1}
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    esd_dir = os.path.join(basedir, esd_dirname)
    assert os.path.isdir(esd_dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # 出力するファイルリスト
    filelist = dict()
    filelist['val'] = list()
    filelist['test'] = list()
    filelist['train'] = list()        
            
    # esd に関する処理 ------------------------------------------------------------------------------
    for spk in speakers_esd:
        dname = spk
        if dname in VAL_DIRS_ESD:
            tr_name = 'val'
        elif dname in TEST_DIRS_ESD:
            tr_name = 'test'
        elif dname in TRAIN_DIRS_ESD:
            tr_name = 'train'
        else:
            continue

        # 話者ごとに処理
        d = os.path.join(esd_dir, f'{spk}')
        # d = 'dataset/ESD/0011'
        # 話者の台本テキストへのパス
        txt_file = os.path.join(d, f'{spk}.txt') 
        # txt_file = 'dataset/ESD/0011/0011.txt
    
        with open(txt_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            filename = line.split('\t')[0] + '.wav' # 0011_000001.wav などのファイル名
            text = line.split('\t')[1]  # 平文
            emotion = line.split('\t')[2]  # 感情
            emotion = emotion.replace('\n', '')  # 半角スペース，改行を削除
            emo2id = {'Neutral':0, 'Happy':1, 'Sad':2, 'Angry':3, 'Surprise':4} # 感情IDへ変換
            eid = emo2id[emotion]
            if eid == 4:
                continue
            filepath = os.path.join(esd_dir, spk, emotion, filename)
            # datasetへのパス|話者id|平文|感情ID|言語ID
            newline = f"{filepath}|{speaker2id_esd[spk]}|{text}|{eid}|{lang2id['en']}\n"
            filelist[tr_name].append(newline)

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])
        print(f'{savename} Saved')
                
                

if __name__ == '__main__':
    main()
