import os
import glob
import sys
from typing import Iterable, List, Optional, Union
from tqdm import tqdm

def main():
    ############################################
    # ここを変える
    dst_dirname = 'filelists/jtes/all'        # 出力ディレクトリ
    dirname = 'jtes'
    dst_filename = 'jtes_audio_sid_text'
    speakers = [f'f{i:02d}' for i in range(1, 51)] + [f'm{i:02d}' for i in range(1, 51)]
    # valが何もないと学習できないので，全データを学習したい場合，手作業で適当にtrainのファイルリストからコピーしておく
    VAL_DIRS = []
    TEST_DIRS = []
    TRAIN_DIRS = speakers
    speaker2id = dict(zip(speakers, range(101)))
    print(speaker2id)
    lang2id = {'ja':0, 'en':1}
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    dir = os.path.join(basedir, dirname, 'wav')
    assert os.path.isdir(dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # 出力するファイルリスト
    filelist = dict()
    filelist['val'] = []
    filelist['test'] = []
    filelist['train'] = []        
            
    for spk in speakers:
        dname = spk
        if dname in VAL_DIRS:
            tr_name = 'val'
        elif dname in TEST_DIRS:
            tr_name = 'test'
        elif dname in TRAIN_DIRS:
            tr_name = 'train'
        else:
            continue
        
        for emotion in ['neu', 'joy', 'sad', 'ang']:
            d = os.path.join(dir, spk, emotion)
            wavlist = os.listdir(d)

            for wav in wavlist:
                filepath = os.path.join(d, wav)
                # text = wav.split('_')[2]  # 文ID
                text = 'None'
                emo2id = {'neu':0, 'joy':1, 'sad':2, 'ang':3, 'Surprise':4} # 感情IDへ変換
                eid = emo2id[emotion]
                # datasetへのパス|話者id|文ID|感情ID|言語ID
                newline = f"{filepath}|{speaker2id[spk]}|{text}|{eid}|{lang2id['ja']}\n"   
                filelist[tr_name].append(newline)

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])
        print(f'{savename} Saved')        

if __name__ == '__main__':
    main()
