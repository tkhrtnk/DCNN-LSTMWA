import os
import glob
from typing import Iterable, List, Optional, Union
import random

def main():
    ############################################
    # ここを変える
    studies_dirname = 'studies'
    # validationとtestのディレクトリ名
    dst_dirname = 'filelists/studies/cv100'        # 出力ディレクトリ
    dst_filename = 'studies_audio_sid_text'  # 出力ファイル名（一部）
    # emotion2id = {'平静':0, '喜び':1, '悲しみ':2}  # 感情-->ID
    emotion2id = {'平静':0, '喜び':1, '悲しみ':2, '怒り':3}
    div = 4 # 100を5分割したセグメントのインデックス
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    studies_dir = os.path.join(basedir, studies_dirname)
    assert os.path.isdir(studies_dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # 出力するファイルリスト
    filelist = dict()
    filelist['val'] = list()
    filelist['test'] = list()
    filelist['train'] = list()

    for type_name in ['ITA']:
        type_dir = os.path.join(studies_dir, type_name)
        # Emotion100-Angry, LD01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]
        filelists = dict()  # 一旦保存
        filelists['train'] = []
        filelists['test'] = []
        filelists['val'] = []

        for dname in dir_list:
            d = os.path.join(type_dir, dname)
            # if dname == 'Recitation324':
            #     continue
            txt_files = sorted(glob.glob(os.path.join(d, 'txt/*.txt'), recursive=True))
            wav_files = sorted(glob.glob(os.path.join(d, 'wav/*Teacher*.wav'), recursive=True))
            print(d)
            for txt_file in txt_files:
                with open(txt_file, mode='r', encoding='utf-8') as f:
                    lines = f.readlines()
                lines = [s for s in lines if s.split('|')[0]=='講師']
                wav_dict = dict(zip(lines, wav_files))
                j = 0
                random.seed(123)
                random.shuffle(lines)
                for line in lines:
                    if j < div*20 or j >= 20*(div+1):
                        tr_name = 'train'
                    else:
                        tr_name = 'val'
                    emotion = line.split('|')[1]  # 感情
                    eid = emotion2id[emotion]
                    
                    # # 怒り（Angry）は除外
                    # if emotion == '怒り':
                    #     i+=1
                    #     continue
                    
                    text = line.split('|',)[2]  # 平文
                    text = text.replace('\u3000', '')  # 全角スペースを削除
                    filepath = wav_dict[line]
                    newline = f'{filepath}|spk|{text}|{eid}|lang\n'
                    filelists[tr_name].append(newline)
                    j+=1

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelists[t])

if __name__ == '__main__':
    main()
