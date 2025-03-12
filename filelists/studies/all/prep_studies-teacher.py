import os
import glob
from typing import Iterable, List, Optional, Union
import random

def main():
    ############################################
    # ここを変える
    studies_dirname = 'studies'
    # validationとtestのディレクトリ名
    VAL_DIRS = [] #['LD04']
    TEST_DIRS = [] #['LD01', 'LD02', 'LD03', 'SD01', 'SD06', 'SD07', 'SD12']
    dst_dirname = 'filelists/studies/all'        # 出力ディレクトリ
    dst_filename = 'studies_audio_sid_text'  # 出力ファイル名（一部）
    # emotion2id = {'平静':0, '喜び':1, '悲しみ':2}  # 感情-->ID
    emotion2id = {'平静':0, '喜び':1, '悲しみ':2, '怒り':3}
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

    for type_name in ['ITA', 'Long_dialogue', 'Short_dialogue']:
        type_dir = os.path.join(studies_dir, type_name)
        # Emotion100-Angry, LD01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]

        for dname in dir_list:
            if dname in VAL_DIRS:
                tr_name = 'val'
            elif dname in TEST_DIRS:
                tr_name = 'test'
            else:
                tr_name = 'train'
            
            d = os.path.join(type_dir, dname)

            files = list()  # 一旦保存
            txt_files = sorted(glob.glob(os.path.join(d, '**/txt/*.txt'), recursive=True))
            wav_files = sorted(glob.glob(os.path.join(d, f'**/wav/*Teacher*.wav'), recursive=True))
            i = 0
            for txt_file in txt_files:
                with open(txt_file, mode='r', encoding='utf-8') as f:
                    lines = f.readlines()
                lines = [s for s in lines if s.split('|')[0]=='講師']
                for line in lines:
                    emotion = line.split('|')[1]  # 感情
                    eid = emotion2id[emotion]
                    
                    # # 怒り（Angry）は除外
                    # if emotion == '怒り':
                    #     i+=1
                    #     continue
                    
                    text = line.split('|',)[2]  # 平文
                    text = text.replace('\u3000', '')  # 全角スペースを削除
                    filepath = wav_files[i]

                    newline = f'{filepath}|spk|{text}|{eid}|lang\n'
                    files.append(newline)

                    i+=1

            filelist[tr_name].extend(files)
    
    for tr_name in ['val', 'test', 'train']:
        random.shuffle(filelist[tr_name])

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])

if __name__ == '__main__':
    main()
