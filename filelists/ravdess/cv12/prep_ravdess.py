import os
import glob
import sys
from typing import Iterable, List, Optional, Union
from tqdm import tqdm

def main():
    ############################################
    # ここを変える
    dst_dirname = 'filelists/ravdess/cv12'        # 出力ディレクトリ
    dirname = 'ravdess'
    dst_filename = 'ravdess_audio_sid_text'
    speakers = range(1, 25)
    VAL_DIRS = [9, 10, 11, 12]
    TEST_DIRS = []
    TRAIN_DIRS = [spk for spk in speakers if spk not in (VAL_DIRS + TEST_DIRS)]
    lang2id = {'ja':0, 'en':1}
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    dir = os.path.join(basedir, dirname)
    assert os.path.isdir(dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # 出力するファイルリスト
    filelist = dict()
    filelist['val'] = []
    filelist['test'] = []
    filelist['train'] = []        
    
    wavlist = find_files_with_extension(dir, 'wav')
    for wav in wavlist:
        file = os.path.basename(wav).split(sep='.')[0]
        split = file.split(sep='-')
        modal = int(split[0])
        channel = int(split[1])
        emotion = int(split[2])
        emotional_intensity = int(split[3])
        statement = int(split[4])
        repetition = int(split[5])
        actor = int(split[6])

        if emotional_intensity != 1:
            continue
        if actor in VAL_DIRS:
            tr_name = 'val'
        elif actor in TEST_DIRS:
            tr_name = 'test'
        elif actor in TRAIN_DIRS:
            tr_name = 'train'
        else:
            continue
        emoid_convt = {1: 0, 3: 1, 4: 2, 5: 3}
        if emotion not in list(emoid_convt.keys()):
            continue
        # datasetへのパス|話者id|文ID|感情ID|言語ID
        newline = f"{wav}|{actor}|{statement}|{emoid_convt[emotion]}|{lang2id['en']}\n"   
        filelist[tr_name].append(newline)

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])
        print(f'{savename} Saved') 

def find_files_with_extension(directory, extension): 
    file_list = [] 
    for root, dirs, files in os.walk(directory): 
        for file in files: 
            if file.endswith(extension): 
                file_list.append(os.path.join(root, file)) 
    return file_list       

if __name__ == '__main__':
    main()
