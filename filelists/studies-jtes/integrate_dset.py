import os 

assert os.getcwd() == '/work/abelab5/t_tana/emo_clf2/filelists'
dst_filename = 'studies-jtes_audio_sid_text'  # 出力ファイル名（一部）


# 作成したいディレクトリのパス 
dir_list = ['studies-jtes/cv1', 'studies-jtes/cv2', 'studies-jtes/cv3', 'studies-jtes/cv4', 'studies-jtes/cv5']

# 統合元のデータが入っているパス
studies_list = ['studies/cv20', 'studies/cv40', 'studies/cv60', 'studies/cv80', 'studies/cv100']
jtes_list = ['jtes/cv10', 'jtes/cv20', 'jtes/cv30', 'jtes/cv40', 'jtes/cv50']

comb_list = [list(pair) for pair in zip(dir_list, studies_list, jtes_list)]

for pair in comb_list: 
    # ディレクトリが存在しない場合に作成 
    if not os.path.exists(pair[0]):
        os.makedirs(pair[0])
    
    studies_cv = [os.path.join(pair[1], file) for file in os.listdir(pair[1])]
    jtes_cv = [os.path.join(pair[2], file) for file in os.listdir(pair[2])]

    for tr_name in ['train', 'val', 'test']:
        tr_studies = next((item for item in studies_cv if tr_name in item), None)
        tr_jtes = next((item for item in jtes_cv if tr_name in item), None)
        
        with open(tr_studies, 'r') as f:
            lines = f.readlines()

        with open(tr_jtes, 'r') as f:
            lines += f.readlines()

        savepath = f'{pair[0]}/{dst_filename}_{tr_name}_filelist.txt'
        with open(savepath, 'w') as f:
            f.writelines(lines)