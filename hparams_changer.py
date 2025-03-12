import re
import os
import argparse
import sys
'''
    python hparams_changer.py [--path] [--hparam] [--value]
    引数をつけるとそのpathのhparamをvalueに変更して終了
    引数をつけないと対話実行モードになる
    config_listのJSONファイルのハイパーパラメータを対話実行モードで一括で変更する
'''

def main():
    
    config_list = [
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0011.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0012.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0013.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0014.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0015.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0016.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0017.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0018.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0019.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_finetune_loso0020.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0011.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0012.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0013.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0014.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0015.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0016.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0017.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0018.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0019.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_train_loso0020.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_finetune_cv10.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_finetune_cv20.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_finetune_cv30.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_finetune_cv40.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_finetune_cv50.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv10.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv20.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv30.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv40.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_train_cv50.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0011.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0012.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0013.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0014.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0015.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0016.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0017.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0018.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0019.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_finetune_loso0020.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0011.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0012.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0013.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0014.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0015.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0016.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0017.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0018.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0019.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/esd_xvector_train_loso0020.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_finetune_cv10.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_finetune_cv20.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_finetune_cv30.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_finetune_cv40.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_finetune_cv50.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_train_cv10.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_train_cv20.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_train_cv30.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_train_cv40.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/jtes_xvector_train_cv50.json'
        # ,
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_finetune_cv20.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_finetune_cv40.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_finetune_cv60.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_finetune_cv80.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_finetune_cv100.json'
        # ,
        # STUDIES -> STUDIES
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv20.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv40.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv60.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv80.json',
        # '/work/abelab5/t_tana/emo_clf2/configs/studies_train_ita_cv100.json'
        # ,
        # # JTES -> STUDIES
        '/work/abelab5/t_tana/emo_clf2/configs/studies_train_jtes_cv20.json',
        '/work/abelab5/t_tana/emo_clf2/configs/studies_train_jtes_cv40.json',
        '/work/abelab5/t_tana/emo_clf2/configs/studies_train_jtes_cv60.json',
        '/work/abelab5/t_tana/emo_clf2/configs/studies_train_jtes_cv80.json',
        '/work/abelab5/t_tana/emo_clf2/configs/studies_train_jtes_cv100.json'
    ]
    ########################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default=None)
    parser.add_argument('-y', '--hyparam', default=None)
    parser.add_argument('-v', '--value', default=None)
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        interactive_exec(config_list)
    else:
        commandline_exec(args.path, args.hyparam, args.value)
        
# コマンドライン引数モード
def commandline_exec(filepath, hparam, value):
     change_hparam(hparam, value, filepath)
    
# 対話実行モード
def interactive_exec(config_list):
    print("<config_list> edit hparams_changer.py to add file")
    for config in config_list:
        print(config)
    print("\n'[name]=[value]' to change hparams, '%g [hparam]' to get hparams, or '%q' to quit")
    while True:
        command = input('> ')
        name, value = read_command(command)
        if name == '%g':
            for config in config_list:
                get_hparam(value, config)
        elif name == '%q':
            exit(0)
        elif name == 0 and value == 0:
            print('Invalid command.')
        else:
            for config in config_list:
                change_hparam(name, value, config)
        
def read_command(command):
    if command == '%q':
        name, value = '%q', 0
    elif command.split(sep=' ')[0] == '%g' and len(command.split(sep=' ')) == 2:
        name, value = '%g', command.split(sep=' ')[1] 
    elif len(command.split(sep='=')) == 2:
        name, value = command.split(sep='=')[0], command.split(sep='=')[1]
        if is_numeric_string(value):
            if '.' in value or '+' in value or '-' in value:
                value = float(value)
            else:
                value = int(value)
    else:
        name, value = 0, 0
    return name, value

# 正規表現パターン: 数字または小数点
def is_numeric_string(s):  
    pattern1 = r'^\d+(\.\d+)?$'
    pattern2 = r"^[+-]?\d+(\.\d+)?[eE][+-]?\d+$"
    return bool(re.match(pattern1, s)) or bool(re.match(pattern2, s))

# 正規表現パターン: カンマとスペース（0個以上連続）および改行 
def ends_with_comma_space_newline(s): 
    pattern = r',\s*\n$' 
    return bool(re.search(pattern, s))

def remove_comma_space_newline(s):
    pattern = r',\s*\n$'
    return re.sub(pattern, '', s)

def change_hparam(name, value, config):
    if not os.path.exists(config):
        print(f'{config} is not exist, so skipped.')
        return
    with open(config, 'r') as f:
        lines = f.readlines()
        print(f'Edit: {config}')
    newlines = []
    name = f'\"{name}\"'
    for line in lines:
        if name in line:
            if isinstance(value, str):
                if '\"' not in value:
                    value = f'\"{value}\"'
                newline = line.split(sep=': ')[0] + ': ' + value
            else:
                newline = line.split(sep=': ')[0] + ': ' + f'{value}'
            if ends_with_comma_space_newline(line):
                newline += ','
                print(f'{name}: ' + line.split(sep=": ")[1].rstrip(',\n') + f' -> {value}')
            else:
                print(f'{name}: ' + line.split(sep=": ")[1].rstrip('\n') + f' -> {value}')
            newline += '\n'
            newlines.append(newline)
        else:
            newlines.append(line)
    with open(config, 'w') as f:
        f.writelines(newlines)

def get_hparam(name, config):
    if not os.path.exists(config):
        print(f'{config} is not exist, so skipped.')
        return
    with open(config, 'r') as f:
        lines = f.readlines()
        print(f'Config: {config}')
    name = f'{name}'
    for line in lines:
        if name in line:
            value = remove_comma_space_newline(line.split(sep=': ')[1])
            print(f'{name}={value}')
            return
    print('No such value.')

if __name__ == '__main__':
    main()
    