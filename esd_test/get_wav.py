import shutil
import os

def main():
    # ファイルパスが書かれたテキストファイル
    file_list_path = '/work/abelab5/t_tana/vits/filelists/esd_test/spk19_test30_filelist.txt_shuffle.txt'

    # コピー先のフォルダ
    dst_dir = '/work/abelab5/t_tana/vits/wav/dubbing/esd_reference'

    # コピー先フォルダが存在しない場合は作成
    os.makedirs(dst_dir, exist_ok=True)

    try:
        # ファイルリストを開いて処理
        with open(file_list_path, 'r') as file:
            for i, line in enumerate(file):
                # 改行を取り除いてファイルパスを取得
                src_path = line.strip().split('|')[0]
                emo = line.strip().split('|')[3]

                if os.path.isfile(src_path):
                    # コピー先のパスを決定
                    base_name = os.path.splitext(os.path.basename(src_path))[0]
                    dst_path = os.path.join(dst_dir, f'{i:03}-{base_name}-emo{emo}.wav')
                    # ファイルをコピー
                    shutil.copy2(src_path, dst_path)
                    print(f'Copied: {src_path} to {dst_path}')
                else:
                    print(f'File not found: {src_path}')
    except Exception as e:
        print(f'Error: {e}')

if __name__=='__main__':
    main()