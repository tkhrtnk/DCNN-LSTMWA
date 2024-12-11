import librosa
import soundfile as sf
import argparse
from glob import glob
import os
from tqdm import tqdm
import shutil

def resample(wav_path_in, target_sr):
    # 音声ファイルを読み込む
    y, sr = librosa.load(wav_path_in, sr=None)
    # リサンプリング
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y_resampled


def save_wav(save_path, wav_data, sr):
    # リサンプリングした音声を保存
    sf.write(save_path, wav_data, sr)


def main():
    # カレントディレクトリが/work/abelab5/t_tanaの場合のみ実行できる
    cd = '/work/abelab5/t_tana'
    assert os.getcwd() == cd, f'type "cd {cd}"'

    parser = argparse.ArgumentParser(description='resampling audio file(s)')
    parser.add_argument('src_dir', help='src dataset directory path')
    parser.add_argument('dst_dir', help='dst save directory path')
    parser.add_argument('sr', help='convert to this sr', type=int)
    parser.add_argument('-s', '--silent', action='store_true', help='omit info(pbar, etc.)')
    args = parser.parse_args()

    assert os.path.exists(args.src_dir)

    src_wav_paths = glob(args.src_dir + '/**/*.wav', recursive=True)
    dst_wav_paths = [wav_path.replace(args.src_dir, args.dst_dir, 1) for wav_path in src_wav_paths]

    if args.silent:
        print(f'{args.src_dir} -> {args.dst_dir} resampling (sr={args.sr})')
        
        wavs = [resample(wav_path, args.sr) for wav_path in src_wav_paths]

        # 書き込み先が存在しないなら，元のデータセットのディレクトリの全ファイルをコピー
        if not os.path.exists(args.dst_dir):
            shutil.copytree(args.src_dir, args.dst_dir)

        for wav_path, wav_data in zip(dst_wav_paths, wavs):
            save_wav(wav_path, wav_data, args.sr)
        print("COMPLETED")

    else:
        wavs = []
        with tqdm(src_wav_paths) as pbar:
            for wav_path in pbar:
                pbar.set_description('step1(resampling)')
                wavs.append(resample(wav_path, args.sr))

        # 書き込み先が存在しないなら，元のデータセットのディレクトリの全ファイルをコピー
        if not os.path.exists(args.dst_dir):
            print(f'copy: {args.src_dir} -> {args.dst_dir}')
            shutil.copytree(args.src_dir, args.dst_dir)

        with tqdm(zip(dst_wav_paths, wavs)) as pbar:
            for wav_path, wav_data in pbar:
                pbar.set_description('step2(write)')
                save_wav(wav_path, wav_data, args.sr)
    

if __name__ == '__main__':
    main()