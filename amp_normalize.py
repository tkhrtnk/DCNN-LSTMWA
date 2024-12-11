import librosa
import soundfile as sf
import argparse
import glob
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', help='dir_path (dir has wav files you want to normalize)')
    parser.add_argument('dst_dir')
    args = parser.parse_args()
    normalize_wavs(args.src_dir, args.dst_dir)

def normalize_wavs(dir_path, save_path):
    # ディレクトリをまずコピーして，以降の操作をコピー先で行う
    if dir_path != save_path:
        # print('Dir copy')
        shutil.copytree(dir_path, save_path)
    # 中間ディレクトリも含めてsave_path以下のwavファイルのパスのリスト取得
    wavlist = glob.glob(f'{save_path}/**/*.wav', recursive=True)
    # print('(2/2) normalization')
    for wav in wavlist:
        normalize(wav)

def normalize(wav_path):
    # 音声ファイルを読み込む
    y, sr = librosa.load(wav_path, sr=None)

    # 音声のAmplitudeが0から1になるように正規化
    y_normalized = librosa.util.normalize(y)

    # 正規化された音声を保存する
    sf.write(wav_path, y_normalized, sr)

if __name__ == '__main__':
    main()