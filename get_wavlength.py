import wave
'''
wavファイル単体あるいはfilelistのファイルを与えることで
全てのwavファイルの時間を分あるいは時間単位で表示
論文の実験条件のデータのサイズについて書くときに使用した
'''

def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def get_total_duration(file_list_path):
    total_duration = 0.0
    with open(file_list_path, 'r') as file:
        for line in file:
            file_path = line.strip().split('|')[0]
            if file_path.endswith('.wav'):
                total_duration += get_wav_duration(file_path)
    return total_duration / 60  # 秒を分に変換


def main():
    file_list_paths = [
        '/work/abelab5/t_tana/emo_clf2/filelists/esd/all/esd_audio_sid_text_test_filelist.txt',
        '/work/abelab5/t_tana/emo_clf2/filelists/esd/all/esd_audio_sid_text_train_filelist.txt',
        '/work/abelab5/t_tana/emo_clf2/filelists/esd/all/esd_audio_sid_text_val_filelist.txt'
    ]
    total_duration_minutes = 0
    for file_list_path in file_list_paths:
        total_duration_minutes += get_total_duration(file_list_path)
    
    print(f"Total duration: {total_duration_minutes:.2f} minutes")
    print(f"Total duration: {total_duration_minutes/60:.2f} hours")

if __name__ == '__main__':
    main()