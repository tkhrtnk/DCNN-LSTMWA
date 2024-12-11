from utils import load_result_pkl, save_result_figure, normalize_cm
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    files = [file for file in os.listdir(f'dump') if args.name in file]
    if files == []:
        return
    _, _, cm = load_result_pkl(os.path.join('dump', files[0]))
    avg_loss, avg_acc, avg_cm = 0, 0, np.zeros_like(cm)
    for file in files:
        loss, acc, cm = load_result_pkl(os.path.join('dump', file))
        avg_loss += loss
        avg_acc += acc
        avg_cm += cm
    avg_loss /= len(files)
    avg_acc /= len(files)
    avg_cm = normalize_cm(cm, 'true')
    save_result_figure(avg_loss, avg_acc, avg_cm, args.name, args.name)

if __name__ == '__main__':
    main()