# data/build_pairs.py
import glob
import os
import random


def build_auto_pairs(folder: str, shuffle: bool = True):
    """
    Given a folder with raw .wav files, automatically create (A, B) song pairs.
    输入只有音乐文件，输出 (A_path, B_path) 列表。
    """
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if shuffle:
        random.shuffle(files)

    pairs = []
    for i in range(len(files) - 1):
        A = files[i]
        B = files[i + 1]
        pairs.append((A, B))

    return pairs
