import os
import numpy as np
import pandas as pd
import librosa


def get_stats(f: np.ndarray):
    return [np.min(f), np.max(f), np.mean(f), np.std(f)]


def get_features(fp: str, class_name: str):
    y, sr = librosa.load(fp, sr=None)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # filepath, class_name, duration, feature_vec_len, [cen, rolloff, bw, zcr] (min, max, avg, std)
    duration = y.shape[0] / sr
    feature_len = spectral_centroid.shape[1]
    features = [fp,
                class_name,
                duration,
                feature_len,
                *get_stats(spectral_centroid),
                *get_stats(spectral_rolloff),
                *get_stats(spectral_bandwidth),
                *get_stats(zcr)]

    # print(spectral_centroid.shape)
    # print(spectral_rolloff.shape)
    # print(spectral_bandwidth.shape)
    # print(zcr.shape)
    # print(mfcc.shape)
    # print(chroma.shape

    return features

def main():
    data_path = "UrbanSound8K"
    meta_df = pd.read_csv(os.path.join(data_path, "UrbanSound8K.csv"))
    data = []
    class_to_idx = {}
    cnt = 0
    for _, row in meta_df.iterrows():
        if row['class'] not in class_to_idx:
            class_to_idx[row['class']] = len(class_to_idx)
        dirpath = os.path.join(data_path, f"fold{row['fold']}")
        fp = os.path.join(dirpath, row['slice_file_name'])

        features = get_features(fp, row['class'])
        data.append(features)
        print(cnt)
        cnt += 1

    feature_names = ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate']
    atr_types = ['min', 'max', 'mean', 'std']

    cols = ['filepath', 'class', 'duration', 'feature_vec_length']
    for feature_name in feature_names:
        for atr_type in atr_types:
            cols.append(f"{feature_name}_{atr_type}")

    df = pd.DataFrame(data, columns=cols)

    df.to_csv('statistics.csv', index=False)


if __name__ == '__main__':
    main()
