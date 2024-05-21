# -*- coding:utf-8 -*-
import os
import glob
import torch
import argparse
import librosa
import numpy as np
import polars as pl
from tqdm import tqdm
from datasets.utils import AugmentMelSTFT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default=os.path.join('..', 'data'))
    parser.add_argument('--trg_path', default=os.path.join('..', 'data_spectrum'))
    parser.add_argument('--sfreq', default=32000, type=int)
    parser.add_argument('--window_size', default=800, type=int)
    parser.add_argument('--hop_size', default=320, type=int)
    parser.add_argument('--n_mels', default=256, type=int)
    return parser.parse_args()


def sleep_audio_parser(args):
    sleep_stage_columns = ['Wake', 'NonREM1', 'NonREM2', 'NonREM3', 'REM']  # Sleep Staging
    apnea_columns = ['ObstructiveApnea', 'MixedApnea', 'CentralApnea']      # Apnea
    hypopnea_columns = ['Hypopnea']                                         # Hypopnea
    arousal_columns = ['Arousal']                                           # Arousal
    snore_columns = ['Snore']                                               # Snore

    mel = AugmentMelSTFT(n_mels=args.n_mels, sr=args.sfreq, win_length=args.window_size, hopsize=args.hop_size)
    mel.eval()

    for src_name in os.listdir(args.src_path):
        subject_name = src_name
        series = pl.read_parquet(os.path.join(args.src_path, src_name, 'label.parquet'))
        sleep_stage = series[sleep_stage_columns].to_numpy()
        sleep_stage = sleep_stage.argmax(axis=-1)

        # Sleep Stage Classification (Wake : 0, Light Sleep : 1, Deep Sleep : 2, REM: 3)
        sleep_stage[sleep_stage == 0] = 0       # 1. Wake     => Wake        (0)
        sleep_stage[sleep_stage == 1] = 1       # 2. Non-REM1 => Light Sleep (1)
        sleep_stage[sleep_stage == 2] = 1       # 3. Non-REM2 => Light Sleep (1)
        sleep_stage[sleep_stage == 3] = 2       # 4. Non-REM3 => Deep Sleep  (2)
        sleep_stage[sleep_stage == 4] = 3       # 5. REM      => REM         (3)

        apnea = np.array([0 if np.sum(sample) == 0 else 1 for sample in series[apnea_columns].to_numpy()])
        arousal = series[arousal_columns].to_numpy().squeeze()
        hypopnea = series[hypopnea_columns].to_numpy().squeeze()
        snore = series[snore_columns].to_numpy().squeeze()

        # [Sleep Staging, Apnea, Hypopnea, Arousal, Snore]
        labels = np.stack([sleep_stage, apnea, hypopnea, arousal, snore], axis=-1)

        if not os.path.exists(os.path.join(args.trg_path, subject_name)):
            os.makedirs(os.path.join(args.trg_path, subject_name))

        subject_name = src_name
        audio_paths = glob.glob(os.path.join(args.src_path, src_name, '*.flac'))
        audio_paths.sort()
        for i, path in enumerate(audio_paths):
            # model to preprocess waveform into mel spectrograms
            (waveform, _) = librosa.core.load(path, sr=args.sfreq, mono=True)
            waveform = torch.from_numpy(waveform[None, :]).to('cpu')
            spectrum = mel(waveform).squeeze().numpy()
            label = labels[i, :]
            np.savez(
                os.path.join(args.trg_path, subject_name, '{0:04d}.npz'.format(i + 1)),
                x=spectrum, y=label
            )


if __name__ == '__main__':
    sleep_audio_parser(get_args())
