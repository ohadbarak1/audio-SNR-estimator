#!/usr/bin/env python
from wav_augment import Wav, BackgroundWav
import numpy as np
import json
import os

if __name__ == '__main__':
    seed = 24601
    np.random.seed(seed)

    json_path = "/home/ohad/DG/par/data_defaults.json"

    with open(json_path, "r") as f:
        augment_params = json.load(f)
    
    fg_wav_files = []
    bg_wav_files = []

    src_dir = augment_params["input"]["source_directory"]
    for d in augment_params["input"]["data_directories"]["speech"]:
        speech_dir = os.path.join (src_dir, d["directory_name"])
        for f in os.listdir(speech_dir):
            if f.endswith(".wav"):
                fg_wav_files.append(os.path.join (speech_dir,f))
    
    for d in augment_params["input"]["data_directories"]["background"]:
        background_dir = os.path.join (src_dir, d["directory_name"])
        for f in os.listdir(background_dir):
            if f.endswith(".wav"):
                bg_wav_files.append(os.path.join (background_dir,f))

    augs = augment_params["input"]["augmentation_defaults"]

    for _ in range(augment_params["input"]["augmentation_number"]):
        fg_idx = np.random.randint(0, len(fg_wav_files))
        bg_idx = np.random.randint(0, len(bg_wav_files))

        bg = BackgroundWav("bkgrnd", bg_wav_files[bg_idx])
        fg = Wav("speech", fg_wav_files[fg_idx], augmentation_samples=augment_params["input"]["augmentation_length"]) 

        aug_wav = fg.augment_data (
            background_wav=bg,
            idx=fg_idx,
            foreground_volume_distribution=augs["foreground_volume_distribution"],
            foreground_volume_domain=augs["foreground_volume_domain"],
            foreground_volume_max=augs["foreground_volume_max"],
            foreground_volume_min=augs["foreground_volume_min"],
            foreground_volume_norm=augs["foreground_volume_min"],
            snr_distribution=augs["snr_distribution"],
            snr_min=augs["snr_min"],
            snr_max=augs["snr_max"]

        )

        aug_wav_path = os.path.join(augment_params["input"]["target_directory"], aug_wav.filename)
        aug_wav.save_wav_file (file_path=aug_wav_path, sample_rate=aug_wav.sample_rate)


'''
            dilate_distribution=augs["dilate_distribution"],
            dilate_min=augs["dilate_min"],
            dilate_max=augs["dilate_max"],
            pitch_shift_distribution=augs["pitch_shift_distribution"],
            pitch_shift_min=augs["pitch_shift_min"],
            pitch_shift_max=augs["pitch_shift_max"]
'''
        



        




