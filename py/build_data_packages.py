#!/usr/bin/env python
from wav_augment import Wav, BackgroundWav
from feature_extraction import FeatureExtract
import numpy as np
import json
import os
from npy_append_array import NpyAppendArray
from datetime import datetime
from random import shuffle
import argparse

def build_file_list (src_dir, data_dirs, key):
    flist = []
    for data_dir in data_dirs:
        d = os.path.join (src_dir, data_dir[key])
        for f in os.listdir(d):
            if f.endswith(".wav"):
                flist.append(os.path.join (d,f))
    return flist

def build_packages (
    augmentation_number=0,
    augmentation_length=16000,
    bg_per_fg=1,
    fg_files=[],
    bg_files=[],
    data_file=None,
    label_file=None,
    augmented_dir="",
    write_augmented=False,
    feature_name="log-mel",
    nfft=512,
    win_length=512,
    hop_length=256,
    wincount=62,
    nfilters=40,
    foreground_volume_distribution="uniform",
    foreground_volume_domain="linear",
    foreground_volume_max=1.000,
    foreground_volume_min=0.01,
    foreground_volume_norm=True,
    snr_distribution="uniform",
    snr_min=0,
    snr_max=24,
    pitch_shift_distribution="uniform",
    pitch_shift_min=-1.0,
    pitch_shift_max=1.0,
    dilate_distribution="uniform",
    dilate_min=0.9,
    dilate_max=1.1):

    os.remove(data_file) if os.path.exists(data_file) else None
    os.remove(label_file) if os.path.exists(label_file) else None
    data_fh = NpyAppendArray(data_file)
    label_fh = NpyAppendArray(label_file)

    for _i_ in range(augmentation_number // bg_per_fg):

        fg_idx = np.random.randint(0, len(fg_files))
        fg = Wav("speech", fg_files[fg_idx], augmentation_samples=augmentation_length)

        for _j_ in range(bg_per_fg):
            bg_idx = np.random.randint(0, len(bg_files))
            bg_label = bg_files[bg_idx].split('/')[-2] # parent directory name is background label 
            bg = BackgroundWav(bg_label, bg_files[bg_idx])
    
            aug_wav = fg.augment_data (
                background_wav=bg,
                idx=fg_idx,
                foreground_volume_distribution=foreground_volume_distribution,
                foreground_volume_domain=foreground_volume_domain,
                foreground_volume_max=foreground_volume_max,
                foreground_volume_min=foreground_volume_min,
                foreground_volume_norm=foreground_volume_norm,
                snr_distribution=snr_distribution,
                snr_min=snr_min,
                snr_max=snr_max,
                pitch_shift_distribution=pitch_shift_distribution,
                pitch_shift_min=pitch_shift_min,
                pitch_shift_max=pitch_shift_max,
                dilate_distribution=dilate_distribution,
                dilate_min=dilate_min,
                dilate_max=dilate_max
            )

            featureObj = FeatureExtract (aug_wav.label,
                                        aug_wav.filepath,
                                        data=aug_wav.wav_data,
                                        sample_rate=aug_wav.sample_rate)
            
            feat = featureObj.generate_feature (
                feature_name=feature_name,
                nfft=nfft,
                win_length=win_length,
                hop_length=hop_length,
                wincount=wincount,
                nfilters=nfilters
            )

            # concatenate augmented audio data into numpy output file
            data_fh.append(feat.features)

            # concatenate augmented SNR label into numpy output file
            applied_snr = aug_wav.meta.get_augmentation_params()["snr"],
            label = np.array([applied_snr], dtype=np.float32)
            n_windows = feat.features.shape[0]
            snr_labels = label.repeat(n_windows) # the SNR label for the audio file is spread over all time windows
            label_fh.append(snr_labels)

            if write_augmented:
                aug_wav_path = os.path.join(augmented_dir, aug_wav.filename)
                aug_wav.save_wav_file (file_path=aug_wav_path, sample_rate=aug_wav.sample_rate)

    data_fh.close()
    label_fh.close()



def data_packager(json_path):

    seed = 121212
    np.random.seed(seed)
    date_string = f'{datetime.now():%Y%m%d_%H%M%S}'

    with open(json_path, "r") as f:
        augment_params = json.load(f)
    
    input = augment_params["input"]
    data_dirs = augment_params["input"]["data_directories"]
    augs = augment_params["input"]["augmentation_defaults"]
    fbank = augment_params["input"]["filterbank_params"]
    src_dir = input["source_directory"]

    train_pct = input["train_pct"]
    valid_pct = input["valid_pct"]
    test_pct = input["test_pct"]
    if train_pct + valid_pct + test_pct != 100:
        raise ValueError (
            "sum of train+valid+test percentage should be 100"
        )

    package_dir = os.path.join(input["package_directory"], date_string+'_'+str(input["augmentation_number"])+'_files')
    augmented_dir = os.path.join(input["augmented_directory"], date_string+'_'+str(input["augmentation_number"])+'_files')
    os.makedirs(package_dir)
    os.makedirs(augmented_dir)

    fg_wav_files = build_file_list(src_dir, data_dirs["speech"], "directory_name")
    bg_wav_files = build_file_list(src_dir, data_dirs["background"], "directory_name")

    shuffle(fg_wav_files)
    shuffle(bg_wav_files)

    train_ib = 0
    train_ie = int(len(fg_wav_files) * float(train_pct) / 100.)
    valid_ib = train_ie
    valid_ie = int(valid_ib + len(fg_wav_files) * float(valid_pct) / 100.)
    test_ib  = valid_ie
    test_ie  = len(fg_wav_files)

    bg_train_ib = 0
    bg_train_ie = int(len(bg_wav_files) * float(train_pct) / 100.)
    bg_valid_ib = bg_train_ie
    bg_valid_ie = int(bg_valid_ib + len(bg_wav_files) * float(valid_pct) / 100.)
    bg_test_ib  = bg_valid_ie
    bg_test_ie  = len(bg_wav_files)


    fgs = [fg_wav_files[train_ib:train_ie],
           fg_wav_files[valid_ib:valid_ie],
           fg_wav_files[test_ib:test_ie]
    ]

    bgs = [bg_wav_files[bg_train_ib:bg_train_ie],
           bg_wav_files[bg_valid_ib:bg_valid_ie],
           bg_wav_files[bg_test_ib:bg_test_ie]
    ]

    data_fnames = [os.path.join (package_dir, 'train_data.npy'),
                   os.path.join (package_dir, 'valid_data.npy'),
                   os.path.join (package_dir, 'test_data.npy')
    ]

    label_fnames = [os.path.join (package_dir, 'train_labels.npy'),
                    os.path.join (package_dir, 'valid_labels.npy'),
                    os.path.join (package_dir, 'test_labels.npy')
    ]

    aug_number =  [int(input["augmentation_number"] * float(train_pct) / 100.),
                   int(input["augmentation_number"] * float(valid_pct) / 100.),
                   int(input["augmentation_number"] * float(test_pct) / 100.)
    ]

    package_args = {
        "augmentation_number": 0,
        "augmentation_length": input["augmentation_length"],
        "bg_per_fg": input["backgrounds_per_foreground"],
        "fg_files": [],
        "bg_files": [],
        "data_file": None,
        "label_file": None,
        "augmented_dir": augmented_dir,
        "write_augmented": True,
        "feature_name": fbank["feature_name"],
        "nfft": fbank["nfft"],
        "win_length": fbank["win_length"],
        "hop_length": fbank["hop_length"],
        "wincount": fbank["wincount"],
        "nfilters":fbank["nfilters"],
        "foreground_volume_distribution": augs["foreground_volume_distribution"],
        "foreground_volume_domain": augs["foreground_volume_domain"],
        "foreground_volume_max": augs["foreground_volume_max"],
        "foreground_volume_min": augs["foreground_volume_min"],
        "foreground_volume_norm": augs["foreground_volume_norm"],
        "snr_distribution": augs["snr_distribution"],
        "snr_min": augs["snr_min"],
        "snr_max": augs["snr_max"],
        "pitch_shift_distribution": augs["pitch_shift_distribution"],
        "pitch_shift_min": augs["pitch_shift_min"],
        "pitch_shift_max": augs["pitch_shift_max"],
        "dilate_distribution": augs["dilate_distribution"],
        "dilate_min": augs["dilate_min"],
        "dilate_max": augs["dilate_max"]
    }

    for i in range(len(fgs)):
        package_args["augmentation_number"] = aug_number[i]
        package_args["fg_files"] = fgs[i]
        package_args["bg_files"] = bgs[i]
        package_args["data_file"] = data_fnames[i]
        package_args["label_file"] = label_fnames[i]

        build_packages (**package_args)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('json_params', type=str, help='json input parameters file')

    args, unparsed = ap.parse_known_args()
    data_packager (args.json_params)





