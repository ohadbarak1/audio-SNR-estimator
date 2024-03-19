#!/usr/bin/env python
from wav_augment import Wav, BackgroundWav
from feature_extraction import FeatureExtract
import numpy as np
import json
import os
from npy_append_array import NpyAppendArray
from datetime import datetime
import hashlib
from random import shuffle

def build_file_list (src_dir, data_dirs, key):
    flist = []
    for data_dir in data_dirs:
        d = os.path.join (src_dir, data_dir[key])
        for f in os.listdir(d):
            if f.endswith(".wav"):
                flist.append(os.path.join (d,f))
    return flist

if __name__ == '__main__':
    seed = 24601
    np.random.seed(seed)
    date_string = f'{datetime.now():%Y%m%d_%H%M%S}'
    #hashlib.shake_128(date_string.encode("ascii")).hexdigest(8)

    json_path = "/home/ohad/DG/par/data_defaults.json"

    with open(json_path, "r") as f:
        augment_params = json.load(f)
    
    train_data_fname = 'train_data.npy'
    train_labels_fname = 'train_labels.npy'
    valid_data_fname = 'valid_data.npy'
    valid_labels_fname = 'valid_labels.npy'
    test_data_fname = 'test_data.npy'
    test_labels_fname = 'test_labels.npy'
    
    # We step over an existing saved data
    os.remove(train_data_fname) if os.path.exists(train_data_fname) else None
    os.remove(train_labels_fname) if os.path.exists(train_labels_fname) else None
    train_data_fh = NpyAppendArray(train_data_fname)
    train_labels_fh = NpyAppendArray(train_labels_fname)

    fg_wav_files = []
    bg_wav_files = []
    
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

    package_dir = os.path.join(input["package_directory"], date_string)
    augmented_dir = os.path.join(input["augmented_directory"], date_string)
    os.makedirs(package_dir)
    os.makedirs(augmented_dir)



        
    fg_wav_files = build_file_list(src_dir, data_dirs["speech"], "directory_name")
    bg_wav_files = build_file_list(src_dir, data_dirs["background"], "directory_name")

    #for d in data_dirs["speech"]:
    #    speech_dir = os.path.join (src_dir, d["directory_name"])
    #    for f in os.listdir(speech_dir):
    #        if f.endswith(".wav"):
    #            fg_wav_files.append(os.path.join (speech_dir,f))
    
    #for d in data_dirs["background"]:
    #    background_dir = os.path.join (src_dir, d["directory_name"])
    #    for f in os.listdir(background_dir):
    #        if f.endswith(".wav"):
    #            bg_wav_files.append(os.path.join (background_dir,f))

    shuffle(fg_wav_files)
    shuffle(bg_wav_files)

    train_ib = 0
    train_ie = len(fg_wav_files) * float(train_pct) / 100.
    valid_ib = train_ie
    valid_ie = train_ie + len(fg_wav_files) * float(valid_pct) / 100.
    test_ib  = valid_ie + len(fg_wav_files) * float (test_pct) / 100.
    test_ie  = len(fg_wav_files)

    fg_train = fg_wav_files[train_ib:train_ie]
    
    for _i_ in range(input["augmentation_number"] // input["backgrounds_per_foreground"]):

        fg_idx = np.random.randint(0, len(fg_wav_files))
        fg = Wav("speech", fg_wav_files[fg_idx], augmentation_samples=input["augmentation_length"])

        for _j_ in range(input["background_per_foreground"]):
            bg_idx = np.random.randint(0, len(bg_wav_files))
            bg_label = bg_wav_files[bg_idx].split('/')[-2] # parent directory name is background label 
            bg = BackgroundWav(bg_label, bg_wav_files[bg_idx])
    
            aug_wav = fg.augment_data (
                background_wav=bg,
                idx=fg_idx,
                foreground_volume_distribution=augs["foreground_volume_distribution"],
                foreground_volume_domain=augs["foreground_volume_domain"],
                foreground_volume_max=augs["foreground_volume_max"],
                foreground_volume_min=augs["foreground_volume_min"],
                foreground_volume_norm=augs["foreground_volume_norm"],
                snr_distribution=augs["snr_distribution"],
                snr_min=augs["snr_min"],
                snr_max=augs["snr_max"]

            )
            '''
                dilate_distribution=augs["dilate_distribution"],
                dilate_min=augs["dilate_min"],
                dilate_max=augs["dilate_max"],
                pitch_shift_distribution=augs["pitch_shift_distribution"],
                pitch_shift_min=augs["pitch_shift_min"],
                pitch_shift_max=augs["pitch_shift_max"]
            '''

            featureObj = FeatureExtract (aug_wav.label,
                                        aug_wav.filepath,
                                        data=aug_wav.wav_data,
                                        sample_rate=aug_wav.sample_rate)
            
            feat = featureObj.generate_feature (
                feature_name = fbank["feature_name"],
                nfft = fbank["nfft"],
                win_length = fbank["win_length"],
                hop_length = fbank["hop_length"],
                wincount = fbank["wincount"],
                nfilters = fbank["nfilters"]
            )


            # concatenate augmented audio data into numpy output file
            train_data_fh.append(feat.features)

            # concatenate augmented SNR label into numpy output file
            applied_snr = aug_wav.meta.get_augmentation_params()["snr"],
            label = np.array([applied_snr], dtype=np.float32)
            n_windows = feat.features.shape[0]
            snr_labels = label.repeat(n_windows) # the SNR label for the audio file is spread over all time windows
            train_labels_fh.append(snr_labels)

            #print (feat.features.shape)
            #aug_wav_path = os.path.join(augmented_dir, aug_wav.filename)
            #aug_wav.save_wav_file (file_path=aug_wav_path, sample_rate=aug_wav.sample_rate)


    train_data_fh.close()
    train_labels_fh.close()







        




