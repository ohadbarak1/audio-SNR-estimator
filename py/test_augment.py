#!/usr/bin/env python
from wav_augment import Wav, BackgroundWav
from feature_extraction import FeatureExtract
import numpy as np
import json
import os
from npy_append_array import NpyAppendArray

if __name__ == '__main__':
    seed = 24601
    np.random.seed(seed)

    json_path = "/home/ohad/DG/par/data_defaults.json"

    with open(json_path, "r") as f:
        augment_params = json.load(f)
    
    fg_wav_files = []
    bg_wav_files = []
    
    input = augment_params["input"]
    data_dirs = augment_params["input"]["data_directories"]
    augs = augment_params["input"]["augmentation_defaults"]
    fbank = augment_params["input"]["filterbank_params"]

    src_dir = input["source_directory"]
    for d in data_dirs["speech"]:
        speech_dir = os.path.join (src_dir, d["directory_name"])
        for f in os.listdir(speech_dir):
            if f.endswith(".wav"):
                fg_wav_files.append(os.path.join (speech_dir,f))
    
    for d in data_dirs["background"]:
        background_dir = os.path.join (src_dir, d["directory_name"])
        for f in os.listdir(background_dir):
            if f.endswith(".wav"):
                bg_wav_files.append(os.path.join (background_dir,f))

    data_output_filename = 'out.npy'
    label_output_filename = 'out_label.npy'
    fdata = NpyAppendArray(data_output_filename)
    ldata = NpyAppendArray(label_output_filename)

    for _ in range(input["augmentation_number"]):
        fg_idx = np.random.randint(0, len(fg_wav_files))
        bg_idx = np.random.randint(0, len(bg_wav_files))

        bg = BackgroundWav("bkgrnd", bg_wav_files[bg_idx])
        fg = Wav("speech", fg_wav_files[fg_idx], augmentation_samples=input["augmentation_length"]) 

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

        # concatenate augmented SNR label into numpy array
        applied_snr = aug_wav.meta.get_augmentation_params()["snr"],
        label = np.array([applied_snr], dtype=np.float32)
        n_windows = feat.features.shape[0]
        #snr_labels = np.hstack ((snr_labels, label.repeat(n_windows)))
        snr_labels = label.repeat(n_windows)

        # concatenate augmented audio data into numpy array
        #feature_data = np.vstack ((feature_data, feat.features))
        print (feat.features.dtype)
        print (snr_labels.dtype)
        fdata.append(feat.features)
        ldata.append(snr_labels)

        #print (feat.features.shape)
        #aug_wav_path = os.path.join(augment_params["input"]["augmented_directory"], aug_wav.filename)
        #aug_wav.save_wav_file (file_path=aug_wav_path, sample_rate=aug_wav.sample_rate)


    data = np.load(data_output_filename, mmap_mode="r")
    labels = np.load(label_output_filename, mmap_mode="r")

    print (data.shape)
    print (labels.shape)






        




