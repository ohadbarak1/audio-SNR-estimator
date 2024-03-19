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
    
    data_output_filename = 'out_data.npy'
    label_output_filename = 'out_label.npy'
    
    # We step over an existing saved data
    os.remove(data_output_filename) if os.path.exists(data_output_filename) else None
    os.remove(label_output_filename) if os.path.exists(label_output_filename) else None
    feature_out = NpyAppendArray(data_output_filename)
    label_out = NpyAppendArray(label_output_filename)

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



    for _i_ in range(input["augmentation_number"] // input["background_per_foreground"]):

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
            feature_out.append(feat.features)

            # concatenate augmented SNR label into numpy output file
            applied_snr = aug_wav.meta.get_augmentation_params()["snr"],
            label = np.array([applied_snr], dtype=np.float32)
            n_windows = feat.features.shape[0]
            snr_labels = label.repeat(n_windows) # the SNR label for the audio file is spread over all time windows
            label_out.append(snr_labels)

            #print (feat.features.shape)
            aug_wav_path = os.path.join(augment_params["input"]["augmented_directory"], aug_wav.filename)
            aug_wav.save_wav_file (file_path=aug_wav_path, sample_rate=aug_wav.sample_rate)


    feature_out.close()
    label_out.close()







        




