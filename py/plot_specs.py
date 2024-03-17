import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from librosa import power_to_db
from librosa.feature import melspectrogram
from librosa.display import specshow
from tqdm import tqdm

directories=["/home/ohad/DG/data/librispeech-dev-clean"]
sr=16000
nfft=512
win_length = 512
overlap=0.5 # overlap of STFT windows [0,1]
hop_length=int((1-overlap)*win_length)+1
n_mels=64
wincount=100

total_wav_time = 1.6
vmin_db=80
vmax_db=0
max_files=3
limit=3
min_rms_ratio=0.1 # ratio of 0.1 = -20 dB relative to maximum energy frame
centroid_threshold=10000




def read_wavs (ref_dir, limit=2000):
    wav_files = [
        os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".wav")
    ]

    wav_files_out = []

    for k, wav_file in enumerate(tqdm(wav_files)): #, desc=f"Generating embeddings for {ref_dir}"):
        if k >= limit:
            break
        wav_files_out.append(wav_file)
		
    print (wav_files_out)
    return wav_files_out
		


if __name__ == "__main__":



    file_paths = {}
    for directory in directories:
        wav_files_ = read_wavs (directory, limit=limit)
        file_paths[directory] = wav_files_

    # flatten file paths dictionary into a single list
    file_names = []
    for k, v in file_paths.items():
        for i in range(len(v)):
            file_names.append(v[i])


    np.random.seed(24602)
    # randomly select only max_files files. set max_files to total number of files if it was set to 0 or > total number of files
    selected_files = np.random.choice (len(file_names), size=max_files, replace=False)


    for i in selected_files:
        print (file_names[i])
        signal, _ = sf.read(file_names[i])
        #S, phase = librosa.magphase(librosa.stft(y=combined_wavs[i], n_fft=nfft, hop_length=hop_length, win_length=win_length))
        S = melspectrogram (signal, sr=sr, n_fft=nfft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)


        S_dB = power_to_db (S, ref=np.max)
        print (S_dB.shape)
        n_windows = S_dB.shape[-1] // wincount
        S_dB = S_dB[:, 0:n_windows*wincount]
        S_dB = S_dB.transpose()
        S_dB = S_dB.reshape(n_windows, wincount, n_mels)
        print (S_dB.shape)

        for w in range(n_windows):
            fig, ax = plt.subplots()
            spec_plot = specshow (S_dB[w, : ,:], y_axis='time', x_axis='mel', sr=sr, fmax=sr//2, ax=ax)
            fig.colorbar (spec_plot, ax=ax, format='%+2.0f, dB')
            ax.set(title=file_names[i])
            plt.show()


