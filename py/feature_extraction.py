import json
import os
import warnings

import numpy as np
from librosa import power_to_db, stft, magphase
from librosa.feature import melspectrogram
from librosa.display import specshow
import matplotlib.pyplot as plt

class FeatureExtract:
    """A feature extractor for input audio in numpy arrays

    Args:
        label (string): The true class label of the starting file.
        filepath (string): The path to the base file on the filesystem.
        data (np.array): The numpy wav data that was generated by the augmentation class.
        sample_rate (int): Expected sample rate of the wavs. Default 16000
        feature_extractor (str): type of features (log-mel / power-spec)
        nfft (int): length of STFT frame
        hop_length (int): number of samples to hop between adjacent frames in output
        wincount (int): total number of frames
        nfilters (int): number of log-mel filters
        preemphasis_coefficient (float): value of leaky average coefficient in pre-emphasis

    """

    def __init__(
        self,
        label,
        filepath,
        features=None,
        data=None,
        sample_rate=16000

    ):
        """Initialize this instance and initialize the static class members if
        this is the first instance of FeatureExtract to be created."""
        self.label = label
        self.filepath = filepath
        self.filename = os.path.basename(self.filepath)
        self.path = os.path.dirname(self.filepath)
        self.sample_rate = sample_rate
        if isinstance(features, np.ndarray):
            self.features = features
        else:
            if isinstance(data, np.ndarray):
                self.wav_data = data
            else:
                self.wav_data = load_wav_file(self.filepath, sample_rate=self.sample_rate)
            self.wav_data = self.wav_data.flatten()

    def get_feature(self):
        """Get the features without the header.

        Returns:
            wav_data (numpy.array) The extracted feature data
        """
        return self.features

    def save_feature_file(self, file_path=None, features=None):
        """Saves audio sample data to a .wav audio file.

        Args:
            file_path (string): Path to save the file to.
            features (numpy array): The feature data to be written.
                Defaults to the current features.
        """
        assert file_path is not None
        if features is None:
            features = self.features

        np.save(file_path, features)


 
    def generate_feature(
        self,
        feature_name="log-mel",
        nfft=512,
        win_length=512,
        hop_length=256,
        wincount=62,
        nfilters=40,
        preemphasis_coefficient=0.96785
    ):
        """Augment the current wave file according to the parameters.

        Args:
            feature_name (str): type of features (log-mel / power-spec)
            nfft (int): length of STFT frame
            hop_length (int): number of samples to hop between adjacent frames in output
            wincount (int): total number of frames
            nfilters (int): number of log-mel filters
            preemphasis_coefficient (float): value of leaky average coefficient in pre-emphasis
            Return:
                (FeatureExtract): A new FeatureExtract instance.
        """
 
        if len(self.wav_data) < hop_length*wincount:
            raise Exception(
                "length of '{}' is only {}. minimum file length for given filterbank_params is"
                .format(self.filepath, len(self.wav_data), hop_length*wincount)
            )
        
        if feature_name == "log-mel":
            S = melspectrogram(
                y=self.wav_data,
                sr=self.sample_rate,
                n_fft=nfft,
                hop_length=hop_length,
                win_length=win_length,
                window='hann',
                center=True,
                pad_mode='constant',
                power=2.0,
                n_mels=nfilters
            )
            S = power_to_db (S, ref=np.max) # ~= 10 * log10(S) - 10 * log10(ref)

        elif feature_name == "power":
            S, _ = magphase(
                stft(
                y=self.wav_data,
                n_fft=nfft,
                hop_length=hop_length,
                win_length=win_length,
                window='hann',
                center=True,
                pad_mode='constant'),
                power=2
            )
        else:
            raise ValueError(f"unknown feature set {feature_name}")
        
        n_windows = S.shape[-1] // wincount
        # crop out remainder of frames
        S = S[:, 0:n_windows*wincount] 
        # transpose so that frequency is on the fast axis. Copy so array stays C-contiguous after transpose.
        feature_data = S.transpose().copy()
        feature_data = feature_data.reshape(n_windows, wincount, nfilters)

        fe = FeatureExtract(
            self.label,
            self.filename,
            features=feature_data
        )
        return fe
    
    
    def get_matplot_waveform(self):
        """Get the matplotlib waveform associated with the current Wav.

        Returns:
            (Matplot Plt)
        """
        plt.figure()
        plt.title("Waveform")
        plt.ylim(-1.0, 1.0)
        plt.plot(self.wav_data.flatten())
        return plt


    def save_waveform(self, file_path=None):
        """Render the waveform of the wave file as an image and save it.

        Args:
            filename (string): Where to save the figure.
        """
        assert file_path is not None
        plt = self.get_matplot_waveform()
        plt.savefig(file_path)
        # plt.show()  # Uncomment to plot out on screen
        plt.close()

    def get_matplot_spectrogram(self, nfft=512):
        """Get the matplotlib spectrogram associated with the current
        Wav."""
        data = self.wav_data.flatten()
        plt.specgram(data, nfft, self.sample_rate)
        plt.axis("off")
        plt.title("Spectrogram")
        return plt


    def save_spectrogram(self, file_path=None):
        """Create a spectrogram image for visualizing the current file and save
        it.

        Args:
            filename (string): Where to save the figure.
        """
        assert file_path is not None
        plt = self.get_matplot_spectrogram()
        plt.savefig(
            file_path,
            dpi=150,  # Dots per inch
            facecolor="none",
            bbox_inches="tight",
            pad_inches=0,
        )
        # plt.show()  # Uncomment to plot out immediately
        plt.close()
