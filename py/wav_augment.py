import json
import os
import warnings

import numpy as np
import soundfile as sf
from scipy import signal
import librosa as la

from wav_utils import load_wav_file, sample_distribution
from wav_metadata import WavMetadata


def plt_import(func):
    def inner(*args, **kwargs):
        global plt
        import matplotlib.pyplot as plt

        return func(*args, **kwargs)

    return inner


class BackgroundWav:
    """A wrapper for wav files that are used as background for
    augmentations. No augmentation functionality.

    Args:
        label (string): The true class label of the starting file.
        filepath (string): The path to the base file we will load from disk.
        sample_rate (int): Expected sample rate of the wavs. Default 16000
    """

    def __init__(self, label, filepath="", data=None, sample_rate=16000):
        """Initialize this wav file by loading from the file system."""
        self.label = label
        self.filepath = filepath
        self.filename = os.path.basename(self.filepath)
        self.path = os.path.dirname(self.filepath)
        self.sample_rate = sample_rate
        if isinstance(data, np.ndarray):
            self.wav_data = data
        else:
            self.wav_data = load_wav_file(self.filepath, sample_rate=self.sample_rate)
        self.wav_data = self.wav_data.flatten()


class Wav:
    """A wrapper for wav files and the operations we want to perform on them,
    including loading, writing, data augmentation, visualization etc.

    Args:
        label (string): The true class label of the starting file.
        filepath (string): The path to the base file on the filesystem.
        data (Wav Obj): The data that was just produced by augmentation.
        meta (WavMetadata): The metadata for the wav.
        augmentation_samples (int): The number of samples to get from a wav during augmentation.
        sample_rate (int): Expected sample rate of the wavs. Default 16000
        foreground_data (np.array): The foreground data used for the augmentation in augment_data.
        background_data (np.array): The background data used for the augmentation in augment_data.
    """

    def __init__(
        self,
        label,
        filepath,
        data=None,
        meta=None,
        augmentation_samples=16000,
        sample_rate=16000,
        foreground_data=None,
        background_data=None,
    ):
        """Initialize this instance and initialize the static class members if
        this is the first instance of Wav to be created."""
        self.label = label
        self.filepath = filepath
        self.filename = os.path.basename(self.filepath)
        self.path = os.path.dirname(self.filepath)
        self.augmentation_samples = augmentation_samples
        self.sample_rate = sample_rate
        if isinstance(data, np.ndarray):
            self.wav_data = data
        else:
            self.wav_data = load_wav_file(self.filepath, sample_rate=self.sample_rate)
        self.wav_data = self.wav_data.flatten()
        self.meta = meta
        self.foreground_data = foreground_data
        self.background_data = background_data

    def get_wav_data(self):
        """Get the wav file samples without the header.

        Returns:
            wav_data (numpy.array) The wav data
        """
        return self.wav_data

    def save_wav_file(self, file_path=None, sample_rate=-1, wav_data=None):
        """Saves audio sample data to a .wav audio file.

        Args:
            filename (string): Path to save the file to.
            sample_rate (int): Samples per second to encode in the file.
            wav_data (numpy array): The data for the wav file we are going to write.
                Defaults to the current wav_data.
        """
        assert file_path is not None
        assert sample_rate > 0
        if wav_data is None:
            wav_data = self.wav_data.flatten()

        sf.write(file_path, wav_data, sample_rate)

    def _apply_data_augmentations(
        self,
        idx,
        background,
        background_label,
        background_samples,
        background_offset,
        foreground_volume_gain,
        sample_rate,
        snr,
        dilate,
        pitch_shift,
        keep_sources=False,
        spectral_shaping_factor=None,
        foreground_volume_norm=True,
    ):
        """Use numpy to augment the data according to the params that have been
        sampled or previously generated.

        Use Librosa for dilation, pitch shifting and feature extraction
        
        Args:
            idx (int): The integer identifier to associate with the metadata.
            background (bool): Should background audio be mixed in?
            background_label (str): the label of the background data
            background_samples (np.array): The background wav data before applying offset
            background_offset (int): the offset value used on background data
            foreground_volume_gain (float): the volume gain to be applied on foreground
            sample_rate (int): Expected sample rate of the wavs.
            snr (float): the snr in dB to be used to augment data
            dilate (float): the dilation factor used to dilate the foreground file.
            pitch_shift (float): the pitch shift factor in semitones used to pitch shift the foreground file.
            keep_sources (bool): Whether or not to keep the foreground and background sources used
                for the augmentation.
            spectral_shaping_factor (float): If not none, A random spectral shape will be applied to
                both foreground and background data.
            foreground_volume_norm (bool): If the foreground should be normalized or not
        Returns:
            Wav an augmented instance of the wav file wrapped by a new instance of
                the Wav class.
        """
        augmentation_length = self.augmentation_samples

        foreground = self.wav_data.flatten()
        if dilate:
            foreground = la.effects.time_stretch (foreground, rate=dilate)
        if pitch_shift:
            foreground = la.effects.pitch_shift (foreground, sr=sample_rate, n_steps=pitch_shift)

        if len(foreground > augmentation_length):
            foreground_offset = np.random.randint(0, len(foreground) - self.augmentation_samples + 1)
            foreground = foreground[foreground_offset : (foreground_offset + augmentation_length)]
        else:
            foreground = np.pad(foreground, (0, augmentation_length - len(foreground)), "constant")

        if spectral_shaping_factor:
            r = spectral_shaping_factor
            # apply a random spectral shape to foreground
            b = [1, np.random.uniform(-r, r), np.random.uniform(-r, r)]
            a = [1, np.random.uniform(-r, r), np.random.uniform(-r, r)]
            foreground = signal.lfilter(b, a, foreground)

            if background:
                # apply a random spectral shape to background
                b = [1, np.random.uniform(-r, r), np.random.uniform(-r, r)]
                a = [1, np.random.uniform(-r, r), np.random.uniform(-r, r)]
                background_samples = signal.lfilter(b, a, background_samples)

        if foreground_volume_norm:
            peak_max = np.max(foreground)
            peak_min = np.min(foreground)
            peak_abs = max(abs(peak_max), abs(peak_min), 1e-10)
            foreground = foreground * 1.0 / peak_abs
        scaled_foreground = foreground * foreground_volume_gain

        if background:
            background_clipped = background_samples[
                background_offset : (background_offset + augmentation_length)
            ]
            background_reshaped = background_clipped.reshape([augmentation_length, 1])
            background_reshaped = background_reshaped.flatten()
            background_reshaped = background_reshaped[:augmentation_length]
            assert len(background_reshaped) == augmentation_length

            # calculate power on foreground where there is no silence
            initial_snr = np.mean(np.power(scaled_foreground[scaled_foreground != 0], 2))\
                        / np.mean(np.power(background_reshaped, 2) + 1e-6)
            
            snr_linear = pow(10.0, snr / 10.0) # snr is specified in dB power units
            noise_coeff = np.sqrt(initial_snr / snr_linear)
            new_background_noise = background_reshaped * noise_coeff

            assert len(scaled_foreground) == augmentation_length
            background_add = scaled_foreground + new_background_noise
            augmented_wav_data = np.clip(background_add, -1.0, 1.0)

        else:
            augmented_wav_data = np.clip(scaled_foreground, -1.0, 1.0)
            new_background_noise = None

        augmented_wav_data = augmented_wav_data.reshape((-1, 1))
        meta = WavMetadata(
            LB=self.label,
            IX=idx,
            AV=foreground_volume_gain,
            SNR=snr,
            FN=self.filename,
            BF=background_label,
            BO=background_offset,
            PS=pitch_shift,
            DL=dilate,
        )
        filename = meta.get_filename()
        if keep_sources:
            sw = Wav(
                self.label,
                filename,
                data=augmented_wav_data,
                meta=meta,
                foreground_data=scaled_foreground,
                background_data=new_background_noise,
            )
        else:
            sw = Wav(
                self.label,
                filename,
                data=augmented_wav_data,
                meta=meta,
            )
        return sw

    def augment_data(
        self,
        background_wav=None,
        idx=0,
        sample_rate=16000,
        background=True,
        foreground_volume_distribution="uniform",
        foreground_volume_domain="linear",
        foreground_volume_max=1.000,
        foreground_volume_min=0.01,
        foreground_volume_norm=True,
        snr_distribution="uniform",
        snr_min=0,
        snr_max=24,
        dilate_distribution=None,
        dilate_min=0.8,
        dilate_max=1.2,
        pitch_shift_distribution=None,
        pitch_shift_min=-2.0,
        pitch_shift_max=2.0,
        keep_sources=False,
        spectral_shaping_factor=None,
    ):
        """Augment the current wave file according to the parameters.

        Args:
            background_wav (BackgroundWav): The background wav file that will augment the
                current file.
            idx (int): The integer identifier to associate with the metadata.
            sample_rate (int): Expected sample rate of the wavs. Default 16000
            background (bool): Should background audio be mixed in?
            volume_distribution (str): volume distribution over range.
            foreground_volume_domain (str): `log` or `linear`. `log` means to uniformly sample from
                the log domain and `linear` means to uniformly sample from the linear domain
            foreground_volume_max (float): The maximum value of the signal in augmentation.
                The full range value is 1.0.
            foreground_volume_min (float): The minimum value of the signal in augmentation.
                The full range value is 0.0, which results in silence.
            foreground_volume_norm (bool): If the foreground should be normalized or not
            snr_distribution (str): snr distribution over range.
            snr_min (int): The minimum signal to noise ratio.
            snr_max (int): The maximum signal to noise ratio.
            dilate_distribution (str): dilation distribution over range.
            dilate_min (float): The minimum dilation to be applied.
            dilate_max (float): The maximum dilation to be applied.
            pitch_shift_distribution (str): pitch shift distribution over range.
            pitch_shift_min (float): The minimum pitch shifting in semitones to be applied.
            pitch_shift_max (float): The maximum pitch shifting in semitones to be applied.
            keep_sources (bool): Whether or not to keep the foreground and background sources used
                for the augmentation.
            spectral_shaping_factor (float): If not none, A random spectral shape will be applied to
                both foreground and background data.
        Return:
            (Wav): A new augmented wav instance.
        """

        is_silence = self.label == "silence"
        if background and background_wav is None:
            raise ValueError(
                "To augment, Please provide a background file when background is "
                "set to True"
            )
        if background:
            background_label = background_wav.label
            background_samples = background_wav.wav_data.flatten()

            if len(background_samples) - self.augmentation_samples >= 0:
                background_offset = np.random.randint(0, len(background_samples) - self.augmentation_samples + 1)
            else:
                # The background files should currently be longer than 1 second
                raise Exception(
                    "length of '{}' is only {}".format(background_wav.filepath, len(background_samples))
                )
        else:
            background_label = "NOBACKGROUND"
            background_samples = None
            background_offset = 0

        if is_silence:
            foreground_volume_gain = 0
        elif foreground_volume_domain == "linear":
            foreground_volume_gain = sample_distribution(
                foreground_volume_min,
                foreground_volume_max,
                foreground_volume_distribution,
            )
        elif foreground_volume_domain == "log":
            # Map the floating point to relative dB range
            foreground_volume_db_min = 20 * np.log10(foreground_volume_min)
            foreground_volume_db_max = 20 * np.log10(foreground_volume_max)
            foreground_volume_db = np.random.uniform(foreground_volume_db_min, foreground_volume_db_max)
            foreground_volume_gain = 10 ** (foreground_volume_db / 20)
        else:
            raise Exception("foreground_volume_domain should be either `linear` or `log`")

        snr = sample_distribution(snr_min, snr_max, snr_distribution)

        if dilate_distribution:
            dilate = sample_distribution(dilate_min, dilate_max, dilate_distribution)
        else:
            dilate = None

        if pitch_shift_distribution:
            pitch_shift = sample_distribution(pitch_shift_min, pitch_shift_max, pitch_shift_distribution)
        else:
            pitch_shift = None

        augmentation_args = {
            "idx": idx,
            "background": background,
            "background_label": background_label,
            "background_samples": background_samples,
            "background_offset": background_offset,
            "foreground_volume_gain": foreground_volume_gain,
            "sample_rate": sample_rate,
            "snr": snr,
            "dilate": dilate,
            "pitch_shift": pitch_shift,
            "keep_sources": keep_sources,
            "spectral_shaping_factor": spectral_shaping_factor,
            "foreground_volume_norm": foreground_volume_norm,
        }
        return self._apply_data_augmentations(**augmentation_args)

    def get_wav_metadata(self):
        """Return the current metadata object or construct one if it is not
        already present."""
        if self.meta is not None:
            return self.meta
        else:
            swm = self.get_wav_metadata_for_file(self.label, self.filename)
            return swm

    @staticmethod
    def get_wav_metadata_for_file(label, filepath):
        """Construct a metadata object for the given file without loading the
        file.

        Args:
            label (string): The true class label of the starting file.
            filepath (string): The path to the base file we will load from disk.

        Returns:
            WavMetadata : The metadata object which would be created for the file.
        """
        return WavMetadata(
            LB=label,
            IX=-1,
            AV=1.0,
            SNR=0.0,
            FN=filepath,
            BF="none",
            BO=-1,
        )

    @plt_import
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

    @plt_import
    def save_waveform(self, file_path=None):
        """Render the waveform of the wave file as an image and save it.

        Args:
            filename (string): Where to save the figure.
        """
        assert file_path is not None
        plt = self.get_matplot_waveform()
        plt.savefig(file_path)
        # plt.show()  # Uncomment to plot out immediately
        plt.close()

    @plt_import
    def get_matplot_spectrogram(self):
        """Get the matplotlib spectrogram associated with the current
        Wav."""
        data = self.wav_data.flatten()
        nfft = 256  # Length of the windowing segments
        fs = 256  # Sampling frequency
        plt.specgram(data, nfft, fs)
        plt.axis("off")
        plt.title("Spectrogram")
        return plt

    @plt_import
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
