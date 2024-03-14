import json
import os
import warnings

import numpy as np
import soundfile as sf
from scipy import signal

from wav_augmenter.wav_utils import (
    av_audio_process,
    get_foreground_time_shift_amount,
    get_wavdata_of_length,
    load_wav_file,
    sample_distribution,
)


def plt_import(func):
    def inner(*args, **kwargs):
        global plt
        import matplotlib.pyplot as plt

        return func(*args, **kwargs)

    return inner


class BackgroundWav:
    """A light weight wrapper for wave files that are used as background for
    augmentations. This class does not provide augmentation functionality.

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
    """A wrapper for wave files and the operations we want to perform on them,
    including loading, writing, data augmentation, visualization, etc.

    Args:
        label (string): The true class label of the starting file.
        filepath (string): The path to the base file we will load from disk.
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
        """Getter for the wave file format samples without the header.

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

    def _np_deterministic_augment_data(
        self,
        random,
        idx,
        background,
        background_label,
        background_samples,
        background_offset,
        foreground_volume_gain,
        keep_in_frame,
        sample_milliseconds,
        sample_rate,
        snr,
        dilate,
        pitch_shift,
        shift_range_ms=None,
        time_shift_amount=None,
        keep_sources=False,
        spectral_shaping_factor=None,
        center=False,
        foreground_volume_norm=True,
    ):
        """Use numpy to augment the data according to the params that have been
        sampled or previously generated.

        Uses ffmpeg filters for dilation and pitch shift. See https://ffmpeg.org/ffmpeg-filters.html

        Args:
            random (numpy.random): The numpy random number generator. This should be shared among
             all instances to ensure different threads are not generating the
             same sequence of data.
            idx (int): The integer identifier to associate with the metadata.
            background (bool): Should background audio be mixed in?
            background_label (str): the label of the background data
            background_samples (np.array): The background wav data before applying offset
            background_offset (int): the offset value used on background data
            foreground_volume_gain (float): the volume gain to be applied on foreground
            keep_in_frame (bool) : Do we want to keep the shifted wave file in the frame?
            sample_milliseconds (float) : The frame length in milliseconds.
            sample_rate (int): Expected sample rate of the wavs.
            snr (float): the snr to be used to augment data
            dilate (float): the dilation factor used to dilate the foreground file. The ffmpeg
                `atempo` filter.
            pitch_shift (float): the pitch shift factor used to pitch shift the foreground file.
                The ffmpeg `atempo` and `asetrate` filters are used.
            shift_range_ms (list(int)): A list comprises two numbers to indicate shift range of
                foreground wav, for example [100, 400] indicates minimum amount
                of shifting right is 100ms and the maximum amount of shifting
                right is 400ms. [-100, 400] indicates the maximum amount of
                shifting left is 100ms and the maximum shifting right is 400ms
            time_shift_amount (int): Number of samples(data points) to shift the foreground wav in
                augmentation. This parameter is mutually exclusive with shift_range_ms
            keep_sources (bool): Whether or not to keep the foreground and background sources used
                for the augmentation.
            spectral_shaping_factor (float): If not none, A random spectral shape will be applied to
                both foreground and background data.
            center (bool): Whether to shift from center during the temporal shifting step.
            foreground_volume_norm (bool): If the foreground should be normalized or not
        Returns:
            Wav an augmented instance of the wav file wrapped by a new instance of
                the Wav class.
        """
        assert (shift_range_ms is None and time_shift_amount is not None) or (
            shift_range_ms is not None and time_shift_amount is None
        ), "shift_range_ms and time_shift_amount are mutually exclusive"

        foreground = self.wav_data.flatten()
        if dilate or pitch_shift:
            foreground = av_audio_process(
                foreground,
                sample_rate,
                dilation_factor=dilate,
                pitch_shift_factor=pitch_shift,
            )
        foreground_len = len(foreground)
        if time_shift_amount is None:
            time_shift_amount = get_foreground_time_shift_amount(
                foreground_len,
                random,
                keep_in_frame,
                sample_milliseconds,
                shift_range_ms,
                sample_rate,
                center,
            )
        augmentation_length = self.augmentation_samples

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
            initial_snr = np.mean(
                np.power(scaled_foreground[scaled_foreground != 0], 2)
            ) / np.mean(np.power(background_reshaped, 2) + 1e-6)
            snr_ratio = pow(10.0, snr / 10.0)
            noise_coeff = np.sqrt(initial_snr / snr_ratio)
            new_background_noise = background_reshaped * noise_coeff
            scaled_foreground = get_wavdata_of_length(
                scaled_foreground, augmentation_length, time_shift_amount
            )
            assert len(scaled_foreground) == augmentation_length
            background_add = scaled_foreground + new_background_noise
            augmented_wav_data = np.clip(background_add, -1.0, 1.0)
        else:
            scaled_foreground = get_wavdata_of_length(
                scaled_foreground, augmentation_length, time_shift_amount
            )
            augmented_wav_data = np.clip(scaled_foreground, -1.0, 1.0)
            new_background_noise = None
        augmented_wav_data = augmented_wav_data.reshape((-1, 1))
        meta = WavMetadata(
            LB=self.label,
            IX=idx,
            AV=foreground_volume_gain,
            SNR=snr,
            SH=time_shift_amount,
            FN=self.filename,
            BF=background_label,
            BO=background_offset,
            RI=ri,
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
        random,
        background_wav=None,
        idx=0,
        sample_rate=16000,
        shift_range_ms=[-30, 30],
        background=True,
        keep_in_frame=False,
        sample_milliseconds=None,
        foreground_volume_distribution="uniform",
        foreground_volume_domain="linear",
        foreground_volume_max=1.000,
        foreground_volume_min=0.01,
        foreground_volume_norm=True,
        snr_distribution="uniform",
        snr_min=-5,
        snr_max=50,
        dilate_distribution=None,
        dilate_min=0.8,
        dilate_max=1.2,
        pitch_shift_distribution=None,
        pitch_shift_min=0.8,
        pitch_shift_max=1.2,
        keep_sources=False,
        spectral_shaping_factor=None,
        center=False,
    ):
        """Augment the current wave file according to the parameters.

        Args:
            random (numpy.random): The numpy random number generator. This should be shared among
                all instances to ensure different threads are not generating the
                same sequence of data.
            background_wav (BackgroundWav): The background wav file that will augment the
                current file.
            idx (int): The integer identifier to associate with the metadata.
            sample_rate (int): Expected sample rate of the wavs. Default 16000
            shift_range_ms (list(int)): A list comprises two numbers to indicate shift range of
                foreground wav, for example [100, 400] indicates minimum amount of shifting right
                is 100ms and the maximum amount of shifting right is 400ms. [-100, 400] indicates
                the maximum amount of shifting left is 100ms and the maximum shifting right is 400ms
            background (bool): Should background audio be mixed in?
            keep_in_frame (bool) : Do we want to keep the shifted wave file in the frame?
            sample_milliseconds (float) : The frame length in milliseconds. Deprecated and will be removed in a future version.
            volume_distribution (str/list(float)): How augmentation is spread over the volume range.
            foreground_volume_domain (str): `log` or `linear`. `log` means to uniformly sample from
                the log domain and `linear` means to uniformly sample from the linear domain
            foreground_volume_max (float): The maximum value of the signal in augmentation.
                The full range value is 1.0.
            foreground_volume_min (float): The minimum value of the signal in augmentation.
                The full range value is 0.0, which results in silence.
            foreground_volume_norm (bool): If the foreground should be normalized or not
            snr_distribution (str/list(float)): How augmentation is spread over the snr range.
            snr_min (int): The minimum signal to noise ratio. This will not be used
                if `background` is false or the pipeline controls this
            snr_max (int): The maximum signal to noise ratio. This will not be used
                if `background` is false or the pipeline controls this
            dilate_distribution (str/list(float)): How augmentation is spread over the dilation range.
            dilate_min (float): The minimum dilation to be applied.
            dilate_max (float): The maximum dilation to be applied.
            pitch_shift_distribution (str/list(float)): How augmentation is spread over the pitch shifting range.
            pitch_shift_min (float): The minimum pitch shifting to be applied.
            pitch_shift_max (float): The maximum pitch shifting to be applied.
            keep_sources (bool): Whether or not to keep the foreground and background sources used
                for the augmentation.
            spectral_shaping_factor (float): If not none, A random spectral shape will be applied to
                both foreground and background data.
            center (bool): Whether to shift from the center or not during temporal shift.
        Return:
            (Wav): A new augmented wav instance.
        """
        if sample_milliseconds is not None:
            warnings.warn(
                "Sample milliseconds is deprecated and not customizable anymore. It's value is automatically set to augmentation_samples * 1000 / sample_rate. This parameter will be removed in a future version.",
                DeprecationWarning,
            )

        sample_milliseconds = self.augmentation_samples * 1000 / self.sample_rate

        is_silence = self.label == "silence"
        if background and background_wav is None:
            raise ValueError(
                "To augment, Please provide a background file when background is "
                "set to True"
            )
        if background:
            assert (
                shift_range_ms[0] < shift_range_ms[1]
            ), f"Shift range is misconfigured as {shift_range_ms}"
            background_label = background_wav.label
            background_samples = background_wav.wav_data.flatten()

            if len(background_samples) - self.augmentation_samples >= 0:
                background_offset = random.randint(
                    0, len(background_samples) - self.augmentation_samples + 1
                )
            else:
                # The background files should currently be longer than 1 second
                raise Exception(
                    "length of '{}' is only {}".format(
                        background_wav.filepath, len(background_samples)
                    )
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
            foreground_volume_db = random.uniform(
                foreground_volume_db_min, foreground_volume_db_max
            )
            foreground_volume_gain = 10 ** (foreground_volume_db / 20)
        else:
            raise Exception(
                "foreground_volume_domain should be either `linear` or `log`"
            )

        snr = sample_distribution(snr_min, snr_max, snr_distribution)
        if dilate_distribution:
            dilate = sample_distribution(dilate_min, dilate_max, dilate_distribution)
        else:
            dilate = None
        if pitch_shift_distribution:
            pitch_shift = sample_distribution(
                pitch_shift_min, pitch_shift_max, pitch_shift_distribution
            )
        else:
            pitch_shift = None

        augmentation_args = {
            "random": random,
            "idx": idx,
            "background": background,
            "background_label": background_label,
            "background_samples": background_samples,
            "background_offset": background_offset,
            "foreground_volume_gain": foreground_volume_gain,
            "shift_range_ms": shift_range_ms,
            "keep_in_frame": keep_in_frame,
            "sample_milliseconds": sample_milliseconds,
            "sample_rate": sample_rate,
            "snr": snr,
            "dilate": dilate,
            "pitch_shift": pitch_shift,
            "keep_sources": keep_sources,
            "spectral_shaping_factor": spectral_shaping_factor,
            "center": center,
            "foreground_volume_norm": foreground_volume_norm,
        }
        return self._np_deterministic_augment_data(**augmentation_args)

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
            SH=0,
            FN=filepath,
            BF="none",
            BO=-1,
            RI="none",
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
