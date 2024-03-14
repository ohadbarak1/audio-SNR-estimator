import numpy as np
import soundfile as sf


def load_wav_file(filepath, sample_rate=16000):
    """Loads an audio file and returns a float PCM-encoded array of samples.

    Args:
        filepath (str): path to the wav file
        sample_rate (int): Expected sample rate of the wavs. Default 16000

    Returns:
        (np.array): The sample data as floats between [-1.0, 1.0).
    """
    res, rate = sf.read(filepath, dtype="float32")
    assert rate == sample_rate, f"Sample rate of {filepath} is not {sample_rate}"
    return res


def sample_distribution(lower_bound, upper_bound, p):
    """Returns a sample from a given range w.r.t a specified probability
    distribution.

    Args:
        lower_bound (float): the lower bound of range.
        upper_bound (float): the upper bound of range.
        p (str): the name of the distribution to be used.

    Returns:
        sample (float): one sampled number.
    """
    if isinstance(p, str) and p not in ["linear", "uniform"]:
        raise ValueError("Only 'linear' or 'uniform' distributions are supported")

    if isinstance(p, str) and p == "uniform":
        sample_value = np.random.uniform(lower_bound, upper_bound)
        return sample_value

    if isinstance(p, str) and p == "linear":
        sample_value = np.random.triangular(lower_bound, lower_bound, upper_bound)
        return sample_value


