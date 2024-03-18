
"""Metadata class provides metadata for an augmentation with
information such as what foreground wav file is used, what background wav file
is used, what is the snr of the augmentation, etc."""

import hashlib
import urllib


class WavMetadata(object):
    """WavMetadata constructs the filename from the list of attributes,
    or parse a filename into the attributes that constructed it.

    Args:
        LB (string): The foreground label
        IX (int): The index of wav file
        AV (float): The audio volume for the foreground
        SNR (float): The signal to noise ratio
        FN (string): The foreground filename
        BF (string): The background filename
        BO (int): The index of the background file to seek to
        PS (float): The amount of pitch shifting to be applied
        DL (float): The amount of dilation to be applied
        filename (string): (optional, mutually exclusive with other params) the existing
            metadata filename we are going to parse.
    """

    def __init__(
        self,
        LB=None,
        IX=None,
        AV=None,
        SNR=None,
        FN=None,
        BF=None,
        BO=None,
        PS=None,
        DL=None,
        filename=None,
    ):
        """Initialize the WavMetadata parameters."""
        assert LB is not None or filename is not None
        assert LB is None or filename is None
        self.DELIMITER = "_"
        if filename is not None:
            elements = filename.split("/")[-1].split(self.DELIMITER)
            self.LB = elements[0][2:]  # class label
            self.FN = elements[1][2:]  # original file name
            self.IX = int(elements[2][2:])  # index
            self.AV = float(elements[3][2:])  # audio volume
            self.SNR = float(elements[4][3:])  # SNR
            self.BO = int(elements[5][2:])  # background offset
            self.BF = elements[6][2:]
            self.PS = elements[7][2:]
            self.DL = elements[8][2:-4]
            # filename ends with .wav, see func get_filename()
        else:
            assert LB is not None
            assert IX is not None
            assert AV is not None
            assert SNR is not None
            assert FN is not None
            assert BF is not None
            assert BO is not None
            self.LB = LB  # class label
            self.IX = IX  # index
            self.AV = AV  # audio volume
            self.SNR = SNR  # SNR
            self.BO = BO  # background offset
            self.FN = FN  # original file name
            self.BF = BF  # background file name
            self.PS = PS  # pitch shifting factor
            self.DL = DL  # dilation factor

    def __str__(self):
        """Get the filename."""
        return self.get_filename()

    def __lt__(self, other):
        """Make the class sortable by its index."""
        return int(self.IX) < int(other.IX)

    def get_augmentation_params(self):
        """Return the dictionary of params applied in the augmentation of the
        file."""
        params = {
            "label": self.LB,
            "foreground_file": self.FN,
            "background_file": self.BF,
            "snr": self.SNR,
            "foreground_volume": self.AV,
            "background_offset": self.BO,
            "pitch_shift": self.PS,
            "dilate": self.DL,
        }
        return params

    def get_FN_without_subfolder(self):
        """Unquotes encoded '/' as %2F into FN back to '/'. Splits result into
        file name & subfolder and returns file name.

        Returns:
            (str): File name
        """
        return urllib.parse.unquote(self.FN).split("/")[-1]

    def get_FN_subfolder(self):
        """Unquotes encoded '/' as %2F into FN back to '/'. Splits result into
        file name & subfolder and returns subfolder name.

        Returns:
            (str): Subfolder name
        """
        parts = urllib.parse.unquote(self.FN).split("/")
        return parts[len(parts) - 2]

    def get_BF_without_subfolder(self):
        """Unquotes encoded '/' as %2F into BF back to '/'. Splits result into
        file name & subfolder and returns file name.

        Returns:
            (str): File name
        """
        return urllib.parse.unquote(self.BF).split("/")[-1]

    def get_BF_subfolder(self):
        """Unquotes encoded '/' as %2F into BF back to '/'. Splits result into
        file name & subfolder and returns subfolder name.

        Returns:
            (str): Subfolder name
        """
        parts = urllib.parse.unquote(self.BF).split("/")
        return parts[len(parts) - 2]

    def get_hash(self):
        """Get the hash of the label and filename.
        """
        filename = self.LB + "/" + self.get_FN_without_subfolder() + ".wav"
        digest = hashlib.sha224(filename.encode("ascii", "strict")).hexdigest()
        return digest

    def get_filename(self):
        """Get the filename.
        """
        PS = 0. if not self.PS else self.PS
        DL = 0. if not self.DL else self.DL
        return (
            "LB"
            + self.LB
            + self.DELIMITER
            + "FN"
            + self.FN
            + self.DELIMITER
            + "IX"
            + str(self.IX)
            + self.DELIMITER
            + "AV"
            + "{:.4f}".format(self.AV)
            + self.DELIMITER
            + "SNR"
            + "{:.4f}".format(self.SNR)
            + self.DELIMITER
            + "BO"
            + str(self.BO)
            + self.DELIMITER
            + "BF"
            + self.BF
            + self.DELIMITER
            + "PS"
            + "{:.4f}".format(PS)
            + self.DELIMITER
            + "DL"
            + "{:.4f}".format(DL)
            + ".wav"
        )

    def set_BF(self, bf):
        """Setter for the source filename of the background.

        Args:
            bf (str): background file name value
        """
        self.BF = urllib.parse.quote(bf, safe="")

    def set_FN(self, fn):
        """Setter for the source filename of the foreground.

        Args:
            fn (str): Foreground file name value
        """
        self.FN = urllib.parse.quote(fn, safe="")

