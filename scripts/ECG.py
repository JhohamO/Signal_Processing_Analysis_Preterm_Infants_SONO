import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import neurokit2 as nk

def extraer_ecg_raw(raw: mne.io.Raw):
    """
    Extract the ECG signal from a mne raw file.
    :param raw: mne raw file
    :return: numpy array with ECG signal and the sampling rate
    """
    ecg_channel = raw.get_data(picks=["ECG"]).squeeze()
    sfreq = raw.info['sfreq']

    return ecg_channel, sfreq

def ecg_processing(ecg_signal, sfreq, )
