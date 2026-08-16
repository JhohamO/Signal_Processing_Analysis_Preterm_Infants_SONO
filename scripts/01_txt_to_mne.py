import pandas  as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib
import mne

#Definir la ruta de trabajo
ROOT= Path(__file__).resolve().parent.parent

"""Definir constantes del código"""
#Ruta de los datos
DATA_DIR = ROOT/'data'
DATA_DIR_ORIGINAL_DATA = DATA_DIR/'original_data'
DATA_MNE = DATA_DIR/'MNE'

#Nombres canales del mne.raw
#Para ajustar a la convención de MNE, se cambió T3-T4 por T7-T8
NAME_CHANNELS = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2", "T7", "T8", "A1", "A2", "Cz", "ROC", "LOC", "ECG",
                 "EMG-", "EMG+"]

#Tipo de dato de cada canal
CH_TYPES = ["eeg"] * 13 + ["eog"] * 2 + ["ecg"] + ["emg"] * 2

#Frecuencia de muestreo de los registros
SFREQ = 250


def load_txt_as_raw(filename):
    """
    Turn a txt file into a raw mne
    :param filename: Name of the file
    :return: mne.io.RawArray
    """

    #Abrir y leer las lineas del archivo txt
    with open(DATA_DIR_ORIGINAL_DATA / filename, 'r', encoding="utf-16") as f:
        lines = f.readlines()

    #Buscar la linea que coincide con la estructura del encabezado
    header_line = next(line for line in lines if line.startswith('% Fecha.Hora'))

    #Separar el texto a a las correspondientes columnas
    header_columns = header_line.split('\t')

    #Quitar las 2 columnas que no pertenecen
    header_columns = header_columns[:1] + header_columns[2:-1]

    #Arreglar los textos de las columnas
    header_columns = [col.replace("%", "").strip() for col in header_columns]

    #Leer el text como un df con sus respectivas columnas
    data = pd.read_csv(DATA_DIR_ORIGINAL_DATA/filename, sep="\t", comment="%", encoding="utf-16", names=header_columns)

    #Remover las columnas que no se van a utilizar por definiciones previas
    data = data.drop(columns=['Fecha.Hora', 'EB', 'Marca','C033', 'C034', 'C035', 'PHOTIC', "C003", "C004", "C011", "C012",
                              "C015", "C016", "C019", "C021", "C025", "C026", "C029", "C030", "C031", "C032"])

    #Transponer para hacerlo vector fila
    data = data.to_numpy().T

    #Pasar de mV a V
    data = data / 1000

    #Croear el objeto de info
    info = mne.create_info(ch_types=CH_TYPES, ch_names=NAME_CHANNELS, sfreq=SFREQ)

    #Crear el raw
    raw = mne.io.RawArray(data, info)

    #Definir el montaje del raw
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    return raw

# Crear carpeta destino
DATA_MNE.mkdir(parents=True, exist_ok=True)

#Recorrer los txt para convertilos a MNE
for txt_file in DATA_DIR_ORIGINAL_DATA.glob("*.txt"):
    #Convertir el txt a raw
    raw = load_txt_as_raw(txt_file)

    #Ruta de salida del archivo
    output_file = DATA_MNE / f"{txt_file.stem}_raw.fif"

    #Guardar como archivo fif
    raw.save(
        output_file,
        overwrite=True
    )
