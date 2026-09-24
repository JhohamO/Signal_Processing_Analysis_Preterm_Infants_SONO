import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import neurokit2 as nk
import pandas as pd
from jinja2.nodes import List

#=======Definir las constantes del código=======
#Rutas
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MNE_FILES = DATA_DIR / "MNE"
RESULTS_DIR = DATA_DIR / "results"


def extraer_ecg_raw(raw: mne.io.Raw):
    """
    Extract the ECG signal from a mne raw file.
    :param raw: mne raw file
    :return: numpy array with ECG signal and the sampling rate
    """
    ecg_channel = raw.get_data(picks=["ECG"]).squeeze()
    sfreq = raw.info['sfreq']

    return ecg_channel, sfreq

def datos_hrv_sujeto (pathfile, tiempos, medidas_HRV):
    """
    Extraer la fila de datos del sujeto. Carga el archivo, lo secciona según tiempos y extrae las medidas de HRV para cada estímulo
    :param pathfile: Ruta al archivo .fif de MNE
    :param tiempos: Dataframe con los tiempos de inicio y final de cada estímulo en datapoints
    :param medidas_HRV: Medidas de HRV a extraer de cada estímulo
    :return: numpy array con los datos de sujeto de tamaño len(estimulos) * medidas_HRV
    """

    #Código del sujeto
    sujeto_id = pathfile.stem[:-4]

    #Si no se tienen lso tiempos, saltarlo
    if sujeto_id not in tiempos.index:
        return None

    # Leer archivo MNE
    raw = mne.io.read_raw_fif(
        pathfile,
        preload=True,
        verbose=False
    )

    # Extraer ECG
    ecg_signal, sampling_rate = extraer_ecg_raw(raw)

    # Tiempos del sujeto
    tiempos_sujeto = tiempos.loc[sujeto_id].values

    # ==========================================
    # Procesar ECG
    # ==========================================

    #Extraer la info de los picos
    signals, info = nk.ecg_process(
        ecg_signal,
        sampling_rate=sampling_rate
    )

    #Solo los picos que son los que se usan para las medidas de HRV
    r_peaks = signals["ECG_R_Peaks"]

    lista_sujeto = []

    # ==========================================
    # Calcular HRV por estímulo
    # ==========================================

    for start in range(0, len(tiempos_sujeto), 2):
        inicio = int(tiempos_sujeto[start])
        final = int(tiempos_sujeto[start + 1])

        # R-peaks correspondientes al estímulo
        peaks_estimulo = r_peaks.iloc[inicio:final].reset_index(drop=True)


        # HRV frecuencial
        hrv_frequencial = nk.hrv_frequency(
            peaks_estimulo,
            sampling_rate=sampling_rate,
            show=False
        )

        #Filtrar a solo las medidas de interés
        valores = hrv_frequencial[
            medidas_HRV
        ].values.squeeze()

        #Agregar las medidas correspondientes del estímulo a la lista
        lista_sujeto.extend(valores)

    return np.array(lista_sujeto)

def nombres_columnas (nombres_estimulos: list, nombres_medidas: list):
    """
    Crear la lista con el nombre de las columnas de los datos de cada sujeto
    :param nombres_estimulos: Nombres de los estímulos
    :param nombres_medidas: Nombres de los medidas que se extrayeron de cada estímulo
    :return: Lista de nombres de las columnas de sujeto
    """

    #Los nombres quedan de la forma: Estimulo_medida
    lista_nombres = [estimulo + "_" + medida for estimulo in nombres_estimulos for medida in nombres_medidas]

    return lista_nombres


"""USAR LAS FUNCIONES"""


#LIsta de medidas de HRV que se extraerán por estímulo
medidas_HRV = ["HRV_LF", "HRV_HF", "HRV_LFHF"]

#Excel donde se definen los tiempos
tiempos = pd.read_excel(DATA_DIR / "info_marcadores.xlsx", index_col=0)

#Crear un diccionario que va a ser base de datos
diccionario_bd_hrv = {}

#Recorrer los sujetos para cuantificar sus datos de HRV
for sujeto in MNE_FILES.glob("*.fif"):
    fila = datos_hrv_sujeto(sujeto, tiempos, medidas_HRV)

    if fila is not None:
        diccionario_bd_hrv[sujeto.stem[:-4]] = fila


#Nombres de las columnas
estimulos = ["Baseline", "Nursing", "Intervention", "Post"]
columnas = nombres_columnas(estimulos, medidas_HRV)

#Guardar en un archivo excel
df_HRV = pd.DataFrame.from_dict(diccionario_bd_hrv, orient="index", columns=columnas)
df_HRV.to_excel(RESULTS_DIR / "HRV.xlsx")

