import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import neurokit2 as nk
import pandas as pd
import joblib

#=======Definir las constantes del código=======
#Rutas
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MNE_FILES = DATA_DIR / "MNE"
RESULTS_DIR = DATA_DIR / "results"

#Bandas de HRV adaptadas a neonatos según literatura (Hz)
BANDA_LF = (0.04, 0.2)
#Límite superior de HF en 1 Hz: por encima de HR/2 (~0.95 Hz con la HR de la muestra) no hay variabilidad real
BANDA_HF = (0.2, 1.0)


def extraer_ecg_raw(raw: mne.io.Raw):
    """
    Extract the ECG signal from a mne raw file.
    :param raw: mne raw file
    :return: numpy array with ECG signal and the sampling rate
    """
    ecg_channel = raw.get_data(picks=["ECG"]).squeeze()
    sfreq = raw.info['sfreq']

    return ecg_channel, sfreq

def datos_hrv_sujeto (pathfile, tiempos, medidas_HRV,
                      estimulos = ("Baseline", "Nursing", "Intervention", "Post")):
    """
    Extraer la fila de datos del sujeto. Carga el archivo, lo secciona según tiempos y extrae las medidas de HRV para cada estímulo
    :param pathfile: Ruta al archivo .fif de MNE
    :param tiempos: Dataframe con los tiempos de inicio y final de cada estímulo en datapoints
    :param medidas_HRV: Medidas de HRV a extraer de cada estímulo
    :param estimulos: Tupla de nombres de los estímulos
    :return:
    - numpy array con los datos de hrv sujeto de tamaño len(estimulos) * medidas_HRV
    - numpy array cob los valores de frecuencia cardiaca promedio por estímulo
    - diccionario de la forma estimulo: serie de tiempo con la frecuencia cardiaca durante el estímulo completo
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

    #Valores de frecuencia cardiaca del sujeto
    ecg_rate = signals["ECG_Rate"]

    lista_sujeto_hrv = []
    lista_sujeto_hr = []
    dict_ecg_rates = {}

    # ==========================================
    # Calcular HRV por estímulo
    # ==========================================

    for ind_start, start in enumerate(range(0, len(tiempos_sujeto), 2)):
        inicio = int(tiempos_sujeto[start])
        final = int(tiempos_sujeto[start + 1])

        # R-peaks correspondientes al estímulo
        peaks_estimulo = r_peaks.iloc[inicio:final].reset_index(drop=True)

        #HR corresponeidnete al estímulo
        hr_estimulo = ecg_rate.iloc[inicio:final].reset_index(drop=True)


        # HRV frecuencial
        hrv_frequencial = nk.hrv_frequency(
            peaks_estimulo,
            sampling_rate=sampling_rate,
            lf=BANDA_LF,
            hf=BANDA_HF,
            show=False
        )

        #Filtrar a solo las medidas de interés
        valores = hrv_frequencial[
            medidas_HRV
        ].values.squeeze()

        #Agregar las medidas correspondientes del estímulo a la lista
        lista_sujeto_hrv.extend(valores)

        #=====Frecuencia cardiaca=======
        #Valor promedio de la frecuencia en el estímulo
        lista_sujeto_hr.append(hr_estimulo.mean())

        #Serie de tiempo de las frecuencias
        dict_ecg_rates[estimulos[ind_start]] = hr_estimulo.to_numpy()

    return np.array(lista_sujeto_hrv), np.array(lista_sujeto_hr), dict_ecg_rates

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
diccionario_bd_hr = {}
series_hr = {}


"""FUNCIONAL"""

#Recorrer los sujetos para cuantificar sus datos de HRV
for sujeto in MNE_FILES.glob("*.fif"):
    resultado = datos_hrv_sujeto(sujeto, tiempos, medidas_HRV)

    if resultado is None:
        continue

    #Desempaquetar el resultado
    hrv_row, hr_row, dict_ecg_rates = resultado
    diccionario_bd_hrv[sujeto.stem[:-4]] = hrv_row
    diccionario_bd_hr[sujeto.stem[:-4]] = hr_row
    series_hr[sujeto.stem[:-4]] = dict_ecg_rates

#Nombres de las columnas
estimulos = ["Baseline", "Nursing", "Intervention", "Post"]
columnas = nombres_columnas(estimulos, medidas_HRV)

#Guardar en un archivo excel
df_HRV = pd.DataFrame.from_dict(diccionario_bd_hrv, orient="index", columns=columnas)
df_HR = pd.DataFrame.from_dict(diccionario_bd_hr, orient="index", columns=estimulos)

#Crear el excel con 2 hojas
with pd.ExcelWriter(RESULTS_DIR / "ECG_results.xlsx") as writer:
    df_HRV.to_excel(writer, sheet_name="HRV")
    df_HR.to_excel(writer, sheet_name="HR")

joblib.dump(series_hr, RESULTS_DIR / "series_hr.joblib")