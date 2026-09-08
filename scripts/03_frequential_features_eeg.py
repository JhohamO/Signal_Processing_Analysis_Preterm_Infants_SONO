import mne.io
from eeg_utils.lectura import cargar_registro, cargar_correspondencias_estimulos, extraer_eventos
from eeg_utils.spectral import crear_segmentos, calcular_psd, promediar_banda
from eeg_utils.metrics import calcular_faa, calcular_supresion
import numpy as np
import pandas as pd
from pathlib import Path
import joblib



"""===========RUTAS=============="""
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR_CLEAN = DATA_DIR / "eeg_clean"
DATA_DIR_RESULTS = DATA_DIR / "results"

#Diccionario correspondencias eventos
DICT_EVENTOS = {
    1: "Baseline",
    2: "Nursing",
    3: "Intervention",
    4: "Post"
}

#Límites de bandas fisiológicas
bandas = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (13, 30)
}

#Cargar el df de tiempos
df_tiempos = pd.read_excel(DATA_DIR / "info_marcadores.xlsx", index_col=0)

#Función para la extracción de las medidas de frecuencia con respecto a LB de cada estímulo
def procesar_eeg_neonatos(filepath, dict_eventos, frecuencias, n_fft=512):
    """
    Extarer la medidas de potencia del EEG para cada estímulo y banda de frecuencia
    :param filepath: Ruta al archivo .fif raw de MNE
    :param dict_eventos: Diccionario de eventos
    :param n_fft:número de puntos para la fft.
    :param frecuencias: Tuplas con fmin y fmax de cada banda de interés
    :return:
    """

    #Leer el archivo
    raw = mne.io.read_raw_fif(filepath, preload=True)

    #Información de los canales
    all_channels = [ch for ch in raw.ch_names if raw.get_channel_types(picks=[ch])[0] == "eeg"]
    good_channels = [ch for ch in all_channels if ch not in raw.info['bads']]

    #Extracción de los tiempos para segmentar
    sujeto = filepath.stem[:-10]
    tiempos_dp = df_tiempos.loc[sujeto, :].to_numpy()

    # Crear los códigos de los eventos
    codes_events = np.repeat(np.arange(1, len(tiempos_dp) // 2 + 1), 2)

    # Crear la estructura de los eventos de MNE
    events = np.column_stack([
        tiempos_dp,
        np.zeros(len(tiempos_dp), dtype=int),
        codes_events
    ]
    )

    #Crear los segmentos del raw
    segmentos = crear_segmentos(raw, events, DICT_EVENTOS)

    #Espectros de potencias
    psds = calcular_psd(segmentos, n_fft=n_fft)

    potencias = {}

    #Recorrer las bandas de potencia predefinidas
    for banda, (fmin, fmax) in bandas.items():
        potencias[banda] = promediar_banda(psds, fmin_banda=fmin, fmax_banda=fmax, n_good_channels=len(good_channels))

    #Calculo de las potencias normalizadas con respecto a LB (como una supresión)
    pot_normalizadas = {}

    for banda in potencias:
        pot_normalizadas[banda] = calcular_supresion(potencias[banda], "Baseline", all_channels, good_channels)

    return pot_normalizadas


def serie_potencias_segmentos(segmentos, n_channels, good_indices, bandas, sfreq=250,
                              wind_length=3, wind_step=1.5):
    """
    Función para la extracción de las series de tiempo de potencia para todos los canales en las bandas de frecuencias
    dadas
    :param segmentos: Diccionario de forma: nombre_estimulo: segmento mne raw
    :param n_channels: Número total de canales de EEG
    :param good_indices: Índices de los canales no marcados como malos
    :param bandas: Diccionario de bandas de frecuencias de forma banda: (límite inferior, límite superior)
    :param sfreq: Frecuencia de muestreo
    :param wind_length: Tamaño de la ventana para la extracción de medidas frecuenciales (segundos)
    :param wind_step: Tamaño del paso del inicio de una ventana a la siguiente (segundos)
    :return: Diccionario con llaves los nombres de los segementos. Dentro de cada segmento las llaves de las distintas
    bandas de frecuencia y dentro de esta llave las correspondientes series de tiempo de frecuencia. matriz (n_ventanas, n_channels)
    """

    #Estrcutura para el return
    psds_segmentos = {
        segmento: {}
        for segmento in segmentos
    }

    #Recorrer los segmentos
    for nombre, segmento in segmentos.items():

        #=======Definir el ventaneo=======
        n_samples = segmento.n_times

        #Pasar tamaño y paso de la ventana a muestras
        wind_samples = int(wind_length * sfreq)
        step_samples = int(wind_step * sfreq)

        #n de ventanas posibles con los parámetros y el segmento dado
        n_ventanas = (n_samples - wind_samples) // step_samples + 1

        psds_bandas = {
            banda: np.full((n_ventanas, len(good_indices)), np.nan)
            for banda in bandas
        }

        #Recorrer la señal ventaneandola
        for i_wind, start in enumerate(range(0, n_samples - wind_samples + 1, step_samples)):

            #Crear el raw de la ventana
            sub_raw = segmento.copy().crop(tmin=start / sfreq, tmax = (start + wind_samples - 1) / sfreq)
            try:
                #Espectro de frecuencia en general
                psd_obj = sub_raw.compute_psd(method="welch", fmin=0.5, fmax=30, n_fft=512)
            except ZeroDivisionError:
                #La fila correspondiente a esa ventana quedará con nan para todos los canales
                continue

            #Si no si se llena esa fila con los valores correspondientes
            else:
                #Obtener la info del objeto de psd
                freqs_psds = psd_obj.freqs
                data_psds = psd_obj.get_data()

                #Recorrer las bandas de frecuencia para filtrar los psds
                for banda, (fmin, fmax) in bandas.items():
                    #Máscara para filtrar frecuencias
                    mask = (freqs_psds >= fmin) & (freqs_psds <= fmax)

                    #Promedio por canal
                    potencia = data_psds[:, mask].mean(axis=1)

                    #Agregar la info de la ventana y la banda
                    psds_bandas[banda][i_wind, :] = potencia

        #Recorrer las columnas
        for i_col in range(len(good_indices)):
            #Recorrer las bandas
            for banda in psds_bandas:

                #Extraer el vector columna
                vector_col = psds_bandas[banda][:, i_col]

                #Valores nan de ese vector columna
                valores_nan = np.isnan(vector_col)

                #Interpolar esos valores
                vector_col[valores_nan] = np.interp(
                    np.flatnonzero(valores_nan),
                    np.flatnonzero(~valores_nan),
                    vector_col[~valores_nan]
                )

                #Actualizar la matriz original
                psds_bandas[banda][:, i_col] = vector_col

        #======Agregar la columna vacía en caso de que sea necesario======
        #psds con la estructura correcta. Inicializado con valores nan
        psds_full = {
            banda: np.full((n_ventanas, n_channels), np.nan)
            for banda in psds_bandas
        }

        #Remplazar los nan por los valores correctos en las columnas que se tengan no marcadas como malas
        for banda in psds_bandas:
            psds_full[banda][:, good_indices] = psds_bandas[banda]

        psds_segmentos[nombre] = psds_full

    return psds_segmentos

def matriz_eventos(nombre_sujeto, df_tiempos):
    """
    Crea la matriz de eventos del sujeto necesaria para crear los segmentos
    :param nombre_sujeto: Nombre del sujeto para buscarlo como índice en el df
    :param df_tiempos: DataFrame con los tiempos para segmentar todos los sujetos
    :return: Matriz de eventos con la estructura de MNE
    """

    #Buscar la columna correspondiente
    tiempos_dp = df_tiempos.loc[nombre_sujeto, :].to_numpy()

    # Crear los códigos de los eventos
    codes_events = np.repeat(np.arange(1, len(tiempos_dp) // 2 + 1), 2)

    # Crear la estructura de los eventos de MNE
    events = np.column_stack([
        tiempos_dp,
        np.zeros(len(tiempos_dp), dtype=int),
        codes_events
    ]
    )

    return events


"""==========Aplicaciones de las funciones=========="""

#=========Generación de la base de datos de los promedios de potencia (procesar_eeg_neonatos)======

#Lista de los procedimientos
procedimientos = ["Nursing", "Intervention", "Post"]

#Lista de canales de EEG
canales = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2", "T7", "T8", "Cz"]

#Crear los nombres de las columnas
nombres_columnas = [proc + "_" + canal for proc in procedimientos for canal in canales]

#Base de datos para guardar el vector por sujeto para cada banda
base_datos = {
    banda: []
    for banda in bandas
}

#Archvivos sin información para hacer los cortes
no_tiempos = []

#Sujetos con información
nombres_sujetos = []

#Recorrer los archivos
for archivo in DATA_DIR_CLEAN.glob("*.fif"):
    try:
        info_archivo = procesar_eeg_neonatos(archivo, DICT_EVENTOS, frecuencias=bandas)

    #El archivo no se encuentra en el excel de tiempos
    except KeyError:
        no_tiempos.append(archivo.stem)
        continue

    else:
        nombres_sujetos.append(archivo.stem[:-10])
        for banda in base_datos:
            vector = np.concatenate((info_archivo[banda]["Nursing"], info_archivo[banda]["Intervention"],
                                     info_archivo[banda]["Post"]))
            base_datos[banda].append(vector)


#Crear la carpeta de resultados
DATA_DIR_RESULTS.mkdir(parents=True, exist_ok=True)

#Guardar la info en un excel
with pd.ExcelWriter(DATA_DIR_RESULTS / "supresion_potencias.xlsx") as writer:
    for banda, matriz in base_datos.items():
        df = pd.DataFrame(matriz, index=nombres_sujetos, columns=nombres_columnas)
        df.to_excel(writer, sheet_name=banda)



#========Creación de las series de tiempo==========================
series_potencia = {}
for archivo in DATA_DIR_CLEAN.glob("*.fif"):
    #Leer el archivo
    raw = mne.io.read_raw_fif(archivo, preload=True)

    #Extraer info de los canales de eeg
    all_channels = [ch for ch in raw.ch_names if raw.get_channel_types(picks=[ch])[0] == "eeg"]
    good_channels = [ch for ch in all_channels if ch not in raw.info['bads']]
    good_indices = [
        i for i, ch in enumerate(all_channels)
        if ch not in raw.info["bads"]
    ]

    #Cargar los eventos
    try:
        events = matriz_eventos(archivo.stem[:-10], df_tiempos)

    #Si no tiene tiempos, continuar con el siguiente sujeto
    except KeyError:
        continue

    #Crear los segmentos
    segmentos = crear_segmentos(raw, events, DICT_EVENTOS)

    #Crear la serie de potencia para el sujeto
    series_potencia[archivo.stem[:-10]] = serie_potencias_segmentos(segmentos, len(all_channels), good_indices,
                                                                    bandas, raw.info["sfreq"])

#Guardar las series de potencia
joblib.dump(series_potencia, DATA_DIR_RESULTS/"series_potencia.joblib")













