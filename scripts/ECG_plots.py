import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from decorator import append

"""========Definir las rutas y constantes============"""
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = ROOT / "figures"

#Serie de frecuencia cardiaca
serie_hr = joblib.load(RESULTS_DIR / "series_hr.joblib")

#Condiciones de aleatorización
aleatorizacion = pd.read_excel(DATA_DIR / "aleatorizacion_hipotermia.xlsx")

#Corregir los nombres de las columnas
aleatorizacion.columns = aleatorizacion.columns.str.strip()

"""=======Funciones==========="""
#Extraer la media y desviación de cada sujeto

def med_desv_hr(serie_hr, estimulo_baseline="Baseline"):
    """
    Extraer la media y desviación de HR de cada uno de los sujetos
    :param serie_hr: Diccionario con las series de tiempo de HR para cada sujeto y cada estímulo
    :param estimulo_baseline: Nombre del estimulo de baseline para cada sujeto
    :return: Diccionario de la forma sujeto:[media, desviación]
    """

    #Diccionario vacío
    med_dev = {}

    #Recorrer los sujetos
    for sujeto in serie_hr:

        #Extraer el vector de HR correspondiente a baseline
        vector_baseline = serie_hr[sujeto][estimulo_baseline]

        #Calcular la media y desviación estándar del vector
        med_dev[sujeto] = np.array([np.mean(vector_baseline), np.std(vector_baseline)])

    return med_dev



def z_score_hr(serie_hr, med_desv):
    """
    Calcular las series pero transformadas a un zscore usando la media y desviación de LB
    :param serie_hr: Serie original de la forma sujeto:estimulo:vector hr
    :param med_desv: Serie con la desviación y media de línea base para cada sujeto
    :return: Diccionario con la misma forma de serie_hr pero transformado a zscore
    """

    #Crear el diccionario a retornar vacío
    serie_zscore = {}

    #Recorrer los sujetos de la serie
    for sujeto, estimulos in serie_hr.items():

        #Crear la llave del sujeto
        serie_zscore[sujeto] = {}

        #Recorrer los estímulos y su vector correspondiente
        for estimulo, vector in estimulos.items():

            #Para cada estímulo, calcular su zscore usando la media y desviación de baseline del sujeto
            serie_zscore[sujeto][estimulo] = (vector - med_desv[sujeto][0]) / med_desv[sujeto][1]

    return serie_zscore


#Ajustar la serie de z-score para los ploteos individuales
def extraer_lineas_verticales_sujeto(sujeto: str, serie_zscore):
    """
    Extraer las líneas verticales del plot del sujeto que indican los finales de cada estímulo
    :param sujeto: Nombre del sujeto tal como se encuentra en serie_zscore
    :param serie_zscore: serie de potencias normalizadas
    :return: lista con el índice del final de estímulo
    """
    #Crear el vector de salida
    lineas_verticales = []

    #Crear el acumulador para ir sumando los tamaños de cada estímulo
    acumulador = 0

    #Recorrer los estímulos del sujeto
    for estimulo, vector in serie_zscore[sujeto].items():

        #Extraer el tamaño del vector
        tamaño = len(vector)

        #Sumarlo al acumulador
        acumulador += tamaño

        #Agregarlo a la lista
        lineas_verticales.append(acumulador)

    #La última línea vertical coincide con el final, por lo que no es necesaria
    lineas_verticales.pop()

    return lineas_verticales

def concatenar_serie_estimulos(sujeto:str, serie_zscore):
    """
    Crea una único vector para eñ sujeto concatenando todos los vectores de los estímulos
    :param sujeto: Nombre del sujeto como se encientrea en serie_zscore
    :param serie_zscore: Diccionario con los valores a concatenar
    :return: Un vector para cada sujeto con su serie de tiempo de HR
    """
    #Lista vacía para el vector del sujeto
    lista_sujeto = []

    #Recorrer los vectores correspondientes a cada estímulo
    for vector in serie_zscore[sujeto].values():

        #Extender la lista con el vector
        lista_sujeto.extend(vector)

    return np.array(lista_sujeto)


def plotear_par(sujeto, par_sujeto, condicion, serie_zscore, carpeta_guardado=None,
                ventana_suavizado = None):
    """
    Plotear la línea de tiempo de HR en control y en musicoterapia
    :param sujeto: Nombre del archivo MT1 del sujeto
    :param par_sujeto: Nombre del archivo MT2 del sujeto
    :param condicion: Condición de aleatorización que define cuál de los 2 archivos corresponde a MT o a control
    :param serie_zscore: Diccionario con las series de HR correspondiente
    :param carpeta_guardado: Ruta de la carpeta para guardar los archivos
    :param ventana_suavizado: Tamaño de la ventana para suavizar la serie
    :return:
    """
    # Si condición es 1, MT1 es musicoterapia. Si es 0, MT2 es musicoterapia
    if condicion == 1:
        sujetos = [sujeto, par_sujeto]
    else:
        sujetos = [par_sujeto, sujeto]

    # Crear dos subplots independientes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), sharex=False)

    # Títulos de cada gráfico
    titulos = ["Musicoterapia", "Control"]

    # Recorrer los ejes para plotear
    for n_ax, ax in enumerate(axes):

        # Extraer la info necesaria para el gráfico del archivo
        lineas_verticales = extraer_lineas_verticales_sujeto(sujetos[n_ax], serie_zscore)
        serie_hr_sujeto = concatenar_serie_estimulos(sujetos[n_ax], serie_zscore)

        #Si está el parámetro de suavizado
        if ventana_suavizado:
            serie_hr_sujeto = (pd.Series(serie_hr_sujeto).rolling(
                window=ventana_suavizado,
                center=True,
                min_periods=1
            ).mean().to_numpy())

        # Crear el eje x en datapointa
        datapoints = np.arange(len(serie_hr_sujeto))

        # Convertir a minutos
        minutos = datapoints / (250 * 60)

        # Plotear la serie
        ax.plot(
            minutos,
            serie_hr_sujeto,
            linewidth=1
        )

        # Agregar las líneas verticales
        for linea_vertical in lineas_verticales:
            ax.axvline(
                x=linea_vertical / (250 * 60),
                linestyle="--",
                linewidth=1
            )

        # Agregar etiquetas de las zonas
        inicios = [0] + lineas_verticales

        for i_est, estimulo in enumerate(estimulos):
            ax.text(
                (inicios[i_est] + 5) / (250 * 60),
                0.95,
                estimulo,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=11
            )

        ax.set_title(titulos[n_ax])
        ax.set_ylabel("HR - zscore")
        ax.set_xlabel("Tiempo (min)")

        ax.grid(alpha=0.2)

    fig.suptitle(sujetos[0][:3], fontsize=14)
    plt.tight_layout()

    if carpeta_guardado:
        # Crear la carpeta en caso que no exista
        carpeta = FIGURES_DIR / carpeta_guardado
        carpeta.mkdir(parents=True, exist_ok=True)

        # Nombre del archivo
        nombre_archivo = f"{sujeto[:3]}_HR.png"

        # Ruta de salida
        output_path = carpeta / nombre_archivo

        # Guardar la figura
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def _recortar_y_concatenar(estimulos_sujeto, minimos):
    """
    Corta cada estímulo a su tamaño mínimo y los concatena en un único vector.
    :param estimulos_sujeto: Diccionario estímulo: vector de un archivo
    :param minimos: Diccionario estímulo: tamaño mínimo
    :return: np.ndarray de largo sum(minimos.values())
    """
    return np.concatenate([
        np.asarray(estimulos_sujeto[estimulo])[:minimo]
        for estimulo, minimo in minimos.items()
    ])


def media_desv_poblacion(serie_zscore, aleatorizacion, minimos):
    """
    Media y desviación de todos los sujetos en cada punto en el tiempo, por condición.
    :param serie_zscore: Diccionario sujeto: estímulo: vector
    :param aleatorizacion: DataFrame con la condición de aleatorización de cada sujeto
    :param minimos: Diccionario estímulo: tamaño mínimo (en muestras)
    :return: Diccionario condición: {"mean": vector, "std": vector, "n": int}
    """
    series = {"Musicoterapia": [], "Control": []}

    # Prefijos únicos de sujeto ("P01", "P02", ...), sin depender de un índice máximo
    ids_sujetos = sorted({nombre[:3] for nombre in serie_zscore})

    for id_sujeto in ids_sujetos:
        mt1 = f"{id_sujeto}_MT1"
        mt2 = f"{id_sujeto}_MT2"

        # Deben estar los dos archivos
        if mt1 not in serie_zscore or mt2 not in serie_zscore:
            continue

        # Estímulos que no alcanzan el largo mínimo (o que no existen)
        saltar = [
            f"{nombre}:{estimulo}"
            for nombre in (mt1, mt2)
            for estimulo, minimo in minimos.items()
            if len(serie_zscore[nombre].get(estimulo, [])) < minimo
        ]

        if saltar:
            print(f"Se salta {id_sujeto} debido a: {saltar}")
            continue

        # Asignar cada archivo a su condición
        condicion = aleatorizacion.iloc[int(id_sujeto[1:]) - 1, 1]
        archivo_mt, archivo_control = (mt1, mt2) if condicion == 1 else (mt2, mt1)

        series["Musicoterapia"].append(_recortar_y_concatenar(serie_zscore[archivo_mt], minimos))
        series["Control"].append(_recortar_y_concatenar(serie_zscore[archivo_control], minimos))

    # Apilar y calcular estadísticos por punto en el tiempo
    resultado = {}
    for nombre_condicion, lista_vectores in series.items():
        if not lista_vectores:
            raise ValueError(f"Ningún sujeto cumplió los mínimos en {nombre_condicion}")

        matriz = np.stack(lista_vectores, axis=0)  # (n_sujetos, n_puntos)

        resultado[nombre_condicion] = {
            "mean": np.nanmean(matriz, axis=0),
            "std": np.nanstd(matriz, axis=0, ddof=1),
            "n": matriz.shape[0],
        }

    return resultado

def _suavizar(vector, ventana):
    """Media móvil centrada. Si no hay ventana, devuelve el vector sin cambios."""
    if not ventana:
        return vector
    return (pd.Series(vector)
            .rolling(window=ventana, center=True, min_periods=1)
            .mean()
            .to_numpy())


def plotear_promedio_grupal_hr(series_promedio, finales_est, etiquetas, fs=250,
                               ventana_suavizado=None, dispersion="std",
                               ylabel="HR - zscore", carpeta_guardado=None,
                               nombre_archivo="HR_promedio_grupal.png"):
    """
    Plotea la media grupal de HR en MT y control, superpuestas, con su banda de dispersión.
    :param series_promedio: Diccionario condición: {"mean": vector, "std": vector, "n": int}
    :param finales_est: Índices (en muestras) del final de cada estímulo, sin contar el último
    :param etiquetas: Etiquetas de cada zona del gráfico (una por estímulo)
    :param fs: Frecuencia de muestreo en Hz, para convertir muestras a minutos
    :param ventana_suavizado: Tamaño (en muestras) de la ventana de suavizado
    :param dispersion: "std" (desviación estándar) o "sem" (error estándar de la media)
    :param ylabel: Etiqueta del eje y
    :param carpeta_guardado: Carpeta (relativa a ROOT) para guardar la imagen
    :param nombre_archivo: Nombre del archivo de salida
    """
    if dispersion not in ("std", "sem"):
        raise ValueError("dispersion debe ser 'std' o 'sem'")

    # Inicios de cada estímulo, para ubicar las etiquetas
    inicios = [0] + list(finales_est)

    # Eje x: muestras -> minutos
    n_puntos = len(next(iter(series_promedio.values()))["mean"])
    minutos = np.arange(n_puntos) / (fs * 60)

    fig, ax = plt.subplots(figsize=(14, 6))

    for condicion, stats in series_promedio.items():
        media = stats["mean"]
        banda = stats["std"]

        if dispersion == "sem":
            banda = banda / np.sqrt(stats["n"])

        media = _suavizar(media, ventana_suavizado)
        banda = _suavizar(banda, ventana_suavizado)

        linea, = ax.plot(minutos, media, linewidth=1.5,
                         label=f"{condicion} (n={stats['n']})")

        # Banda con el mismo color de la línea
        ax.fill_between(minutos, media - banda, media + banda,
                        color=linea.get_color(), alpha=0.15)

    # Líneas verticales de división de estímulos
    for final in finales_est:
        ax.axvline(x=final / (fs * 60), color="k", linestyle="--", linewidth=1)

    # Etiquetas de cada zona
    for inicio, etiqueta in zip(inicios, etiquetas):
        ax.text(
            (inicio + 5) / (fs * 60),
            0.95,
            etiqueta,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=11
        )

    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if carpeta_guardado:
        carpeta = ROOT / carpeta_guardado
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / nombre_archivo, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

def recortar_par_al_minimo(serie_mt, serie_control):
    """
    Recorta cada estímulo de ambos archivos al menor largo entre los dos y los concatena.
    :param serie_mt: Diccionario estímulo: vector del archivo de musicoterapia
    :param serie_control: Diccionario estímulo: vector del archivo control
    :return: (vector_mt, vector_control, minimos), con minimos como diccionario estímulo: largo
    """
    # Solo estímulos presentes en ambos, en el orden del archivo de MT
    estimulos = [e for e in serie_mt if e in serie_control]

    minimos = {e: min(len(serie_mt[e]), len(serie_control[e])) for e in estimulos}

    vector_mt = np.concatenate([np.asarray(serie_mt[e])[:minimos[e]] for e in estimulos])
    vector_control = np.concatenate([np.asarray(serie_control[e])[:minimos[e]] for e in estimulos])

    return vector_mt, vector_control, minimos


def plotear_par_superpuesto(sujeto, par_sujeto, condicion, serie_zscore, fs=250,
                            carpeta_guardado=None, ventana_suavizado=None):
    """
    Plotea HR de musicoterapia y control del mismo sujeto en un único gráfico,
    recortando cada estímulo al mínimo entre ambas condiciones.
    :param sujeto: Nombre del archivo MT1 del sujeto
    :param par_sujeto: Nombre del archivo MT2 del sujeto
    :param condicion: 1 si MT1 es musicoterapia, 0 si lo es MT2
    :param serie_zscore: Diccionario sujeto: estímulo: vector
    :param fs: Frecuencia de muestreo en Hz
    :param carpeta_guardado: Carpeta (relativa a FIGURES_DIR) para guardar la imagen
    :param ventana_suavizado: Tamaño (en muestras) de la ventana de suavizado
    """
    archivo_mt, archivo_control = (sujeto, par_sujeto) if condicion == 1 else (par_sujeto, sujeto)

    vector_mt, vector_control, minimos = recortar_par_al_minimo(
        serie_zscore[archivo_mt], serie_zscore[archivo_control]
    )

    vector_mt = _suavizar(vector_mt, ventana_suavizado)
    vector_control = _suavizar(vector_control, ventana_suavizado)

    minutos = np.arange(len(vector_mt)) / (fs * 60)

    # Límites y etiquetas salen de los mínimos, no de una variable global
    finales = np.cumsum(list(minimos.values()))
    inicios = np.concatenate([[0], finales[:-1]])

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(minutos, vector_mt, linewidth=1.2, label="Musicoterapia")
    ax.plot(minutos, vector_control, linewidth=1.2, label="Control")

    # Líneas verticales (sin la última, que coincide con el final)
    for final in finales[:-1]:
        ax.axvline(x=final / (fs * 60), color="k", linestyle="--", linewidth=1)

    for inicio, estimulo in zip(inicios, minimos):
        ax.text(
            (inicio + 5) / (fs * 60),
            0.95,
            estimulo,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=11
        )

    ax.set_title(sujeto[:3], fontsize=14)
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("HR - zscore")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if carpeta_guardado:
        carpeta = FIGURES_DIR / carpeta_guardado
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / f"{sujeto[:3]}_HR_superpuesto.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

"""USAR LAS FUNCIONES"""

#=============Plots individuales================

#Extraer la media y desviación de baseline para todos los sujetos
med_dev = med_desv_hr(serie_hr)

#Hacer la trasnformación z de las series usando media y desviación de línea base
serie_zscore = z_score_hr(serie_hr, med_dev)

#Estimulos
estimulos = ["Baseline", "Nursing", "Intervention", "Post"]

#Recorrer los índices de los sujetos
for i in range(1, 18):

    #Crear el número correspondiente al sujeto
    numero = "0" + str(i) if i < 10 else str(i)

    #Crear los nombres de los 2 archivos del sujeto
    sujeto = f"P{numero}_MT1"
    par_sujeto = f"P{numero}_MT2"

    #Si no se encuentran los 2 salto
    if sujeto not in serie_zscore or par_sujeto not in serie_zscore:
        continue

    #Si se encuentran los 2 defino la condición de aleatorización
    condicion = aleatorizacion.iloc[i-1, 1]

    #Plotear
    plotear_par(sujeto, par_sujeto, condicion, serie_zscore, carpeta_guardado="HR_individual", ventana_suavizado=500)
    plotear_par_superpuesto(sujeto, par_sujeto, condicion, serie_zscore,
                            carpeta_guardado="HR_individual_superpuesto", ventana_suavizado=500)

#==========Plot grupal===============
#Definir los mínimos
minimos = {
    "Baseline": len(serie_zscore["P01_MT1"]["Baseline"]),
    "Nursing" : len(serie_zscore["P12_MT1"]["Nursing"]),
    "Intervention": len(serie_zscore["P11_MT2"]["Intervention"]),
    "Post": len(serie_zscore["P13_MT1"]["Post"])
}

series_promedio = media_desv_poblacion(serie_zscore, aleatorizacion, minimos)

finales_est = np.cumsum(list(minimos.values()))[:-1]

plotear_promedio_grupal_hr(
    series_promedio, finales_est, etiquetas=list(minimos.keys()),
    ventana_suavizado=500, carpeta_guardado="figures/HR_grupal"
)
#TODO: -Agregar el guardado
# -Hacer la gráfica grupal,
