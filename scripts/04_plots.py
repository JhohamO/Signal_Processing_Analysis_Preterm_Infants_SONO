import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")

"""=============Constantes del código========================="""
#Definir las rutas
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR_RESULTS = DATA_DIR / "results"

#nombres de los canales
NAME_CHANNELS = [ "Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2", "T7", "T8", "Cz"]

#Leer la estructura con la información
series_potencia = joblib.load(DATA_DIR_RESULTS/"series_potencia.joblib")

#Leer la aleatorización
aleatorizacion = pd.read_excel(DATA_DIR/"aleatorizacion_hipotermia.xlsx")

#Evitar espacios innecesarios en los nombres de las columnas
aleatorizacion.columns = aleatorizacion.columns.str.strip()


"""=====================Definición de funciones=================================="""

def media_desv_baseline(series_potencia, nombre_baseline = "Baseline"):
    """
    Extraer la media y desviación de cada canal para cada sujeto en Baseline
    :param series_potencia: Series de potencia con la estructura Sujeto:Estimulo:Banda:Matriz (n_ventanas, n_channels)
    :param nombre_baseline: Nombre del estímulo usado como Baseline
    :return: Diccionario con estructura Sujeto:Banda: Canal:Vector [media, desviación]
    """

    #Diccionario a retornar
    med_desv = {}

    #Recorrer los sujetos de la serie
    for sujeto in series_potencia.keys():

        #Crear un diccionario vacio para el sujeto
        med_desv[sujeto] = {}

        #Recorrer las distintas bandas de frecuencia de baseline
        for banda, matriz in series_potencia[sujeto]["Baseline"].items():

            #Agregar la llave de la matriz
            med_desv[sujeto][banda] = {}

            #Recorrer las columnas de la matriz para extraer desv y media por canal
            for col in range(matriz.shape[1]):
                med_desv[sujeto][banda][col] = [np.mean(matriz[:, col]), np.std(matriz[:, col])]

    return med_desv

def z_score_potencias(series_potencia, media_desv_baselines):
    """
    Hacer el cálculo del z-score de las las series de potencia
    :param series_potencia: Diccionario con la serie de potencia original de forma Sujeto:Estimulo:Banda:Matriz (n_ventanas, n_channels)
    :param media_desv_baselines: Diccionario con los valores de media y desviación de cada línea base con forma Sujeto: Banda: Canal: Vector [media, std]
    :return: Misma estructura que series_potencia pero con lso valores transformados usando z-score
    """
    series_potencia_zscore = {}

    # Recorrer sujetos
    for sujeto in series_potencia.keys():
        series_potencia_zscore[sujeto] = {}

        # Recorrer estímulos
        for estimulo in series_potencia[sujeto].keys():
            series_potencia_zscore[sujeto][estimulo] = {}

            # Recorrer bandas
            for banda, matriz in series_potencia[sujeto][estimulo].items():
                matriz_zscore = matriz.copy()

                # Recorrer las columnas de la matriz para hacer la puntuación z
                for col in range(matriz.shape[1]):
                    matriz_zscore[:, col] = ((matriz[:, col] - media_desv_baselines[sujeto][banda][col][0]) /
                                             media_desv_baselines[sujeto][banda][col][1])

                # Agrego la matriz a la estrucutura
                series_potencia_zscore[sujeto][estimulo][banda] = matriz_zscore

    return series_potencia_zscore


def ajustar_info_plot(sujeto: str, par_sujeto:str, series_potencia_zscore: dict):
    """
    Acomodar la información para el ploteo haciendo 2 cosas: 1)Extrayendo la ubicación en número de ventanas de las línea verticales
    que representan los cambios de estímulo. 2) Concatenando la información de los segmentos en una única matriz
    :param sujeto: Nombre del sujeto, Ejemplo "P01_MT1"
    :param par_sujeto: Nombre del correspondiente archivo par, ejemplo "P01_MT2"
    :param series_potencia_zscore: Diccionario con las potencias con medidas zscore de la forma
    Sujeto:Estimulo:Banda -> Matriz (n_ventanas, n_channels)
    :return: Diccionario con la ubicación de las líneas verticales en ventanas de la forma sujeto: [ven1, ven2, ven3]
    Diccionario con las matrices de potencia de las distintas bandas de la forma sujeto:banda -> matriz
    """

    # Guardar ubicación para las líneas verticales
    lineas_vert = {
        sujeto: [],
        par_sujeto: [],
    }

    # Guardar los acumulados
    acumulado_sujeto = 0
    acumulado_par = 0

    # Guardar la información de las líneas verticales
    for estimulo in series_potencia_zscore[sujeto].keys():
        # Extraer las matrices de datos en la primera key de cualquier banda
        matriz_sujeto = series_potencia_zscore[sujeto][estimulo][next(iter(series_potencia_zscore[sujeto][estimulo]))]
        matriz_par = series_potencia_zscore[par_sujeto][estimulo][next(iter(series_potencia_zscore[sujeto][estimulo]))]

        # Sumar a los acumulados
        acumulado_sujeto += matriz_sujeto.shape[0]
        acumulado_par += matriz_par.shape[0]

        # Agregar los acumulados a la lista
        lineas_vert[sujeto].append(acumulado_sujeto)
        lineas_vert[par_sujeto].append(acumulado_par)

    # Eliminar la línea que corresponde al final, no es necesaria en la visualización
    lineas_vert[sujeto].pop()
    lineas_vert[par_sujeto].pop()

    # Crear listas vacías para cada banda
    serie_no_estimulos = {
        sujeto: {
            banda: []
            for banda in series_potencia_zscore[sujeto][next(iter(series_potencia_zscore[sujeto]))].keys()
        },
        par_sujeto: {
            banda: []
            for banda in series_potencia_zscore[sujeto][next(iter(series_potencia_zscore[sujeto]))].keys()}
    }

    # Recorrer los estímulos
    for estimulo in series_potencia_zscore[sujeto]:

        # Recorrer las bandas
        for banda, matriz in series_potencia_zscore[sujeto][estimulo].items():
            # Agregar la matriz del sujeto
            serie_no_estimulos[sujeto][banda].append(matriz)

            # Matriz del sujeto par
            matriz_par = series_potencia_zscore[par_sujeto][estimulo][banda]
            serie_no_estimulos[par_sujeto][banda].append(matriz_par)

    # Matrices de las series
    matrices_series_bandas = {
        sujeto: {}
        for sujeto in serie_no_estimulos.keys()
    }

    # Concatenar las listas en matrices
    for key in serie_no_estimulos:
        for banda, listas in serie_no_estimulos[key].items():
            matrices_series_bandas[key][banda] = np.vstack(listas)

    return lineas_vert, matrices_series_bandas

def extraer_media_desv_poblacional(minimos, series_potencia_zscore, aleatorizacion):
    """
    Crear 2 matrices (una de medias, otra de desviaciones) de la forma (n_ventanas, n_channels) donde cada item corresponde a la media
    (o desviación) de de todos los sujetos en esa determinada ventana y en ese determinado canal para cada condición (MT y control)
    :param minimos: Iterable con el tamaño mínimo de cada estpimulo en ventanas para ser tenido en cuenta
    :param series_potencia_zscore: Diccionario de forma Sujeto:Estímulo:Banda:Matriz (n_ventanas, n_channels) con las potencias
    en zscore value de cada ventana
    :param aleatorizacion: DataFrame donde se define la condición de aleatorización
    :return: Diccionario de la forma Condición:Banda:Media/Desv:Matriz (n_ventanas, n_channels)
    """

    #========Recorrer los sujetos para extraer información============
    #Definir el índice del último sujeto a recorrer
    n_max = int(sorted(series_potencia_zscore.keys())[-1][1:3])

    # Crear un diccionario para guardar las matrices de información de cada sujeto en cada banda
    series_completas = {}

    # Series para los sujetos bajo MT y control
    series_MT = {}
    series_control = {}

    # Recorrer los índices
    for n in range(1, n_max + 1):

        # Crear el número del sujeto
        numero = "0" + str(n) if n < 10 else str(n)

        # Crear los nombres de los archivos
        sujeto = "P" + numero + "_MT1"
        par_sujeto = "P" + numero + "_MT2"

        # Si no se encuentran ambos en las series, saltar al siguiente sujeto
        if sujeto not in series_potencia_zscore.keys() or par_sujeto not in series_potencia_zscore.keys():
            continue

        #Extraer el número de ventanas de cada estímulo para cada sujeto
        lens_sujeto = []
        lens_par_sujeto = []

        for estimulo in series_potencia_zscore[sujeto]:
            v_sujeto = next(iter(series_potencia_zscore[sujeto][estimulo].values())).shape[0]
            v_par_sujeto = next(iter(series_potencia_zscore[par_sujeto][estimulo].values())).shape[0]
            lens_sujeto.append(v_sujeto)
            lens_par_sujeto.append(v_par_sujeto)

        lens_sujeto = np.array(lens_sujeto)
        lens_par_sujeto = np.array(lens_par_sujeto)

        # Si el sujeto es más corto que los mínimos, se salta también
        if not np.all(minimos <= lens_sujeto) or not np.all(minimos <= lens_par_sujeto):
            continue

        # Extraer la condición de aleatorización
        condicion = aleatorizacion.iloc[int(sujeto[1:3]) - 1, 1]

        # =======Extraer las matrices de deatos de cada sujeto en cada estímulo pero cortando hasta el límite=======
        # Bandas de potencia con las que se cuenta
        bandas = next(iter(series_potencia_zscore[sujeto].values()))

        # Crear la estructura para el sujeto y su par en las series
        series_completas[sujeto] = {
            banda: [] for banda in bandas
        }

        series_completas[par_sujeto] = {
            banda: [] for banda in bandas
        }

        # Llenar las listas recorriendo los estímulos y las bandas
        for n_est, (estimulo, bandas) in enumerate(series_potencia_zscore[sujeto].items()):

            for banda, matriz in bandas.items():
                # Agregar la matriz cortada a cada lista anterior
                series_completas[sujeto][banda].append(matriz[:minimos[n_est], :])

                matriz_par = series_potencia_zscore[par_sujeto][estimulo][banda][:minimos[n_est], :]
                series_completas[par_sujeto][banda].append(matriz_par)

        # Llenar la correspondiente serie de cada MT o de control dependiendo de la condición de aleatorización
        if condicion == 1:
            series_MT[sujeto] = series_completas[sujeto]
            series_control[par_sujeto] = series_completas[par_sujeto]

        else:
            series_MT[par_sujeto] = series_completas[par_sujeto]
            series_control[sujeto] = series_completas[sujeto]

    # Concatenar verticalmente las matrices de cada banda. i.e Unir las series de tiempo de los distintos estímulos
    for sujeto, bandas in series_MT.items():
        for banda, matrices in bandas.items():
            series_MT[sujeto][banda] = np.vstack(matrices)

    for sujeto, bandas in series_control.items():
        for banda, matrices in bandas.items():
            series_control[sujeto][banda] = np.vstack(matrices)

    #========Apilar las matrices de cada sujeto usando un 3ra dimensión (n_ventanas, n_channels, n_sujetos)======

    #La matriz de 3 dimensiones se calcula por banda independiente
    apiladas_MT = {
        banda: []
        for banda in bandas
    }

    apiladas_control = {
        banda: []
        for banda in bandas
    }

    # Recorrer los sujetos MT y control
    for sujeto, bandas in series_MT.items():
        for banda, matriz in bandas.items():
            apiladas_MT[banda].append(matriz)

    for sujeto, bandas in series_control.items():
        for banda, matriz in bandas.items():
            apiladas_control[banda].append(matriz)

    # Apilar la lista para quedar con un matriz de dimensiones (n_ventanas_total, n_channels_n_sujetos)
    for banda in apiladas_MT:
        apiladas_MT[banda] = np.stack(apiladas_MT[banda], axis=2)
        apiladas_control[banda] = np.stack(apiladas_control[banda], axis=2)

    # Crear la estructura para guardar la media y desviación de cada punto de la serie
    series_promedio = {
        "MT": {
            banda: {"media": None, "std": None}
            for banda in bandas
        },
        "Control": {
            banda: {"media": None, "std": None}
            for banda in bandas
        }
    }

    # Recorrer las matrices apiladas para extraer la media y desviación de cada canal en cada punto
    for banda in apiladas_MT:
        series_promedio["MT"][banda]["media"] = np.nanmean(apiladas_MT[banda], axis=2)
        series_promedio["MT"][banda]["std"] = np.nanstd(apiladas_MT[banda], axis=2)

        series_promedio["Control"][banda]["media"] = np.nanmean(apiladas_control[banda], axis=2)
        series_promedio["Control"][banda]["std"] = np.nanstd(apiladas_control[banda], axis=2)

    return series_promedio

def plot_banda(matriz_1,matriz_2,sujeto_1,sujeto_2,banda,canales, condicion
               ,ventanas_lineas=None, ventana_suavizado=None, guardar=False, carpeta_guardado=None):
    """
    Grafica una banda de potencia para dos sujetos.

    Cada subplot tiene su propio eje X, ya que los sujetos
    pueden tener diferente número de ventanas.

    Parameters
    ----------
    matriz_1, matriz_2 : np.ndarray
        Matrices de potencia con dimensiones
        (n_ventanas, n_canales).

    sujeto_1, sujeto_2 : str
        Identificadores de los sujetos.

    banda : str
        Nombre de la banda de frecuencia.

    canales : list
        Nombres de los canales EEG.

    ventanas_lineas : dict, optional
        Diccionario con los tiempos, en datapoints en los que
        se deben colocar las líneas verticales para cada sujeto.
    """

    # Crear dos subplots independientes
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 9),sharex=False)

    # Crear una paleta con un color por canal
    colores = plt.cm.tab20(
        np.linspace(0, 1, len(canales))
    )

    # Asociar cada subplot con su sujeto y matriz dependiendo de la condición
    #Si condición es 1 significa que MT1 fue músicoterapia y MT2 control.
    #Ajustar el orden según la aleatorización, para que la condición de MT siempre se grafique en la parte superior
    if condicion == 1:
        sujetos = [sujeto_1, sujeto_2]
        matrices = [matriz_1, matriz_2]
    else:
        sujetos = [sujeto_2, sujeto_1]
        matrices = [matriz_2, matriz_1]

    #Teniendo en cuenta que músicoterapia siempre se grafica en la parte superior, se definen los títulos de las gráficas
    titulos = ["Músicoterapia", "Control"]

    # Recorrer los dos sujetos
    for i_sujeto, (ax, sujeto, matriz) in enumerate(zip(axes, sujetos, matrices)):

        # Crear eje X usando el índice de las ventanas
        ventanas = np.arange(matriz.shape[0])

        #Crear el vector con los valores en tiempo en minutos. Cada ventana tiene pasos de 1.5 s.
        minutos = ventanas * 1.5 / 60


        # Identificar columnas que no sean completamente NaN
        columnas_validas = ~np.all(
            np.isnan(matriz),
            axis=0
        )

        # Graficar cada canal
        for canal_idx, canal in enumerate(canales):

            # Ignorar el canal si toda la columna es NaN
            if not columnas_validas[canal_idx]:
                continue

            # Extraer la serie
            serie = matriz[:, canal_idx]

            # Aplicar media móvil si se especificó una ventana
            if ventana_suavizado is not None:
                serie = (
                    pd.Series(serie)
                    .rolling(
                        window=ventana_suavizado,
                        center=True,
                        min_periods=1
                    )
                    .mean()
                    .to_numpy()
                )

            # Plotear la serie
            ax.plot(
                minutos,
                serie,
                color=colores[canal_idx],
                alpha=0.7,
                linewidth=1,
                label=canal
            )

        # Agregar las líneas verticales correspondientes
        if ventanas_lineas is not None:

            #Posiciones de las líneas verticales
            posiciones = ventanas_lineas.get(sujeto, [])

            for ventana in posiciones:
                ax.axvline(
                    x=ventana * 1.5 / 60,   #Convertir el número de ventana a un datapoint
                    linestyle="--",
                    linewidth=1
                )

            #Agregar etiquetas de las fases
            if len(posiciones) == 3:
                etiquetas = ["Baseline", "Nursing", "Intervention", "Post"]

                #Inicios de las secciones para ubicar la etiqueta
                inicios = [0] + posiciones

                #Colocar cada etiqueta al inicio del intervalo
                for i, etiqueta in enumerate(etiquetas):
                    ax.text(
                        (inicios[i] + 2) * 1.5 / 60,
                        0.95,
                        etiqueta,
                        transform=ax.get_xaxis_transform(),
                        ha= "left",
                        va= "top",
                        fontsize=11
                    )

        # Configuración del subplot
        ax.set_title(titulos[i_sujeto])
        ax.set_ylabel("Potencia (z-score)")
        ax.set_xlabel("Tiempo (min)")

        ax.legend(
            loc="upper right",
            ncol=2
        )

        ax.grid(alpha=0.2)

    # Título general
    fig.suptitle(
        f"Banda {banda} - {sujeto_1[:3]}",
        fontsize=14
    )

    plt.tight_layout()

    #Guardado
    if guardar:
        #Pasar la carpeta a un objeto Path
        carpeta = ROOT / Path(carpeta_guardado)

        #Crearla
        carpeta.mkdir(parents=True, exist_ok=True)

        nombre_archivo = f"{sujeto[:3]}_{banda}.png"
        fig.savefig(carpeta / nombre_archivo, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

def plotear_promedio_grupal(series_promedio, finales_est, etiquetas, ventana_suavizado=None):
    """
    Plotea las línea de tiempo medias y banda de dispersión de cada banda de frecuencia
    :param series_promedio: Diccionario de la forma Condición: Banda: Medida (media o desviación): Matriz (n_ventanas, n_canales)
    :param finales_est: Valores en ventanas que corresponden al final de cada estímulo (sin contar el último)
    etiquetas: Lista de etiquetas de cada zona del gráfico
    ventana_suavizado: Tamaño de la ventana para suavizar las series de tiempo
    :return:
    """

    #Ubicar los inicios de cada estímulo para ubicar posteriormenete el nombre de cada sección
    inicios = [0] + list(finales_est)

    # Definir el eje x del plot. Primero en ventanas
    ventanas = np.arange(next(iter(series_promedio["MT"].values()))["media"].shape[0])

    #Ahora convertido a minutos
    minutos = ventanas * 1.5 / 60

    # Recorrer las condiciones
    for intervencion, bandas in series_promedio.items():

        # Crear el gráfico para cada condición
        fig, axes = plt.subplots(len(series_promedio[intervencion]), 1, figsize=(12, 12), sharex=True,
                                 gridspec_kw= dict(hspace=0))

        # Seleccionar la paleta de colores
        cmap = plt.get_cmap("viridis")
        colores = cmap(np.linspace(0.1, 0.9, len(NAME_CHANNELS)))

        # Recorrer las bandas
        for ax, banda in zip(axes, bandas.keys()):

            # Extraer las matrices correspondientes para la banda
            medias = series_promedio[intervencion][banda]["media"]
            stds = series_promedio[intervencion][banda]["std"]

            # Recorrer los canales para plotearlos
            for n_chan, channel in enumerate(NAME_CHANNELS):
                # Media y desviación del canal
                media = medias[:, n_chan]
                std = stds[:, n_chan]

                if ventana_suavizado:
                    media = (pd.Series(media).rolling(
                        window=ventana_suavizado,
                        center=True,
                        min_periods=1
                    ).mean().to_numpy())

                    std = (pd.Series(std).rolling(
                        window=ventana_suavizado,
                        center=True,
                        min_periods=1
                    ).mean().to_numpy())

                # Plotear la media
                ax.plot(minutos, media, color=colores[n_chan], label=channel)

                # Plotear la banda de dispersión
                ax.fill_between(
                    minutos,
                    media - std,
                    media + std,
                    color=colores[n_chan],
                    alpha=0.15
                )

            # Agregar las líneas verticales de división de los estímulos
            for final in finales_est:
                ax.axvline(
                    x=final * 1.5 / 60,
                    linestyle="--",
                    linewidth=1
                )

            for i_eti, etiqueta in enumerate(etiquetas):
                ax.text(
                    (inicios[i_eti] + 2) * 1.5 / 60,
                    0.95,
                    etiqueta,
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="top",
                    fontsize=11
                )
            #Remover las lineas de separación de los subplots
            # for spine in ['top', 'right', 'bottom', 'left']:
            #     ax.spines[spine].set_visible(False)
            # ax.yaxis.tick_right()
            ax.set_ylabel(banda)

        axes[-1].set_xlabel("Tiempo")

        # Titular las gráficas
        titulo = intervencion

        # Arreglar el nombre de la intervención de músico terapia
        if titulo == "MT":
            titulo = "Music therapy"

        # Agregar el título a la gráfica
        fig.suptitle(titulo)

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower right",
            ncol=6
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()


"""============USAR LAS FUNCIONES=================="""
#Extracción de la media y desviación de Baseline de cada sujeto en cada banda
media_desviacion = media_desv_baseline(series_potencia)

#Transformación z de las frecuencias teniendo en cuenta media y desviación de Baseline
series_potencia_zscore = z_score_potencias(series_potencia, media_desviacion)


"""Plots grupales"""

#Etiquetas de las zonas a nombrar
etiquetas = ["Baseline", "Nursing", "Intervention", "Post"]

#Definir los mínimos por ventana
#LB 4.9218 min, #Nursing 3.25 min, Intervention 9.31 min y Post 4.12 min
minimos = [series_potencia_zscore["P01_MT1"]["Baseline"][next(iter(series_potencia_zscore["P01_MT1"]["Baseline"].keys()))].shape[0],
           series_potencia_zscore["P12_MT1"]["Nursing"][next(iter(series_potencia_zscore["P12_MT1"]["Nursing"].keys()))].shape[0],
           series_potencia_zscore["P11_MT2"]["Intervention"][next(iter(series_potencia_zscore["P11_MT2"]["Intervention"].keys()))].shape[0],
           series_potencia_zscore["P13_MT1"]["Post"][next(iter(series_potencia_zscore["P13_MT1"]["Post"].keys()))].shape[0]]

#El valor de nusring se puede variar para tener en cuenta más datos, pero menos sujetos

#Los valores mínimos son también los valores para las líneas verticales del gráfico
finales_est = np.cumsum(minimos)[:-1]

#Extarer las series de tiempo de potencia media y la desviación de cada serie de tiempo
series_promedio = extraer_media_desv_poblacional(minimos, series_potencia_zscore, aleatorizacion)

plotear_promedio_grupal(series_promedio, finales_est, etiquetas, ventana_suavizado=5)

# TODO: (Posibles)
#  - Hacer np.clip sobre los datos originales a valores muy extremos (definir límite)
#  - Hacer los gráficos promedios comparanndo cada banda en MT vs Control
#  - Definir un eje y fijo para todos lo gráficos grupales

"""Plots individuales"""

#Recorrer de 1 a 15 para ir creando los gráficos
for n in range(1, 16):
    numero = "0" + str(n) if n < 10 else str(n)

    #Crear los nombres de los archivos del sujeto
    sujeto = "P" + numero + "_MT1"
    par_sujeto = "P" + numero + "_MT2"

    #Si no están ambos, lo salto
    if sujeto not in media_desviacion.keys() or par_sujeto not in media_desviacion.keys():
        continue

    #Si sí están:

    #Extraer la condición de aleatorización. Si el valor es 1, MT1 corresponde músicoterapia y MT2 a control. cero es lo contrario
    condicion = aleatorizacion.iloc[int(sujeto[1:3]) - 1, 1]

    #Extraer la info para plotear a ambos sujetos
    lineas_vert, matrices_series_bandas = ajustar_info_plot(sujeto, par_sujeto, series_potencia_zscore)

    #Plotear la info de todas las bandas
    for banda in matrices_series_bandas[sujeto]:
        matriz_1 = matrices_series_bandas[sujeto][banda]
        matriz_2 = matrices_series_bandas[par_sujeto][banda]

        plot_banda(
            matriz_1=matriz_1,
            matriz_2=matriz_2,
            sujeto_1=sujeto,
            sujeto_2=par_sujeto,
            banda=banda,
            canales=NAME_CHANNELS,
            ventanas_lineas=lineas_vert,
            condicion=condicion,
            ventana_suavizado=5,
            guardar=True,
            carpeta_guardado="figures/single_subjetcs"
        )









