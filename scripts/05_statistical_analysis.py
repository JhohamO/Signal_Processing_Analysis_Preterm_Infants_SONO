import pandas as pd
import numpy as np
import seaborn as sns
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

#Definir las rutas
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = DATA / "results"
FIGURES = ROOT / "figures"

#Leer el archivo de aleatorización
aleatorizacion = pd.read_excel(DATA / "aleatorizacion_hipotermia.xlsx", index_col=0)

#Evitar espacios innecesarios en los nombres de las columnas
aleatorizacion.columns = aleatorizacion.columns.str.strip()


#Función para agregar la intervención correspondiente a cada sujeto
def agregar_col_intervencion(aleatorizacion, base_datos, n_inds=18):
    """
    Agrega una columna a cada base de datos que aclara qué intervención representa esa fila y agrega columna del
    nombre del código del sujeto
    :param n_inds: Índice máximo del sujeto
    :param aleatorizacion: Excel aclaratorio sobre qué intervención recibió cada registro
    :param base_datos: Base de datos a agregar valores
    :return: base_datos + columna intervención + columna sujeto - index
    """

    #Crear el diccionario de intervención vacío
    intervencion = {}

    for i in range(1, n_inds + 1):
        # Crear el índice del sujeto
        numero = "0" + str(i) if i < 10 else str(i)

        # Crear los nombres de ambos archivos del sujeto
        cond_1, cond_2 = "P" + numero + "_MT1", "P" + numero + "_MT2"

        # Extraer la condición de aletorización del sujeto
        cond_aleatorizacion = aleatorizacion.iloc[i - 1, 0]

        # Si la condición de aletorización es 1, significa que MT1 recibió musicoterapia
        if cond_aleatorizacion == 1:
            intervencion[cond_1] = "Musicoterapia"
            intervencion[cond_2] = "Control"
        else:
            intervencion[cond_1] = "Control"
            intervencion[cond_2] = "Musicoterapia"

    # Crear la columna que aclara la intervención en la base de datos
    base_datos["Intervencion"] = base_datos.index.map(intervencion)
    base_datos["Sujeto"] = base_datos.index.str[:3]
    base_datos.reset_index(inplace=True)
    base_datos.drop(columns=["index"], inplace=True)

    return base_datos


def preparar_datos_medida(df, estimulos, medida):
    """
    Filtra los sujetos que tienen ambas condiciones de intervención y transforma una medida de formato ancho a formato largo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una fila por sujeto-condición.
    estimulos : list
        Lista de estímulos, por ejemplo:
        ["Baseline", "Nursing", "Intervention", "Post"]
    medida : str
        Medida a evaluar, por ejemplo "HRV_LFHF".

    Returns
    -------
    pd.DataFrame
        DataFrame en formato largo con las columnas:
        Sujeto, Intervencion, Estimulo y medida.
    """

    # Identificar sujetos que tienen ambas condiciones
    sujetos_pareados = (
        df.groupby("Sujeto")["Intervencion"]
        .nunique()
    )

    sujetos_pareados = sujetos_pareados[
        sujetos_pareados == 2
    ].index

    # Filtrar sujetos pareados
    df_pareado = df[
        df["Sujeto"].isin(sujetos_pareados)
    ].copy()

    # Crear nombres de las columnas que contienen la medida
    columnas_medida = [
        f"{estimulo}_{medida}"
        for estimulo in estimulos
    ]

    # Transformar a formato largo
    df_largo = df_pareado.melt(
        id_vars=["Sujeto", "Intervencion"],
        value_vars=columnas_medida,
        var_name="Estimulo",
        value_name=medida
    )

    # Eliminar el sufijo "_medida" del nombre del estímulo
    df_largo["Estimulo"] = df_largo["Estimulo"].str.replace(
        f"_{medida}",
        "",
        regex=False
    )

    # Mantener el orden especificado por el usuario
    df_largo["Estimulo"] = pd.Categorical(
        df_largo["Estimulo"],
        categories=estimulos,
        ordered=True
    )

    return df_largo


def calcular_cambio_baseline(df_largo, medida, nombre_baseline="Baseline"):
    """
    Resta el valor de Baseline a cada estímulo dentro de cada sujeto e intervención.
    Las filas de Baseline conservan su valor crudo (se usan como control de comparabilidad).
    :param df_largo: DataFrame en formato largo generado por preparar_datos_medida()
    :param medida: Nombre de la columna con la medida
    :param nombre_baseline: Nombre del estímulo usado como Baseline
    :return: DataFrame con la misma forma + columna Variable ("crudo" o "cambio")
    """

    # Valor de Baseline de cada sujeto en cada intervención
    es_baseline = df_largo["Estimulo"] == nombre_baseline
    valores_baseline = df_largo.loc[es_baseline, ["Sujeto", "Intervencion", medida]].rename(
        columns={medida: "valor_baseline"}
    )

    # Asociar a cada fila su valor de Baseline (el merge left conserva el orden)
    df_cambio = df_largo.merge(valores_baseline, on=["Sujeto", "Intervencion"], how="left")
    es_baseline = df_cambio["Estimulo"] == nombre_baseline

    # Cambio respecto a Baseline en los estímulos que no son Baseline
    df_cambio.loc[~es_baseline, medida] = df_cambio.loc[~es_baseline, medida] - df_cambio.loc[~es_baseline, "valor_baseline"]

    df_cambio["Variable"] = np.where(es_baseline, "crudo", "cambio")
    df_cambio.drop(columns=["valor_baseline"], inplace=True)

    return df_cambio


def wilcoxon_por_estimulo(df, medida, estimulos_holm=("Intervention", "Post"), corregir=True):
    """
    Compara Musicoterapia vs Control mediante Wilcoxon pareado
    para cada estímulo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame en formato largo generado por calcular_cambio_baseline().
    medida : str
        Nombre de la columna con la medida.
    estimulos_holm : tuple
        Estímulos que son hipótesis del efecto. Solo sobre estos se aplica la corrección de Holm;
        los demás son controles de comparabilidad.
    corregir : bool
        Si es False no se aplica la corrección de Holm (análisis exploratorio).

    Returns
    -------
    pd.DataFrame
        Resultados de la prueba por estímulo.
    """

    resultados = []

    for estimulo in df["Estimulo"].cat.categories:

        datos = df[
            df["Estimulo"] == estimulo
        ].copy()

        # Convertir las condiciones a columnas
        pareado = datos.pivot(
            index="Sujeto",
            columns="Intervencion",
            values=medida
        )

        # Eliminar sujetos que tengan algún valor faltante
        pareado = pareado.dropna(
            subset=["Musicoterapia", "Control"]
        )

        n = len(pareado)

        # Diferencia pareada para reportar su magnitud
        diferencias = pareado["Musicoterapia"] - pareado["Control"]

        # Wilcoxon pareado
        estadistico, p = wilcoxon(
            pareado["Musicoterapia"],
            pareado["Control"],
            alternative="two-sided"
        )

        resultados.append({
            "Estimulo": estimulo,
            "Variable": datos["Variable"].iloc[0],
            "Tipo": "hipotesis" if estimulo in estimulos_holm else "control",
            "n": n,
            "mediana_dif": diferencias.median(),
            "W": estadistico,
            "p": p
        })

    resultados = pd.DataFrame(resultados)

    if not corregir:
        return resultados

    # Corrección por comparaciones múltiples solo sobre las hipótesis
    es_hipotesis = resultados["Tipo"] == "hipotesis"
    resultados["p_holm"] = np.nan
    resultados.loc[es_hipotesis, "p_holm"] = multipletests(
        resultados.loc[es_hipotesis, "p"],
        method="holm"
    )[1]

    return resultados


def efecto_orden(df, medida, aleatorizacion):
    """
    Compara la diferencia Musicoterapia - Control entre los sujetos que recibieron musicoterapia en la sesión 1
    y los que la recibieron en la sesión 2 (Mann-Whitney), para detectar un efecto de periodo.
    :param df: DataFrame en formato largo generado por calcular_cambio_baseline()
    :param medida: Nombre de la columna con la medida
    :param aleatorizacion: DataFrame de aleatorización (condición 1 -> MT1 es musicoterapia)
    :return: DataFrame con los resultados por estímulo
    """

    resultados = []

    for estimulo in df["Estimulo"].cat.categories:

        # Diferencia MT - Control de cada sujeto
        pareado = df[df["Estimulo"] == estimulo].pivot(
            index="Sujeto",
            columns="Intervencion",
            values=medida
        ).dropna(subset=["Musicoterapia", "Control"])

        diferencias = pareado["Musicoterapia"] - pareado["Control"]

        # Sujetos que recibieron musicoterapia en la sesión 1
        mt_sesion_1 = np.array([
            aleatorizacion.iloc[int(sujeto[1:]) - 1, 0] == 1
            for sujeto in diferencias.index
        ])

        dif_sesion_1 = diferencias[mt_sesion_1]
        dif_sesion_2 = diferencias[~mt_sesion_1]

        estadistico, p = mannwhitneyu(dif_sesion_1, dif_sesion_2, alternative="two-sided")

        resultados.append({
            "Estimulo": estimulo,
            "n_MT_sesion_1": len(dif_sesion_1),
            "n_MT_sesion_2": len(dif_sesion_2),
            "mediana_dif_MT_sesion_1": dif_sesion_1.median(),
            "mediana_dif_MT_sesion_2": dif_sesion_2.median(),
            "U": estadistico,
            "p": p
        })

    return pd.DataFrame(resultados)


def plot_violin(df, medida, carpeta_guardado=None):
    """
    Genera un boxplot del cambio respecto a Baseline de una medida, mostrando la distribución
    de cada condición de intervención para cada estímulo.
    """

    # Solo los cambios: Baseline crudo está en otra escala
    df_cambios = df[df["Variable"] == "cambio"].copy()
    df_cambios["Estimulo"] = df_cambios["Estimulo"].cat.remove_unused_categories()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df_cambios,
        x="Estimulo",
        y=medida,
        hue="Intervencion",
        ax=ax
    )

    ax.set_ylabel(f"{medida} (cambio vs Baseline)")
    ax.set_xlabel("Estímulo")

    fig.tight_layout()

    #Guardar
    if carpeta_guardado:
        carpeta = FIGURES / carpeta_guardado
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / f"{medida}.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def wilcoxon_eeg(supresion_banda, estimulos, canales):
    """
    Compara Musicoterapia vs Control (Wilcoxon pareado, sin corrección) en cada canal y estímulo de una banda.
    Los valores ya son el cambio respecto a Baseline en dB. Un canal vacío solo excluye al par en ese canal.
    :param supresion_banda: DataFrame de una banda con columnas Estimulo_Canal + Intervencion y Sujeto
    :param estimulos: Lista de estímulos a comparar
    :param canales: Lista de canales de EEG
    :return: DataFrame con los resultados por canal y estímulo
    """

    resultados = []

    for canal in canales:
        # Formato largo del canal (la medida es el nombre del canal)
        df_canal = preparar_datos_medida(supresion_banda, estimulos, canal)
        df_canal["Variable"] = "cambio"

        resultado_canal = wilcoxon_por_estimulo(df_canal, canal, corregir=False)
        resultado_canal.insert(0, "Canal", canal)
        resultados.append(resultado_canal)

    return pd.concat(resultados, ignore_index=True)


def plot_topomapas(resultados_eeg, estimulos, canales, carpeta_guardado=None, alpha=0.05,
                   columna_valor="mediana_dif", etiqueta_color="Mediana MT − Control (dB)",
                   titulo="Diferencia de potencia MT − Control",
                   nombre_archivo="topomapas_supresion.png"):
    """
    Topomapas de una medida de diferencia MT - Control (dB) para cada banda (filas) y estímulo (columnas).
    Los canales con p < alpha (sin corregir) se marcan.
    :param resultados_eeg: Diccionario banda: DataFrame con columnas Canal, Estimulo, p y columna_valor
    :param estimulos: Lista de estímulos (columnas de la figura)
    :param canales: Lista de canales de EEG
    :param carpeta_guardado: Carpeta (relativa a FIGURES) para guardar la imagen
    :param alpha: Umbral para marcar los canales
    :param columna_valor: Columna con el valor a graficar
    :param etiqueta_color: Etiqueta de la barra de color
    :param titulo: Título de la figura
    :param nombre_archivo: Nombre del archivo de salida
    """

    # Posiciones de los electrodos según el montaje 10-20
    info = mne.create_info(ch_names=canales, sfreq=250, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    # Escala de color simétrica y común a todos los mapas
    limite = max(tabla[columna_valor].abs().max() for tabla in resultados_eeg.values())

    fig, axes = plt.subplots(len(resultados_eeg), len(estimulos), figsize=(3 * len(estimulos), 3 * len(resultados_eeg)))

    for i_banda, (banda, tabla) in enumerate(resultados_eeg.items()):
        for i_est, estimulo in enumerate(estimulos):
            ax = axes[i_banda, i_est]

            # Valores en el orden de los canales
            datos = tabla[tabla["Estimulo"] == estimulo].set_index("Canal").loc[canales]

            im, _ = mne.viz.plot_topomap(
                datos[columna_valor].to_numpy(),
                info,
                axes=ax,
                cmap="RdBu_r",
                vlim=(-limite, limite),
                mask=(datos["p"] < alpha).to_numpy(),
                mask_params=dict(marker="o", markerfacecolor="k", markersize=6),
                show=False
            )

            if i_banda == 0:
                ax.set_title(estimulo)
            if i_est == 0:
                ax.set_ylabel(banda)

    fig.colorbar(im, ax=axes.ravel().tolist(), label=etiqueta_color, shrink=0.6)
    fig.suptitle(titulo)

    if carpeta_guardado:
        carpeta = FIGURES / carpeta_guardado
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / nombre_archivo, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def ancova_eeg(potencia_banda, estimulos, canales, nombre_baseline="Baseline", estimulos_control=("Nursing",)):
    """
    Efecto de la musicoterapia ajustado por Baseline (ANCOVA de diferencias pareadas) en cada canal y estímulo:
    d_estimulo = b0 + b1 * d_baseline, con d = Musicoterapia - Control en potencia absoluta (dB).
    b0 es el efecto ajustado y b1 cuánto se arrastra la diferencia de Baseline.
    Un canal vacío solo excluye al par en ese canal.
    :param potencia_banda: DataFrame de una banda con columnas Estimulo_Canal + Intervencion y Sujeto
    :param estimulos: Lista de estímulos a evaluar
    :param canales: Lista de canales de EEG
    :param nombre_baseline: Nombre del estímulo usado como covariable
    :param estimulos_control: Estímulos previos a la intervención (controles de comparabilidad)
    :return: DataFrame con los resultados por canal y estímulo
    """

    resultados = []

    for canal in canales:
        # Valores de Baseline por sujeto y condición
        baseline = potencia_banda.pivot(index="Sujeto", columns="Intervencion", values=f"{nombre_baseline}_{canal}")
        d_baseline = baseline["Musicoterapia"] - baseline["Control"]

        for estimulo in estimulos:
            valores = potencia_banda.pivot(index="Sujeto", columns="Intervencion", values=f"{estimulo}_{canal}")
            d_estimulo = valores["Musicoterapia"] - valores["Control"]

            # Solo pares con dato en ambas sesiones
            datos = pd.DataFrame({"d_estimulo": d_estimulo, "d_baseline": d_baseline}).dropna()

            modelo = sm.OLS(datos["d_estimulo"], sm.add_constant(datos["d_baseline"])).fit()
            ic95 = modelo.conf_int().loc["const"]

            resultados.append({
                "Canal": canal,
                "Estimulo": estimulo,
                "Tipo": "control" if estimulo in estimulos_control else "hipotesis",
                "n": len(datos),
                "efecto_ajustado": modelo.params["const"],
                "ic95_inf": ic95[0],
                "ic95_sup": ic95[1],
                "t": modelo.tvalues["const"],
                "p": modelo.pvalues["const"],
                "pendiente_baseline": modelo.params["d_baseline"]
            })

    return pd.DataFrame(resultados)


"""============USAR LAS FUNCIONES=================="""

#Leer los resultados de ECG
hrv = pd.read_excel(RESULTS / "ECG_results.xlsx", sheet_name="HRV", index_col=0)
hr = pd.read_excel(RESULTS / "ECG_results.xlsx", sheet_name="HR", index_col=0)

#La hoja HR no tiene sufijo de medida: Baseline -> Baseline_HR
hr.columns = [f"{columna}_HR" for columna in hr.columns]

#HRV en escala log: medidas sesgadas y el cambio queda como ln(estímulo / Baseline)
hrv = np.log(hrv)
hrv.columns = hrv.columns.str.replace("_HRV_", "_ln_HRV_")

# Agregar la intervención y el sujeto a las 2 bases de datos
agregar_col_intervencion(aleatorizacion, hrv)
agregar_col_intervencion(aleatorizacion, hr)

estimulos = ["Baseline", "Nursing", "Intervention", "Post"]

#Base de datos de cada medida
medidas = {
    "HR": hr,
    "ln_HRV_LF": hrv,
    "ln_HRV_HF": hrv,
    "ln_HRV_LFHF": hrv
}

resultados_wilcoxon = {}
resultados_orden = []
datos_cambio = {}

for medida, base_datos in medidas.items():
    # Formato largo y cambio respecto a Baseline
    df_largo = preparar_datos_medida(base_datos, estimulos, medida)
    datos_cambio[medida] = calcular_cambio_baseline(df_largo, medida)

    # Musicoterapia vs Control
    resultados_wilcoxon[medida] = wilcoxon_por_estimulo(datos_cambio[medida], medida)

    # En escala log, exp(mediana_dif) es el cociente MT/Control del cambio relativo
    if medida.startswith("ln_"):
        resultados_wilcoxon[medida]["cambio_relativo"] = np.exp(resultados_wilcoxon[medida]["mediana_dif"])

    # Efecto del orden de las sesiones
    orden = efecto_orden(datos_cambio[medida], medida, aleatorizacion)
    orden.insert(0, "Medida", medida)
    resultados_orden.append(orden)

    print(f"\n========== {medida} ==========")
    print(resultados_wilcoxon[medida])
    print("\nEfecto de orden:")
    print(orden)

#Guardar los resultados en un excel
with pd.ExcelWriter(RESULTS / "estadistica_ECG.xlsx") as writer:
    for medida, tabla in resultados_wilcoxon.items():
        tabla.to_excel(writer, sheet_name=medida, index=False)
    pd.concat(resultados_orden).to_excel(writer, sheet_name="Efecto_orden", index=False)

#Boxplots del cambio respecto a Baseline
for medida, df_cambio in datos_cambio.items():
    plot_violin(df_cambio, medida, carpeta_guardado="estadistica_ECG")


"""============EEG (exploratorio, sin corrección)=================="""

#Leer todas las hojas: diccionario banda -> DataFrame (cambio respecto a Baseline en dB)
supresion = pd.read_excel(RESULTS / "supresion_potencias.xlsx", sheet_name=None, index_col=0)

estimulos_eeg = ["Nursing", "Intervention", "Post"]
canales = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2", "T7", "T8", "Cz"]

resultados_eeg = {}

for banda, supresion_banda in supresion.items():
    agregar_col_intervencion(aleatorizacion, supresion_banda)
    resultados_eeg[banda] = wilcoxon_eeg(supresion_banda, estimulos_eeg, canales)

    print(f"\n========== EEG {banda} ==========")
    print(resultados_eeg[banda])

#=====Revisión del denominador: potencia absoluta (dB re µV²/Hz), incluido Baseline=====
potencias_absolutas = pd.read_excel(RESULTS / "potencias_absolutas.xlsx", sheet_name=None, index_col=0)
estimulos_absolutos = ["Baseline"] + estimulos_eeg

resultados_absolutos = {}
resultados_ancova = {}

for banda, potencia_banda in potencias_absolutas.items():
    agregar_col_intervencion(aleatorizacion, potencia_banda)
    resultados_absolutos[banda] = wilcoxon_eeg(potencia_banda, estimulos_absolutos, canales)

    print(f"\n========== EEG {banda} - potencia absoluta ==========")
    print(resultados_absolutos[banda])

    # Análisis principal: efecto ajustado por Baseline (la supresión sobrecorrige)
    resultados_ancova[banda] = ancova_eeg(potencia_banda, estimulos_eeg, canales)

    print(f"\n========== EEG {banda} - ANCOVA (ajustado por Baseline) ==========")
    print(resultados_ancova[banda])

#=====Revisión de artefactos: % BAD, EMG y EOG=====
artefactos = pd.read_excel(RESULTS / "indicadores_artefacto.xlsx", index_col=0)
agregar_col_intervencion(aleatorizacion, artefactos)

#Indicador: (estímulos, tipo de variable). BAD en crudo; EMG y EOG como cambio respecto a Baseline
indicadores = {
    "BAD": (estimulos_absolutos, "crudo"),
    "EMG_alta": (estimulos_eeg, "cambio"),
    "EMG_total": (estimulos_eeg, "cambio"),
    "EOG_delta": (estimulos_eeg, "cambio")
}

resultados_artefactos = []

for indicador, (estimulos_indicador, variable) in indicadores.items():
    df_indicador = preparar_datos_medida(artefactos, estimulos_indicador, indicador)
    df_indicador["Variable"] = variable

    resultado = wilcoxon_por_estimulo(df_indicador, indicador, corregir=False)
    resultado.insert(0, "Indicador", indicador)
    resultados_artefactos.append(resultado)

resultados_artefactos = pd.concat(resultados_artefactos, ignore_index=True)

print("\n========== EEG - indicadores de artefacto ==========")
print(resultados_artefactos)

#Guardar los resultados en un excel
with pd.ExcelWriter(RESULTS / "estadistica_EEG.xlsx") as writer:
    for banda, tabla in resultados_eeg.items():
        tabla.to_excel(writer, sheet_name=banda, index=False)
    for banda, tabla in resultados_absolutos.items():
        tabla.to_excel(writer, sheet_name=f"{banda}_absoluta", index=False)
    for banda, tabla in resultados_ancova.items():
        tabla.to_excel(writer, sheet_name=f"{banda}_ANCOVA", index=False)
    resultados_artefactos.to_excel(writer, sheet_name="Artefactos", index=False)

plot_topomapas(resultados_eeg, estimulos_eeg, canales, carpeta_guardado="estadistica_EEG")

plot_topomapas(resultados_ancova, estimulos_eeg, canales, carpeta_guardado="estadistica_EEG",
               columna_valor="efecto_ajustado", etiqueta_color="Efecto ajustado por Baseline (dB)",
               titulo="Efecto MT − Control ajustado por Baseline (ANCOVA)",
               nombre_archivo="topomapas_ANCOVA.png")
