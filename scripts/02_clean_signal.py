import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#Defino las rutas de los archivos
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DATA_MNE = DATA_DIR / 'MNE'
DATA_EEG_CLEAN = DATA_DIR / 'eeg_clean'

# def load_fif(filepath):
#     """
#     Load and prepare the .fif file before cleaning
#     :param filepath: Filepath of the EEG file
#     :return: Raw object prepared for cleaning
#     """

def limpieza_eeg(filepath):
    """
    EEG cleaning using: Re referentiation, frequential filtering, ICA.
    The pipeline is interactive, which allows taking decisions during the cleaning
    :param filepath: filepath .fif 
    :return: eeg file after cleaning
    """

    #Leer el archivo MNE
    raw = mne.io.read_raw_fif(filepath, preload=True)

    #Cortar la última muestra de cada canal que es vacía y genera problemas posteriormente
    raw.crop(
        tmax=(raw.n_times - 2) / raw.info['sfreq']
    )

    #Convertir los canales bipolares de EMG a un canal
    raw = mne.set_bipolar_reference(
        raw,
        anode='EMG+',
        cathode='EMG-',
        ch_name='EMG'
    )

    """Plotear como primera observación si hay que tener precaución con pasos posteriores"""
    fig = raw.plot(show=False)
    fig.canvas.manager.set_window_title("EEG RAW-Revisar referencias")
    plt.show(block=True)

    #Registro de las decisiones tomandas durante la limpieza
    cleaning_log = {}
REVISAR SI ESTO ES CORRECTO
    #Seleccionar la referencia según la inspección visual
    print("\n ¿Cómo referenciar?")
    print("1: Promedio de A1 y A2 como referencia")
    print("2: A1 como referencia")
    print("3: A2 como referencia")
    print("4: Saltar re-referencia")
    opcion_referencia = int(input("Opción de referencia: "))

    # Re-referencia
    if opcion_referencia == 1:
        ref_channels = ["A1", "A2"]
    elif opcion_referencia == 2:
        ref_channels = ["A1"]
    elif opcion_referencia == 3:
        ref_channels = ["A2"]
    else:
        ref_channels = None

    #Aplicar la referencia
    if ref_channels is not None:
        raw.set_eeg_reference(ref_channels=ref_channels)

    cleaning_log["electrodos_referencia"] = ref_channels

    # Crear las dos versiones de trabajo y de ICA
    raw_filtered = raw.copy().filter(0.1, 40)
    raw_ica = raw.copy().filter(1, 40)

    #Plotear el raw_re referenciado y filtrado
    fig = raw_filtered.plot(show=False)
    fig.canvas.manager.set_window_title("Post filtrado")
    plt.show(block=True)

    #A1/A2 se usan únicamente como canales de referencia del EEG. Removidos después de rereferenciar
    raw_filtered.drop_channels(["A1", "A2"])
    raw_ica.drop_channels(["A1", "A2"])

    #ICA
    ica = mne.preprocessing.ICA(random_state=44)
    ica.fit(raw_ica)

    #EMG correlations
    sources = ica.get_sources(raw_filtered).get_data()
    emg = raw_filtered.get_data(picks='EMG')[0]

    emg_correlations = np.array([
        np.corrcoef(source, emg)[0, 1]
        for source in sources
    ])

    #Plotear la correlación
    fig_emg = ica.plot_scores(emg_correlations, show=False)
    fig_emg.canvas.manager.set_window_title("EMG scores")


    #EOG and ECG scores
    eog_inds, eog_scores = ica.find_bads_eog(raw_filtered)
    ecg_inds, ecg_scores = ica.find_bads_ecg(raw_filtered)

    cleaning_log["eog_scores_ROC"] = list(eog_scores[0])
    cleaning_log["eog_scores_LOC"] = list(eog_scores[1])
    cleaning_log["ecg_scores"] = list(ecg_scores)
    cleaning_log["emg_correlations"] = list(emg_correlations)



    #Plot scores
    fig_eog = ica.plot_scores(eog_scores, show=False)
    fig_eog.canvas.manager.set_window_title("EOG scores")

    fig_ecg = ica.plot_scores(ecg_scores, show=False)
    fig_ecg.canvas.manager.set_window_title("ECG scores")

    plt.show(block=True)

    #Ver componentes
    fig_ica_components = ica.plot_components(inst=raw_filtered, show=False)
    fig_ica_components.canvas.manager.set_window_title("ICA components")
    plt.show(block=True)

    #Plotear sources en caso de considerarse necesario
    print("\n ¿Plotear sources?")
    print("y: Sí")
    print("n: No")
    plot_sources = input("¿Plotear?")

    if plot_sources == "y":
        fig_sources = ica.plot_sources(raw_filtered, show=False)
        fig_sources.canvas.manager.set_window_title("Sources plot")
        plt.show(block=True)

    #Plotear overlay para ver qué tanta información remueve el componente (en caso de ser necesario)
    print("\n Seleccione los componentes que planea plotear. En caso de que no lo desee deje vacía la entrada")
    texto_overlay = input("Componentes a probar su exclusión (ej: 2,6,8): "). strip()

    if texto_overlay:
        componentes_overlay = [int(x) for x in texto_overlay.split(",")]
    else:
        componentes_overlay = []

    #Ploteo el overlay
    for comp in componentes_overlay:
        fig_overlay = ica.plot_overlay(raw_filtered, exclude=[comp], show=False)
        fig_overlay.canvas.manager.set_window_title("Componente {} plot".format(comp))
        plt.show(block=True)

    #Componentes a remover
    print("\n Selecciobes los components a remover. En caso de que no lo desee deje vacía la entrada")
    texto_remover = input("Componentes a excluir (ej: 2,6,8): ").strip()
    if texto_remover:
        components_remover = [int(x) for x in texto_remover.split(",")]
    else:
        components_remover = []

    cleaning_log["componentes_removidos"] = components_remover

    #Remover los componentes seleccionados
    ica.apply(raw_filtered, exclude=components_remover)

    #Graficar el resultado y sobre esta realizar limpieza manual
    fig_limpieza = raw_filtered.plot(show=False)
    fig_limpieza.canvas.manager.set_window_title("Realizar limpieza manula de artefactos sobrevivientes")
    plt.show(block=True)

    #Comentarios sobre la limpieza
    comentarios = input("Comente sobre las decisiones tomandas durante la limpieza")
    cleaning_log["comentarios"] = comentarios

    return cleaning_log, raw_filtered


#Crear el diccionario donde se guardaran los comentarios de cada archivo
cleaning_logs = {}

#Crear el directorio de salida
DATA_EEG_CLEAN.mkdir(parents=True, exist_ok=True)

#Recorrer los archivos
for file in DATA_MNE.glob("*.fif"):

    #Aplicar la función
    cleaning_log, raw_limpio = limpieza_eeg(file)

    #Guardar el comentario
    cleaning_logs[file.stem[:-3]] = cleaning_log

    # Guardar el raw después de la limpieza
    output_path = DATA_EEG_CLEAN / f"{file.stem[:-3]}_clean_eeg.fif"
    raw_limpio.save(output_path, overwrite=True)



