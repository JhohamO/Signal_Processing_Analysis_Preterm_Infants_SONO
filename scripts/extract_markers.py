import pandas as pd
import numpy as np
from pathlib import Path

from setuptools._distutils import text_file

#Definir las rutas
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
DATA_DIR_ORGINAL = DATA_DIR / 'original_data'

#Frecuencia de muestreo
SFREQ = 250

#Sujeto de prueba
sujeto = "P01_MT2.txt"

#Crear una función para cargar el text
def cargar_txt(filepath):
    """
    Crear el pd-DataFrame desde el txt
    :param filepath: Ruta al archivo txt
    :return: DataFrame con los datos del archivo txt
    """

    #Leer como txt y buscar el encabezado
    with open(filepath, "r", encoding="utf-16") as f:
        lines = f.readlines()

    #Línea del encabezado
    header_line = next(line for line in lines if line.startswith('% Fecha.Hora'))

    #Separarlo como columnas
    header_columns = header_line.split('\t')

    #Quitar las 2 columnas que no pertenecen
    header_columns = header_columns[:1] + header_columns[2:-1]

    #Arreglar los textos de las columnas
    header_columns = [col.replace("%", "").strip() for col in header_columns]

    #Leer el text como un df con sus respectivas columnas
    data = pd.read_csv(filepath, sep="\t", comment="%", encoding="utf-16", names=header_columns)

    return data


"""========Explicación de los archivos con problemas===========
-P06 en realidad es un archivo duplicado
-P03_MT1 los markers faltantes son a las 15:00 y 14:49, pero el archivo inicia a las 15:04
-P05_MT1 el marker faltante es a las 11:49, el archivo inicia a las 11:58
- No lo removí pero P13_MT1 tiene un marker 17:53 y el archivo inicia 05:47. Asumí que fue un error y modifiqué al 2do formato
"""

menos_marcadores = {"P06_MT1", "P03_MT1", 'P05_MT1'}

#Diccionario para llenar con la info de cada sujeto
markers_info = {}

#Recorrer todos los sujetos
for sujeto in DATA_DIR_ORGINAL.glob('*.txt'):

    #Saltar los archivo con inconvenientes por solucionar
    if sujeto.stem in menos_marcadores:
        continue

    #Cargar los datos
    data = cargar_txt(DATA_DIR_ORGINAL / sujeto)

    #Extrar la serie de horas para los casos donde hay que agregar el marcador manualmente
    horas = data["Fecha.Hora"].str.split(" ").str[1]

    #Datapoints de las marcas
    marcas = np.array(data.loc[data.EB == "ON"].index)

    """=====Casos donde hay que arreglar marcador de forma manual======"""
    #4 marcadores
    if sujeto.stem == "P07_MT2":
        marcas = marcas[1:]

    #4 marcadores
    elif sujeto.stem == "P08_MT2":
        marcas = marcas[:-1]

    #2 marcas
    elif sujeto.stem == "P01_MT1":
        #Hora del marcador faltante
        hora = "13:47"

        #Selección del último índice donde sale la hora deseada
        idx = horas[horas.str.startswith(hora)].index[-1]

        #Agregarlo a marcas como el marcador del medio en este caso
        marcas = np.insert(marcas, 1, idx)

    #1 marcador
    elif sujeto.stem == "P03_MT1":
        hora1 = "14:49"
        hora2 = "15:00"
        idx1 = horas[horas.str.startswith(hora1)].index[-1]
        idx2 = horas[horas.str.startswith(hora2)].index[-1]

        #Agregar los 2 primeros marcadores
        marcas = np.insert(marcas, 0, [idx1, idx2])

    #2 marcadores
    elif sujeto.stem == "P04_MT2":
        hora = "17:34"
        idx = horas[horas.str.startswith(hora)].index[-1]

        #Agregar como primer marcador
        marcas = np.insert(marcas, 0, idx)

    #2 marcadores
    elif sujeto.stem == "P05_MT1":
        hora = "11:49"
        idx = horas[horas.str.startswith(hora)].index[-1]

        #Agregar como primer marcador
        marcas = np.insert(marcas, 0, idx)

    # Cero marcadores
    elif sujeto.stem == "P13_MT1":
        hora1 = "05:53"
        hora2 = "05:59"
        hora3 = "06:09"

        idx1 = horas[horas.str.startswith(hora1)].index[-1]
        idx2 = horas[horas.str.startswith(hora2)].index[-1]
        idx3 = horas[horas.str.startswith(hora3)].index[-1]

        marcas = np.array([idx1, idx2, idx3])

    #Creo marcadores
    elif sujeto.stem == "P13_MT2":
        hora1 = "08:34"
        hora2 = "08:40"
        hora3 = "08:51"

        idx1 = horas[horas.str.startswith(hora1)].index[-1]
        idx2 = horas[horas.str.startswith(hora2)].index[-1]
        idx3 = horas[horas.str.startswith(hora3)].index[-1]

        marcas = np.array([idx1, idx2, idx3])

    #Creo los marcadores de inicio-final (en datapoints) de cada etapa. Estructura necesaria para usar las funciones ya creadas
    marcadores_ini_fin = np.array([(marcas[0] - 1) - (250*5*60), marcas[0] - 1, marcas[0], marcas[1], marcas[1] + 1,
                                   marcas[2], marcas[2] + 1, (marcas[2]+ 1) + (250*5*60)])

    # 250*5*6 corresponden a 5 minutos de registro.

    #Revisión que los markers no se salgan de los límites
    if marcadores_ini_fin[0] < 0 or marcadores_ini_fin[-1] > len(data)-2:
        print(f"El sujeto {sujeto.stem} tienen una inconsistencia en sus markers")

        #Si cumple ambas condiciones
        if marcadores_ini_fin[0] < 0 and marcadores_ini_fin[-1] > len(data)-2:
            print("No hay suficiente espacio ni para línea base ni para el post",
                  [marcadores_ini_fin[0] / 250, (marcadores_ini_fin[-1] - len(data)-2) / 250])

            #Corregir dejando el primer marker como el tiempo 0 y el último marker como el final del registro
            marcadores_ini_fin[0] = 0
            marcadores_ini_fin[-1] = len(data)-2

        #Si solo no tiene suficiente espacio para el post
        elif marcadores_ini_fin[-1] > len(data)-2:
            print("No suficiente tiempo post", (marcadores_ini_fin[-1] - len(data)-2) / 250)
            #Corregir
            marcadores_ini_fin[-1] = len(data)-2

        #Si no, no tiene suficiente espacio para linea base
        else:
            print("No suficiente tiempo para línea base", marcadores_ini_fin[0] / 250)

            #Corregir
            marcadores_ini_fin[0] = 0

    #Agregar la info a el diccionario
    markers_info[sujeto.stem] = marcadores_ini_fin

    #Guardar el diccionario
    markers = pd.DataFrame.from_dict(markers_info, orient='index', columns=['ini_lb', "fin_lb", "ini_nurse", "fin_nurse",
                                                                            "ini_int", "fin_int", "ini_post", "fin_post"])
    markers.to_excel(DATA_DIR / "info_marcadores.xlsx")








