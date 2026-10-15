# MEMORY.md — Procesamiento y análisis de investigación 
Memoria del proyecto entre sesiones. Máximo ~50 líneas: resume o elimina lo que ya no 
aporte. 
## Estado actual 
1. `01_txt_to_mne.py`: `data/original_data/*.txt` (UTF-16, mV) → `data/MNE/*_raw.fif` (250 Hz, montaje 10-20, T3/T4 renombrados a T7/T8).
2. `extract_markers.py`: → `data/info_marcadores.xlsx`, 8 columnas en muestras (ini/fin de Baseline, Nursing, Intervention, Post). Baseline y Post = 5 min alrededor de las marcas `EB == "ON"`. Los marcadores faltantes/erróneos se corrigen a mano por sujeto (if/elif por `sujeto.stem`) y hay sujetos excluidos (`menos_marcadores`).
3. `02_clean_signal.py`: **interactivo** (`input()` + ventanas de MNE bloqueantes): re-referencia, filtro 0.1–40 Hz, ICA y limpieza manual → `data/eeg_clean/*_clean_eeg.fif` + `cleaning_logs.json`. Salta sujetos ya procesados. No ejecutarlo de forma automática.
4. `03_frequential_features_eeg.py`: potencias Delta/Theta/Alpha/Beta → `results/supresion_potencias.xlsx` (relativas a Baseline) y `results/series_potencia.joblib` (ventanas de 3 s, paso 1.5 s).
5. `04_plots.py`: z-score respecto a Baseline; gráficos individuales y promedio grupal de EEG.
6. `ECG.py` (lee `data/MNE`, no EEG limpio): HRV (LF, HF, LF/HF) y HR con neurokit2 → `results/ECG_results.xlsx` (hojas HRV, HR) y `results/series_hr.joblib`. `ECG_plots.py` grafica HR.
7. `05_statistical_analysis.py`: lee `ECG_results.xlsx` (hojas HRV y HR). Wilcoxon pareado MT vs Control por estímulo para HR, HRV_LF, HRV_HF, HRV_LFHF + efecto de orden (Mann-Whitney) → `results/estadistica_ECG.xlsx` y boxplots en `figures/estadistica_ECG/`. Hoy n = 10 pares. Sección EEG: Wilcoxon por canal y banda sobre `supresion_potencias.xlsx` (ya en dB respecto a Baseline) → `results/estadistica_EEG.xlsx` y `figures/estadistica_EEG/topomapas_supresion.png`.

- `RESULTADOS.md` (raíz): resumen de resultados significativos de HR, HRV y EEG, con lo esperado por azar en EEG. Actualizarlo si cambian los análisis.

## Aprendizajes y errores a evitar
- La hoja HR de `ECG_results.xlsx` no tiene sufijo de medida (`Baseline`, no `Baseline_HR`): se renombra al leerla.
- Si un Excel de `results/` está abierto en Excel, el guardado falla con PermissionError: cerrarlo antes de ejecutar.

## Decisiones (y por qué)
- Los promedios grupales recortan cada estímulo al mínimo del grupo, tomado de sujetos fijos (P01_MT1, P12_MT1, P11_MT2, P13_MT1) ya que son los sujetos con menores duraciones en los estímulos correspondientes.
- Estadística ECG con cambio respecto a Baseline (estímulo − Baseline), porque las dos sesiones parten de niveles basales distintos.
- Hipótesis = Intervention y Post (Holm por medida). Baseline crudo y Nursing − Baseline son controles sin Holm: antes de la intervención no se espera diferencia, y un p no significativo no prueba igualdad → se reporta `mediana_dif`.
- Efecto de orden (MT en sesión 1 vs 2) porque el estado clínico cambia entre sesiones bajo hipotermia.
- HRV (LF, HF, LF/HF) en ln: cambio = ln(estímulo/Baseline), `cambio_relativo` = exp(mediana_dif). Motivo: medidas sesgadas (skew 1.5–2.5 → |skew| < 0.5 en ln), LF/HF es un cociente y Wilcoxon asume diferencias simétricas. Decidido por argumento, no por el p. HR se queda en lpm.
- Bandas HRV de `ECG.py` adaptadas a neonatos: LF 0.04–0.2 Hz, HF 0.2–1.0 Hz (antes, las de adultos de neurokit2: 0.15 / 0.4 Hz), porque la respiración neonatal está en ≈0.7–1 Hz.
- HF hasta 1.0 Hz y no 1.5 Hz (que propone la literatura): la serie RR tiene una muestra por latido, así que solo hay variabilidad real hasta HR/2. Con la HR de la muestra (91–150 lpm, mediana 113, baja por la hipotermia), ese límite es 0.76–1.25 Hz (mediana 0.95). Por encima, la potencia sale de la interpolación de neurokit2, no del corazón. Las referencias de 1.5 Hz parten de HR neonatales de ~140–160 lpm.
- EEG sin corrección múltiple (exploratorio, 88 pruebas de hipótesis): p crudo + mediana_dif en dB. Con n = 10, Bonferroni/Holm exigiría p < 0.0006, pero el p mínimo de Wilcoxon es ≈ 0.002. Interpretar patrones espaciales, no canales aislados. Un canal vacío solo excluye al par en ese canal (P07: Fp1 y Cz con n = 9).
## Próximos pasos
- EEG supresión: 42/88 pruebas con p < 0.05, todas con MT > Control; Nursing 0/44.
- Revisión (`03` genera `potencias_absolutas.xlsx` e `indicadores_artefacto.xlsx`; `05` las prueba):
  - En potencia absoluta, el Baseline (y Nursing) de MT es más bajo que el de control (Delta −1.7 dB, 5/11 canales), mientras que en Intervention casi no hay diferencia (Delta 0/11). Beta sí difiere en Intervention (+1.4 dB, 5/11).
  - Control tuvo más tiempo BAD en Baseline (18 % vs 7 %, p = 0.08).
  - EMG 20–40 Hz no difiere (EMG filtrado a 40 Hz: límite de este control).
  - → El efecto en Delta viene sobre todo del denominador (posible artefacto residual en el Baseline de control o regresión a la media). Beta es el hallazgo más creíble.
- **EEG: la ANCOVA es el análisis principal; la supresión queda como descriptiva.** Modelo `ancova_eeg`: d_estímulo = β0 + β1·d_baseline (d = MT − Control en potencia absoluta dB, OLS, sin término de orden). Motivo: β1 tiene mediana 0.30 y la supresión supone β1 = 1, así que sobrecorregía (42/88 → 9/88). Resultados: Delta P3, P4 y T8 en Intervention y Post (~+2 dB); Beta O1, O2 y T8 en Intervention (~+2.5 dB); Nursing 3/44. Salidas: hojas `{Banda}_ANCOVA` y `topomapas_ANCOVA.png`.
  - Pendiente opcional: p por permutación de β0, porque 20/132 modelos tienen residuos no normales (Shapiro).
- Ejecutar `ECG.py` y luego `05` con las bandas nuevas (los resultados actuales son con las bandas de adultos). Cerrar antes los Excel de `results/`.