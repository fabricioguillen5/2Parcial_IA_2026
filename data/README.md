# Fuentes y preparación de los datos de Pulso IA

Pulso IA usa dos fuentes de ECG. Cada una aporta las señales y las etiquetas para una tarea distinta. Los scripts de esta carpeta preparan las entradas del [cuaderno 01](../notebooks/01_preprocesamiento.ipynb).

## Etapa 1. Reposo y actividad

Los registros provienen de [ECG-interference, de fbme-ictm](https://github.com/fbme-ictm/ECG-interference). El repositorio organiza los CSV de ECG de un sensor portátil en carpetas por sujeto y publica información sobre las actividades, los electrodos y la licencia del dataset.

1. Descargar o clonar la fuente. Desde la raíz de esa descarga, donde se encuentra su carpeta `data/`, ejecutar `python <ruta-a-relocate.py>` apuntando al [script del proyecto](relocate.py).
2. El script copia los CSV cuyo nombre contiene `klud` a `data_resting/`. Copia a `data_not_resting/` los que contienen `ruky` (movimientos de brazos), `chodza` (caminata), `beh` (carrera) o `drepy` (sentadillas). Conserva los originales y antepone el identificador de la carpeta del sujeto al nombre de cada copia.
3. Colocar ambas carpetas generadas en `TP_IA/RestingClass/` en Drive, o indicar sus ubicaciones en los cuadernos. El cuaderno 01 limpia y segmenta la señal en 4000 muestras por ventana y separa entrenamiento y validación por sujeto. Para expresar la duración en segundos, el proyecto usa 500 Hz. La herramienta de etiquetado de la fuente también usa 500 Hz como valor predeterminado.

Las etiquetas del TP se derivan de la actividad indicada en el nombre de archivo. El script conserva solo los CSV que coinciden con los nombres descritos. Si una descarga usa códigos numéricos para la actividad, consultar la convención de la fuente y comprobar el formato antes de aplicar el script.

## Etapa 2. Clasificación de cuatro ritmos

La fuente es [A large scale 12-lead electrocardiogram database for arrhythmia study, v1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/). PhysioNet documenta registros de 12 derivaciones, 10 segundos y 500 Hz. El TP utiliza la derivación I y un subconjunto de diagnósticos.

1. Descargar la versión 1.0.0 de PhysioNet. Descomprimirla en `ecg-arrhythmia/` de modo que contenga `WFDBRecords/` y `ConditionNames_SNOMED-CT.csv`. Ejecutar el [script de conversión](conversion.py) desde la carpeta que contiene `ecg-arrhythmia/`. Sus dependencias se instalan con `python -m pip install wfdb pandas numpy pyarrow`.
2. El script abre los pares `.hea` y `.mat` con WFDB y extrae la señal física de la derivación I. Lee los códigos de diagnóstico del campo `Dx:` de la cabecera y asigna el primer código objetivo de la lista. Omite los registros sin código objetivo y aplica `numpy.nan_to_num` a la señal al convertirla. La asignación usa el diccionario fijo de códigos del script.
3. El archivo generado es `chapman_lead_I_consolidated.parquet`, con columnas `record_id`, `label` y `signal`. Colocarlo en `TP_IA/Detector/` en Drive o indicar su ruta en los cuadernos. El cuaderno 01 comprueba las señales de 5000 muestras y divide por clase, ordenando por identificador de registro.

| Código SNOMED CT | Clase del TP |
| --- | --- |
| `426177001` | Bradicardia sinusal |
| `427084000` | Taquicardia sinusal |
| `164889003` | Fibrilación auricular |
| `426783006` | Ritmo sinusal normal |

Referencia del dataset. Zheng, J., Guo, H. y Chu, H. (2022). *A large scale 12-lead electrocardiogram database for arrhythmia study* (v1.0.0). PhysioNet. [DOI 10.13026/wgex-er52](https://doi.org/10.13026/wgex-er52). La página de la fuente incluye las citas adicionales y las condiciones de uso.

El [instructivo de Colab](../notebooks/LEEME_COLAB.md) detalla la estructura de Drive, el orden de ejecución y las salidas de los cuadernos.
