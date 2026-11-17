# Stage 1: adaptación para la demostración en cascada

## Cambio

El modelo original `public/stage1_resnet.onnx` reconoce reposo en sus CSV de sensores, pero clasifica como actividad casi todos los ECG clínicos de la muestra de desarrollo. Se conserva sin cambios.

La variante `public/stage1_resnet_domain.onnx` mantiene la ResNet y su clasificador original congelados. Añade un adaptador neuronal 128 → 32 (ReLU) → 1 entrenado para reconocer el origen de adquisición a partir de las representaciones del ECG. Para los registros clínicos se utiliza el protocolo de reposo documentado por [PhysioNet, Shaoxing y Ningbo](https://physionet.org/content/ecg-arrhythmia/1.0.0/), también descrito en la [publicación de Chapman-Shaoxing](https://doi.org/10.1038/s41597-020-0386-x).

La salida se combina como `P(actividad) = (1 − P(adquisición clínica)) × P(actividad del modelo original)`. Se exportan log-probabilidades en `logits`; el softmax de la página recupera las probabilidades combinadas. Se mantienen el orden [reposo, actividad], la entrada float32 `[batch,1,4000]`, la normalización por ventana y argmax.

El adaptador utiliza la señal, nunca la opción seleccionada, el nombre del archivo o la clase de arritmia. Es una adaptación al protocolo de adquisición de estas fuentes, no un detector universal de movimiento en cualquier dispositivo clínico. No se evaluó movimiento con dispositivos clínicos.

## Datos y separación

- Sensores: 30 CSV de reposo y 120 de actividad, 1075 ventanas reales de 4000 muestras. Los tiempos de los CSV son compatibles con 500 Hz.
- Clínica: 4800 registros muestreados de las cuatro clases de ritmo en el Parquet aportado, con protocolo de adquisición en reposo documentado. La etiqueta de actividad procede del protocolo, no del diagnóstico cardíaco.
- Los cuatro registros clínicos de la página fueron excluidos antes del muestreo: JS41805, JS41459, JS29551 y JS02317.
- Ajuste: 4005 ventanas; validación: 934; evaluación interna: 936. Los sujetos de sensores se mantienen juntos; los registros clínicos no se comparten entre particiones.
- El origen del adaptador se balancea entre reposo clínico, reposo de sensores y actividad de sensores. Se aplica inversión de polaridad únicamente como aumento de datos de ajuste.
- Adam, semilla 42, capa oculta de 32 unidades; selección de época mediante el promedio de recall de los tres grupos de validación. No se ajusta un umbral sobre los cinco ejemplos.

## Resultados internos y límites

| Grupo de evaluación | Antes | Variante |
|---|---:|---:|
| Reposo clínico reconocido (720 registros) | 20/720, 2,8 % | 693/720, 96,3 % |
| Reposo de sensores reconocido (43 ventanas) | 41/43, 95,3 % | 41/43, 95,3 % |
| Actividad de sensores reconocida (173 ventanas) | 150/173, 86,7 % | 140/173, 80,9 % |

La adaptación mejora sustancialmente el paso de ECG clínicos a Stage 2, con una reducción de 5,8 puntos en el reconocimiento de actividad de sensores en esta partición. Los resultados agregados, hashes y ejemplos están en `stage1-evaluation.json`.

Las particiones de evaluación fueron inspeccionadas durante el desarrollo de varios candidatos. No son una prueba final intacta. No se puede reconstruir la pertenencia de entrenamiento del extractor original desde el ONNX. Estas cifras describen la evaluación interna de la adaptación y no equivalen a validación clínica independiente ni a una precisión garantizada para señales nuevas.

El correcto funcionamiento de los cinco ejemplos comprueba la integración; no mide precisión estadística. Stage 2 conserva la adaptación previa de 5000 a 4000 muestras, cuya precisión tampoco tiene una validación independiente.

## Reproducción

Instalar NumPy, ONNX, ONNX Runtime y PyArrow. Ejecutar `scripts/adapt-stage1-domain.py --help` y proporcionar las carpetas CSV, el Parquet, el ONNX original, `public/signals.json` y una carpeta de resultados fuera del repositorio. El procedimiento exporta la variante solamente si cumple las comprobaciones definidas y conserva los pesos originales.

Verificación de la cascada en el mismo runtime WASM del navegador: `node scripts/verify-cascade.cjs`. Diagnóstico de Stage 1: `node scripts/diagnose-stage1.cjs [ruta-modelo]`.

Los CSV originales, el Parquet y los artefactos con señales crudas se mantienen fuera del repositorio y no se publican en Vercel. Las fuentes clínicas requieren atribución conforme a su licencia CC BY 4.0; los enlaces anteriores identifican sus autores y metodología. Este proyecto añade la adaptación descrita.
