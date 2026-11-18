# Cuadernos para Google Colab

El [documento de fuentes y preparación](../data/README.md) enlaza los datasets originales y describe cómo generar las entradas de ambas etapas con los scripts del proyecto.

Ejecutar en orden 01, 02, 03 y 04. Todos incluyen instalación, montaje de Drive y configuración; pueden abrirse en sesiones diferentes y comparten archivos guardados.

1. Subir a Google Drive la carpeta del proyecto con esta estructura (o editar las rutas en cada cuaderno):

```text
MyDrive/TP_IA/
  RestingClass/
    data_resting/*.csv
    data_not_resting/*.csv
  Detector/
    chapman_lead_I_consolidated.parquet
```

2. Abrir https://colab.research.google.com/ y cargar cada archivo .ipynb mediante Archivo → Subir cuaderno. Editar PROJECT_DIR y las rutas de datos en los cuatro archivos. Autorizar el montaje de Drive cuando se solicite.
3. Mantener STAGES=[1,2] para ambas etapas o usar [2] si primero se trabaja con Chapman. La etapa 1 requiere los CSV originales; los Parquet de cuatro/siete características no reemplazan las señales crudas.
4. Ejecutar 01 para preparar señales, características, registros de limpieza y particiones. Requiere memoria suficiente para las señales; los datos preparados se guardan comprimidos en results_colab/prepared/.
5. Ejecutar 02 para entrenar los tres modelos base sin búsqueda: regresión logística, Random Forest y ResNet 1D. La red usa 10 épocas por etapa (configurable) y conviene seleccionar GPU también para este cuaderno. Se generan seis filas de métricas con STAGES=[1,2].
6. En 03, seleccionar GPU desde Entorno de ejecución → Cambiar tipo de entorno de ejecución para acelerar ResNet. TRAIN_RESNET=False permite solo los modelos clásicos; las curvas de convergencia de la red requieren entrenarla. Se guarda el mejor checkpoint y se evalúa después de recargarlo.
7. Ejecutar 04 para generar predicciones, métricas, matrices, histogramas, ejemplos y figuras exportables. GENERATE_LEARNING_CURVES es opcional; las curvas de convergencia de la red se generan automáticamente si hay historial.

## Salidas

results_colab/ contiene CSV de limpieza, características, particiones, búsquedas completas, métricas, predicciones, historiales por época, versiones, modelos y figuras. El cuaderno 04 permite descargar resultados en ZIP. Se omiten del ZIP los archivos .npz preparados para no duplicar señales.

## Diferencias deliberadas respecto de los scripts

- Baseline_Power corregida; escalado dentro de cada fold; evaluación de la red con el mejor checkpoint.
- Se añade modelo base, CSV completos de Grid Search e histogramas de error probabilístico.
- Se usa umbral fijo 0,5/argmax: no se calibran umbrales sobre las etiquetas que se reportan. Las métricas pueden diferir de las originales.
- Todas las métricas precision/recall/F1 se reportan macro en ambas etapas.
- ResNet conserva la estructura residual y las pérdidas originales (FocalLoss ponderada en etapa 1, cross-entropy en etapa 2), con una implementación de arquitectura común. Son nuevos entrenamientos; no se cargan checkpoints originales. En etapa 2 se usa batch 128 para reducir memoria.
- Se descartan señales Chapman no finitas, constantes o de longitud distinta de 5000 y se registran. Las particiones de esta versión se guardan y comparten entre todos los modelos.
- Para reutilizar el modelo exportado, deben coincidir características, normalización, orden de clases y umbrales. Las salidas ONNX de ResNet son logits.

## Estado de validación

Los cuadernos se entregan sin salidas ejecutadas. No se ejecutaron en una cuenta de Colab ni se entrenaron modelos en esta sesión. No hay métricas simuladas. Revisar las comprobaciones de datos de 01 antes de entrenar. Si se vuelve a preparar datos, repetir los entrenamientos; se verifica la versión para evitar mezclar modelos viejos con datos nuevos.

El preprocesamiento asume 500 Hz y el convenio de nombres por sujeto del script original. Confirmar ambos con el dataset.

Referencias oficiales: [archivos en Colab](https://colab.research.google.com/notebooks/io.ipynb), [FAQ y recursos](https://research.google.com/colaboratory/faq.html).
