# Resultados archivados

- [Métricas verificadas](metricas_verificadas.csv) contiene accuracy y F1 macro para las variantes comparadas en cada etapa.
- [Mejores hiperparámetros](mejores_hiperparametros.csv) contiene la combinación ganadora de cada búsqueda y su F1 macro medio en validación cruzada.
- [Procedencia de las figuras](fuentes_figuras.csv) identifica los CSV de predicciones usados por el cuaderno 05. La columna de origen registra las rutas de la corrida en Colab.
- [Figuras del cuaderno 05](figuras/) reúne la comparación de modelos, curvas, ejemplos y matrices. Los histogramas de error de las dos ResNet base están en [figuras](figuras/).

El cuaderno 03 exporta las tablas completas al ejecutar las búsquedas (`stage1_rf_grid_search.csv`, `stage1_lr_grid_search.csv`, `stage2_rf_grid_search.csv` y `stage2_lr_grid_search.csv`). Esos CSV detallados y los cuadernos con sus salidas de ejecución quedan pendientes de incorporar desde Drive.
