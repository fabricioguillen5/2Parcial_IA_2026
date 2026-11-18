# Resultados archivados

- [Métricas verificadas](metricas_verificadas.csv) contiene accuracy y F1 macro para las variantes comparadas en cada etapa.
- [Mejores hiperparámetros](mejores_hiperparametros.csv) contiene la combinación ganadora de cada búsqueda y su F1 macro medio en validación cruzada.
- [Procedencia de las figuras](fuentes_figuras.csv) identifica los CSV de predicciones usados por el cuaderno 05. La columna de origen registra las rutas de la corrida en Colab.
- [Figuras del cuaderno 05](figuras/) reúne la comparación de modelos, curvas, ejemplos y matrices. Los histogramas de error de las dos ResNet base están en [figuras](figuras/).

## Resultados de la corrida de Colab

| Etapa | Regresión logística | Random Forest |
| --- | --- | --- |
| 1 | [2 configuraciones](colab/stage1_lr_grid_search.csv) | [8 configuraciones](colab/stage1_rf_grid_search.csv) |
| 2 | [3 configuraciones](colab/stage2_lr_grid_search.csv) | [16 configuraciones](colab/stage2_rf_grid_search.csv) |

Las cuatro tablas completas conservan los resultados de los tres folds, el promedio, la desviación y el ranking. Las combinaciones ganadoras coinciden con el resumen de hiperparámetros del cuaderno 05.

- [Métricas de validación](colab/validation_metrics.csv) para las 12 combinaciones de etapa y modelo.
- [Historiales y metadatos](colab/) de ResNet. El checkpoint es la época 9 en ambas redes base y la época 7 en ambas corridas del cuaderno 03.
- [Predicciones anonimizadas](colab/predicciones/) para los 12 pares de etapa y modelo. Las claves aleatorias permiten cruzar las tablas de una misma etapa. Se eliminaron los identificadores originales y se reemplazó el orden de los registros. Las tablas contienen etiquetas y puntajes del modelo.
- [Distribución de las particiones](colab/distribucion_particiones.csv) y [resumen de limpieza](colab/resumen_limpieza.csv). Etapa 1 contiene 859 ventanas de entrenamiento y 216 de validación. Etapa 2 contiene 26 969 registros de entrenamiento y 6743 de validación.

Estos archivos provienen de la carpeta `results_colab` exportada desde Drive. El ZIP contenía salidas de la ejecución. Los archivos `.ipynb` de [notebooks](../notebooks/) son las versiones locales disponibles, sin salidas guardadas.
