# Pulso IA

Trabajo práctico de Inteligencia Artificial de Hilario Yudiz y Fabricio Guillen, estudiantes de Ingeniería Mecatrónica de la Facultad de Ingeniería de la Universidad Nacional de Asunción.

El proyecto estudia dos tareas sobre señales ECG. La primera distingue reposo y actividad en ventanas de 4000 muestras. La segunda utiliza la derivación I de registros de 5000 muestras para clasificar bradicardia sinusal, taquicardia sinusal, fibrilación auricular y ritmo sinusal normal.

## Cómo reproducir el trabajo

1. Consultar las fuentes y preparar las entradas según el [instructivo de datos](data/README.md).
2. Seguir el [instructivo de Colab](notebooks/LEEME_COLAB.md) y ejecutar los cuadernos 01 a 04 en orden. El cuaderno 05 prepara las figuras para la defensa a partir de los resultados y datos guardados.
3. Consultar los [resultados archivados](results/README.md).

El preprocesamiento genera 12 características para la primera etapa y 11 para la segunda. Los modelos clásicos reciben esas características. La ResNet 1D recibe la señal normalizada por ventana. La comparación incluye regresión logística y Random Forest, con Grid Search y F1 macro como criterio de selección. El cuaderno 03 también entrena una ResNet con reducción de la tasa de aprendizaje y parada temprana.

## Resultados en validación

| Tarea | Modelo seleccionado | Accuracy | F1 macro |
| --- | --- | ---: | ---: |
| Reposo y actividad | ResNet 1D base | 92,13 % | 87,32 % |
| Cuatro ritmos ECG | ResNet 1D base | 98,78 % | 97,64 % |

Los resultados corresponden a la evaluación por etapa en Colab. Queda pendiente una prueba final independiente del sistema completo.

## Material de la entrega

- [Fuentes de datos y scripts](data/README.md)
- [Cuadernos y pasos de ejecución](notebooks/LEEME_COLAB.md)
- [Resultados y figuras](results/README.md)
- [Página pública de Pulso IA](https://fabricioguillen5.github.io/2Parcial_IA_2026/)
- [CV del equipo](https://fabricioguillen5.github.io/2Parcial_IA_2026/cv/)
