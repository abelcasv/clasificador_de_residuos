# Clasificador de Residuos con Redes Convolucionales (PyTorch)
## Práctica 1 – Aprendizaje Automático II (Curso 2026/2027)

Este repositorio contiene la solución completa para la **Práctica 1 de Aprendizaje Automático II**, orientada a la clasificación automática de residuos mediante redes convolucionales en PyTorch.

---

### Contenidos del Proyecto
* [`main.ipynb`](main.ipynb): Cuaderno principal con todo el flujo de trabajo:
  * Carga robusta con `SafeImageFolder` y partición estratificada (70% Train, 15% Val, 15% Test).
  * Data Augmentation en entrenamiento (rotaciones, volteos, color jitter, affine, normalización ImageNet).
  * Red convolucional modular propia desarrollada desde cero (`RedConvolucionalResiduos`).
  * Mecanismo de parada anticipada (**Early Stopping**) con paciencia de 5 épocas y restauración del mejor checkpoint.
  * Funciones de visualización de **curvas de pérdida y precisión** y **matriz de confusión** con Seaborn.
  * Búsqueda de hiperparámetros manual (5 configuraciones documentadas).
  * **[Extra]** Búsqueda automática de hiperparámetros (**Hyperparameter Tuning**) con **Optuna** (muestreo bayesiano TPE).
  * **[Extra]** Transferencia de conocimiento (**Transfer Learning**) comparando **ResNet-18** y **MobileNetV2**.
  * Código estrictamente libre de comentarios (`#`) conforme a la directriz del docente.
* [`RESUMEN_PROCESO_BUSQUEDA_HIPERPARAMETROS.md`](RESUMEN_PROCESO_BUSQUEDA_HIPERPARAMETROS.md): Memoria formal y detallada del proceso de experimentación, tabla comparativa de configuraciones, diagnóstico del problema del metal y resultados finales.
* [`results/`](results/): Gráficas conjuntas de pérdida y precisión, matrices de confusión y registros numéricos de las pruebas.
* [`Basura_Dataset/`](Basura_Dataset/): Conjunto de datos con 2.340 imágenes divididas en 7 clases (`train/`, `val/`, `test/`).

---

### Resultados Clave

| Configuración | Arquitectura | Entrada | Optimizador / LR | Test Acc | Recall Metal |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Prueba 1 | CNN Propia (3 convs) | 96x96 | Adam / 0.001 | 69,00% | 38,30% |
| Prueba 2 | CNN Propia + FC 256 | 96x96 | Adam / 0.001 | 61,00% | 23,40% |
| Prueba 3 | CNN Propia (128x128) | 128x128 | Adam / 0.001 | 66,00% | 29,79% |
| Prueba 4 | CNN Propia (Regularizada) | 128x128 | Adam / 0.0005 | 67,00% | 38,30% |
| **Prueba 5 (Final)** | **ResNet-18 (Transfer Learning)** | **224x224** | **Adam / 0.0005** | **91,34%** | **91,49%** |

*El modelo final supera con holgura el objetivo del 85% de acierto y resuelve por completo la confusión entre metal y vidrio.*
