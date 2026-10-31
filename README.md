# Clasificación Automatizada de Residuos Urbanos mediante Visión Artificial

**Autores:** Abel Castro Villar y Lucas Rodríguez Hernández  
**Contexto:** Proyecto de Aprendizaje Automático II - Grado en Ciencia e Ingeniería de Datos (ULPGC)

## Descripción del Proyecto
Este proyecto desarrolla y evalúa un sistema de visión artificial basado en Deep Learning (PyTorch) para la clasificación automatizada de residuos sólidos en plantas de triaje. El objetivo es optimizar la separación de basura, un proceso vital para la viabilidad de la economía circular.

El modelo clasifica imágenes en 7 categorías distintas:
Cartón (`t-cardboard`), Residuos electrónicos (`t-e-waste`), Vidrio (`t-glass`), Metal (`t-metal`), Orgánico (`t-organic`), Plástico (`t-plastic`) y Textil (`t-textile`).

## Metodología y Arquitecturas
El proyecto se desarrolla en tres fases experimentales incrementales:

1. **Diseño de CNN desde cero (`WasteCNN`):** 
   Construcción de una red convolucional modular propia. A través de 5 configuraciones manuales, se solucionaron cuellos de botella críticos, como la confusión óptica entre el metal y el vidrio, aplicando Early Stopping, Dropout adaptativo y variando la profundidad convolucional.
2. **Optimización Bayesiana (AutoML):** 
   Uso del framework Optuna con el algoritmo TPE (*Tree-structured Parzen Estimator*) y poda asíncrona (*MedianPruner*) para explorar el espacio de hiperparámetros y aislar el límite asintótico de la red artesanal.
3. **Transfer Learning:** 
   Adaptación de arquitecturas de estado del arte preentrenadas en ImageNet:
   * **ResNet-18:** Como arquitectura residual de contraste (11,18 M de parámetros).
   * **MobileNetV2:** Como arquitectura ligera y eficiente basada en convoluciones separables en profundidad (3,5 M de parámetros).

## Resultados Destacados
El modelo MobileNetV2 resultó ser el ganador absoluto, logrando un equilibrio óptimo entre eficiencia computacional y capacidad predictiva en el conjunto de test (datos nunca vistos durante el entrenamiento):

* **Accuracy Global:** 95,53%
* **Recall en clase crítica (Metal):** 91,49% (con 0 confusiones con vidrio).
* **Precisión perfecta (100%):** En categorías como Metal, Textil, Cartón y Electrónica.
* **Eficiencia de Parámetros:** 3,5 Millones (ideal para un futuro despliegue en dispositivos Edge / Edge AI).

## Tecnologías Utilizadas
* **Deep Learning:** PyTorch, Torchvision.
* **Optimización de Hiperparámetros:** Optuna.
* **Métricas y Evaluación:** Scikit-Learn.
* **Visualización:** Matplotlib, Seaborn.
* **Preprocesamiento:** Transformaciones estocásticas *on-the-fly* (Data Augmentation).

## Documentación y Memoria Técnica
El desglose completo de la experimentación, marcos teóricos, análisis de errores, curvas de aprendizaje y matrices de confusión se encuentra documentado en la memoria formal:
* **`Memoria_Practica1_AA2_Abel_Lucas.pdf`**: Documento PDF completo con el análisis de todas las fases del proyecto.

## Ejecución del Código
El cuaderno principal `Practica1_AA2_Abel_Lucas.ipynb` está diseñado de forma autónoma y estructurado bajo principios SOLID. 

Al ejecutar la celda de preparación del entorno (Sección 0), el script clona automáticamente este repositorio y despliega el dataset en el entorno de ejecución local o en Google Colab. No es necesaria ninguna configuración manual de rutas ni descargas externas para reproducir el proceso completo de entrenamiento y evaluación.
