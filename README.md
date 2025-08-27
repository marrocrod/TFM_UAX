# Predicción de estabilidad proteica con modelos clásicos y cuánticos  

Este repositorio contiene el código y los experimentos realizados en el Trabajo Fin de Máster (TFM), cuyo objetivo es la **predicción de la variación de energía libre de Gibbs (ΔΔG)** en mutaciones puntuales de proteínas. Se implementan y comparan tres enfoques distintos:  

- **XGBoost** (modelo clásico basado en árboles de decisión).  
- **Red neuronal clásica (FCNN)** implementada en TensorFlow/Keras.  
- **Red neuronal cuántico-clásica (QNN)** implementada en Qiskit + PyTorch.  

El trabajo busca explorar las capacidades y limitaciones de los modelos clásicos frente a las nuevas aproximaciones cuánticas en un problema real de bioinformática estructural.  

---

## 📂 Contenido del repositorio  

- `notebook.ipynb` → Notebook principal con el pipeline completo:  
  - Preprocesamiento de datos y extracción de características.  
  - Implementación y entrenamiento de XGBoost.  
  - Implementación y entrenamiento de la red neuronal clásica (FCNN).  
  - Implementación y entrenamiento de la red cuántica híbrida (QNN).  
  - Comparación de resultados entre los tres modelos.  

- `s2648.csv` → Dataset utilizado (si se incluye o instrucciones para obtenerlo).  
- `README.md` → Este archivo.  

---

## 📊 Dataset  

El dataset utilizado es **S2648**, ampliamente empleado en la literatura para el estudio de la estabilidad proteica en mutaciones puntuales. Contiene 2648 ejemplos con información sobre proteínas, cadenas, posición de mutación, residuos wild-type y mutados, junto con el valor experimental de ΔΔG (en kcal/mol).  

---

## 🚀 Ejecución  

Este notebook está pensado para ejecutarse directamente en un entorno con python 3.10 o 3.11

