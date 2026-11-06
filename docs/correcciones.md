# Correcciones de la versión original

Este documento describe los errores que tenía la versión original del notebook ([`notebook/version_original/Notebook_final.ipynb`](../notebook/version_original/Notebook_final.ipynb)) y cómo se corrigen en [`notebook/estabilidad_proteica.ipynb`](../notebook/estabilidad_proteica.ipynb).

Esta fase **solo corrige errores**: los modelos, las pérdidas y los hiperparámetros son los mismos. Las métricas que se obtengan serán, previsiblemente, **más bajas que las originales**. No es que los modelos empeoren: se dejan de medir con ventaja.

---

## A. Errores en los datos

### 1. Posición de la mutación frente a índice de la secuencia

S2648 indica la posición con la **numeración del PDB**. Esa numeración casi nunca empieza en 1, puede tener huecos y puede llevar códigos de inserción. La versión original la usaba directamente como índice de la secuencia extraída:

```python
pos = row['position'] - 1          # PDB residue number used as a sequence index
aa = seq[pos]
```

En cualquier cadena cuya numeración no coincida con la secuencia, el residuo leído, sus vecinos y el embedding de ProtT5 eran los de **otra posición**. Ahora se construye el mapa *(número de residuo, código de inserción) → índice* recorriendo los residuos de la estructura, y se comprueba que el aminoácido en ese índice es el residuo original de la mutación. La celda informa de cuántas mutaciones coincidían con el índice antiguo y de cuántas se descartan.

### 2. Mutaciones con código de inserción

La expresión regular `([A-Z])(\d+)([A-Z])` lee `L27CN` como «L en 27, mutado a C», cuando es «L en 27C, mutado a N». Afecta a 4 mutaciones de 1LVE. Ahora se usa `^([A-Z])(\d+)([A-Z]?)([A-Z])$`.

### 3. Estructuras sin formato PDB

Algunas entradas, como 2A01, ya no se distribuyen en formato `.pdb`, y la descarga fallaba. Ahora se prueba el formato PDB y, si no está disponible, mmCIF.

## B. Errores en las características

### 4. Índice del embedding de ProtT5

```python
emb_pos = embedding[position + 1]   # original
```

ProtT5 (un T5) **no añade token inicial**: la secuencia `M K V …` se tokeniza como `▁M ▁K ▁V … </s>`. La salida `k` corresponde al residuo `k`, así que `position + 1` tomaba el embedding del **residuo siguiente** al mutado. Ahora se toma el índice correcto, y una aserción comprueba que hay exactamente un token por residuo más `</s>`.

Además, el embedding se calcula **una vez por cadena** (132) en lugar de una vez por mutación (2 648), y se guarda en `cache/prott5/`.

### 5. Recuento de aminoácidos vecinos

```python
aa_index = ord(aa) - ord('A')       # alphabetical index, 0..25
if 0 <= aa_index < 20: neighbor_counts[aa_index] += 1
```

El índice alfabético no es un índice de aminoácido. Las letras A–T ocupaban las 20 posiciones, así que se contaban B, J, O y otras que no son aminoácidos, y **V, W e Y nunca se contaban**. Ahora el recuento se hace sobre los 20 aminoácidos estándar.

## C. Errores en la evaluación

### 6. Particiones por mutación en lugar de por proteína

`KFold(shuffle=True)` y `train_test_split` reparten **mutaciones**: mutaciones de la misma proteína (1EY0 tiene 482) quedaban a la vez en entrenamiento y en prueba. El modelo podía aprender rasgos propios de cada proteína y las métricas sobrestimaban su capacidad para proteínas nuevas. Ahora se usan `GroupKFold(5)` y `GroupShuffleSplit(20 %)` agrupando por la columna `CLID`, que es el *cluster* de homología de S2648.

### 7. Preprocesado ajustado con todos los datos

`StandardScaler` (y, en la QNN, también la `PCA`) se ajustaban con el conjunto completo antes de dividir, así que la información de prueba influía en el entrenamiento. Ahora cada modelo hace todo su preprocesado dentro de su función `fit_predict`, con los datos de entrenamiento de la partición.

### 8. La QNN usaba la prueba para entrenar

```python
pred_val = predict_in_batches(net, Xte_q_t, Xte_c_t)   # test fold
sch.step(val_combo.item())                             # LR scheduler on the test fold
if val_combo.item() < best: best_state = ...           # early stopping on the test fold
```

La parada temprana y el planificador de la tasa de aprendizaje se guiaban por el conjunto de evaluación. Ahora se separa un 10 % del entrenamiento como validación, agrupado por proteína.

### 9. La QNN se evaluaba sobre otras mutaciones y otras particiones

| | XGBoost / FCNN | QNN (original) |
|---|---|---|
| Mutaciones | 2 647 | 2 464 (filtrado propio) |
| Particiones | `KFold` sobre su matriz | `KFold` sobre otra matriz |
| Posición | `posición − 1` | número de residuo del PDB sin mapear |

La QNN repetía por su cuenta la descarga y el filtrado. Las métricas de los tres modelos no se calculaban, por tanto, sobre las mismas mutaciones. Ahora los tres comparten mutaciones y particiones, y la posición sale del mapa corregido.

La QNN **mantiene su vector propio sin ProtT5**, porque es una decisión de diseño descrita en la memoria (apartado 4.3): mantener pequeña la entrada al circuito. `QNN_USE_PROTT5 = True` permite probarla con ProtT5.

### 10. XGBoost con dos configuraciones

La validación cruzada usaba 300 árboles de profundidad 4 y la prueba fija, 400 de profundidad 5, así que la validación cruzada no estimaba el modelo que se informaba. Ahora hay una sola configuración (400, profundidad 5) para las dos evaluaciones.

### 11. Validación interna y kNN por mutación

La validación de la FCNN (`train_test_split`) y los pliegues de la meta-característica kNN (`KFold`) también mezclaban mutaciones de una misma proteína. Ahora se agrupan por proteína.

## D. Problemas de reproducibilidad

### 12. Rutas, descargas e instalaciones

- Rutas fijas de Colab (`/content/pdbs`, `/mnt/data/...`).
- Dos descargas independientes del PDB.
- Instalaciones con `pip` repartidas por el notebook.
- Las salidas guardadas de la prueba fija de la QNN no se correspondían con el código de su celda.

Ahora hay una configuración única, una sola caché en `cache/` y una celda de instalación que, en Colab, clona el repositorio. El notebook se entrega sin salidas, para que las que aparezcan sean siempre las de la ejecución actual.

### 13. Conflicto entre TensorFlow y PyTorch

Si TensorFlow carga antes su LLVM, crear después un optimizador de PyTorch puede abortar el kernel con `LLVM ERROR: inconsistency in registered CommandLine options`. Ahora el notebook importa `torch._dynamo` al principio, antes que TensorFlow.

---

## Lo que no se ha cambiado (fase de mejora)

Estos puntos no son errores, pero limitan las métricas o el rendimiento. Quedan para la siguiente fase:

- **Codificación de los residuos.** Se usa `ord(letra) − 65`, un número sin significado químico. Una codificación *one-hot* o por propiedades sería más adecuada.
- **Variables sin usar.** No se usan `pH`, `T` ni la estructura 3D: accesibilidad al disolvente, contactos o estructura secundaria.
- **Antisimetría.** No se explota que $\Delta\Delta G(\text{mutado}\to\text{original}) = -\Delta\Delta G(\text{original}\to\text{mutado})$ para aumentar los datos.
- **Velocidad de la QNN.** `EstimatorQNN` tarda del orden de 20 minutos por época. Un circuito de 2 qubits se puede simular de forma exacta en PyTorch en milisegundos.
- **Ajuste de hiperparámetros.** No se ha hecho ningún ajuste sistemático con el nuevo protocolo.
