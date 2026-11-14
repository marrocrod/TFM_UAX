# Mejoras sobre la versión corregida

La [fase de corrección](correcciones.md) dejó el pipeline sin errores metodológicos, pero sin tocar los modelos ni las características. Esta segunda fase añade ocho mejoras a [`notebook/estabilidad_proteica.ipynb`](../notebook/estabilidad_proteica.ipynb).

> [!NOTE]
> Todavía no hay cifras de esta versión: hace falta una ejecución completa con datos reales. Las mejoras de características están motivadas, pero **su efecto sobre las métricas está por medir**. Cada una tiene un interruptor en el Bloque 2 del notebook para poder medirlo.

## Resumen

| # | Mejora | Interruptor | Dónde |
|---|---|---|---|
| 1 | Simulación exacta de la QNN en PyTorch | — | Bloque 11 |
| 2 | ProtT5 de la secuencia mutada | `USE_PROTT5_DIFF` | Bloque 6 |
| 3 | Codificación química de los residuos | — | Bloque 5 |
| 4 | Características de la estructura 3D | `USE_STRUCTURE` | Bloque 4 |
| 5 | pH y temperatura | `USE_CONDITIONS` | Bloque 5 |
| 6 | Antisimetría del ΔΔG | `AUGMENT_ANTISYMMETRY`, `ANTISYMMETRIC_PREDICTION` | Bloque 8 |
| 7 | Ángulos del circuito acotados | — | Bloque 12 |
| 8 | Ablación de la QNN | — | Bloque 13 |

Con todos los interruptores en `False` se obtiene la línea base: la versión corregida, con la nueva codificación de residuos.

---

## 1. Simulación exacta de la QNN en PyTorch

**Antes.** `EstimatorQNN` evaluaba el circuito muestra a muestra y tardaba del orden de 20 minutos por época en Colab. Por eso el entrenamiento se limitaba a 12 – 18 épocas.

**Ahora.** El estado de 2 qubits son 4 amplitudes complejas. La clase `ReuploadingCircuit` aplica las puertas ($R_Y$, $R_Z$ y CNOT) como operaciones de PyTorch sobre todo el lote, y el gradiente sale por *autograd*. El resultado es exacto, igual que con `StatevectorEstimator`.

- **Verificación.** El Bloque 11 construye el mismo circuito en Qiskit y comprueba, con entradas y pesos aleatorios, que los valores esperados coinciden (diferencia del orden de 10⁻⁷). Si no coinciden, el notebook se detiene.
- **Tiempo.** Un entrenamiento con tamaño real (unas 3 400 filas) tarda segundos en CPU.
- **Consecuencias.** Se entrena hasta 60 épocas con parada temprana, y se elimina la dependencia de `qiskit-machine-learning`.

El simulador es general en el número de qubits, así que `QNN_QUBITS` se puede cambiar sin tocar el código.

## 2. ProtT5 de la secuencia mutada

**Antes.** Solo se usaba el embedding del residuo original. Todas las mutaciones de una misma posición (por ejemplo, L27A y L27N) tenían el mismo vector de 1 024 dimensiones: el embedding describía el entorno, pero no la mutación.

**Ahora.** Para cada mutación se calculan dos embeddings en la posición mutada: $e_{wt}$ con la secuencia original y $e_{mut}$ con la secuencia mutada. El modelo recibe $e_{wt}$ y la diferencia $e_{mut} - e_{wt}$ (2 048 dimensiones).

**Coste.** Una pasada de ProtT5 por mutación en lugar de una por cadena. Se guarda en `cache/prott5/` por cadena y solo se calcula una vez.

## 3. Codificación química de los residuos

**Antes.** Los residuos se codificaban con `ord(letra) − 65`, que impone un orden alfabético sin significado químico.

**Ahora.**

| Bloque | Dim. | Contenido |
|---|---:|---|
| Residuos | 40 | *one-hot* del residuo de partida y del de llegada |
| Posición | 1 | índice / (longitud − 1) |
| Vecinos | 42 | volumen, hidrofobicidad y flexibilidad de cada vecino a ±7 |
| Composición | 20 | recuento de cada aminoácido en la ventana |
| Fisicoquímica | 3 | Δ volumen, Δ hidrofobicidad y Δ flexibilidad |

Se eliminan dos variables redundantes: `mutation_type` (ya contenida en los dos *one-hot*) y el índice absoluto (ya contenido en la posición relativa).

## 4. Características de la estructura 3D

Las estructuras ya se descargaban, pero solo para extraer la secuencia. Ahora se calcula un perfil del residuo mutado sobre la cadena aislada, sin hidrógenos ni ligandos, solo con Biopython:

| Variable | Definición |
|---|---|
| `rsa` | área accesible al disolvente (Shrake-Rupley) dividida entre el máximo teórico del aminoácido (Tien *et al.*, 2013) |
| `contacts_8`, `contacts_12` | número de Cα a menos de 8 y 12 Å |
| `depth` | distancia del Cα al centro de la cadena, dividida entre el radio de giro |

La motivación es que las mutaciones en el núcleo enterrado suelen desestabilizar más que las de la superficie.

## 5. pH y temperatura

S2648 incluye el pH y la temperatura de cada medida, que no se usaban. Ahora entran como dos variables.

## 6. Antisimetría del ΔΔG

Por definición, la mutación inversa tiene un ΔΔG de signo contrario:

$$\Delta\Delta G_{B\to A} = -\,\Delta\Delta G_{A\to B}$$

Se usa de dos formas, aplicadas igual a todos los modelos dentro de `evaluate()`:

- **En el entrenamiento** (`AUGMENT_ANTISYMMETRY`): a cada mutación de entrenamiento se le añade su inversa con $-\Delta\Delta G$. El conjunto de entrenamiento se duplica.
- **En la predicción** (`ANTISYMMETRIC_PREDICTION`): se predice la mutación directa y la inversa y se devuelve $\tfrac12\,[f(\text{directa}) - f(\text{inversa})]$.

Las métricas se calculan solo sobre las mutaciones directas, que son las experimentales.

**Cómo se construye la inversa.** Se intercambian los *one-hot*, se cambia el signo de las diferencias fisicoquímicas, $e_{mut}$ pasa a ser el embedding de partida y la diferencia de ProtT5 cambia de signo.

**Limitación.** Para la mutación inversa se usan la estructura, el pH y la temperatura de la proteína original, porque no existe estructura experimental del mutante. Es una aproximación.

## 7. Ángulos del circuito acotados

**Antes.** Las componentes PCA entraban sin acotar como ángulos de $R_Y$. Como la rotación es periódica, dos valores muy distintos podían producir la misma rotación.

**Ahora.** Las componentes se estandarizan y se llevan a $(-\pi, \pi)$ con $2\arctan(z)$. La rama clásica sigue recibiendo las componentes estandarizadas, sin esa transformación.

## 8. Ablación de la QNN

Se evalúa un cuarto modelo, `QNN (lineal)`: la misma red, los mismos datos, las mismas particiones y la misma semilla, cambiando solo el circuito por una capa `Linear(16 → 3)`.

| | Circuito | Capa lineal |
|---|---|---|
| Parámetros de la rama | 32 | 51 |
| Resto de la red | idéntico | idéntico |

Sirve para responder a la pregunta que la versión anterior no podía responder: si el circuito aporta algo medible frente a un componente clásico del mismo tamaño. El notebook muestra la diferencia de PCC partición a partición.

---

## Limpieza del notebook

- La antisimetría se aplica en un único sitio (`evaluate()`), no en cada modelo.
- Se elimina el reajuste extra de XGBoost que solo servía para dibujar curvas. En su lugar se muestra la importancia de cada bloque de características.
- La QNN y su ablación comparten una única función de entrenamiento.
- El notebook pasa a tener 8 partes y 14 bloques, con las características en su propia parte y el protocolo de evaluación antes de los modelos.

## Qué se ha verificado y qué no

**Verificado**, con estructuras y embeddings sintéticos:

- El notebook se ejecuta de principio a fin, con todas las mejoras activadas y con todas desactivadas.
- El simulador coincide con Qiskit con 2 y con 3 qubits.
- Con el conjunto completo se localizan las mutaciones, incluidas las 4 con código de inserción.
- La mutación inversa es coherente: *one-hot* intercambiado y diferencias de signo contrario.

**Sin verificar:** el efecto de las mejoras sobre las métricas. Requiere la ejecución completa con las estructuras del PDB y ProtT5.

## Pendiente

- Ejecución completa y medida del efecto de cada mejora, activándolas de una en una.
- Ajuste de hiperparámetros con el protocolo agrupado.
- Decidir el tamaño del circuito: el código usa 2 qubits y 16 componentes; la memoria describe 4 qubits y 24 componentes.
