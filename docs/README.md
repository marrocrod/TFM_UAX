# Documentación

| Documento | Descripción |
|---|---|
| [`Memoria.pdf`](Memoria.pdf) | Memoria completa del TFM (98 páginas, septiembre de 2025): estado del arte, metodología, resultados y conclusiones |
| [`metodologia.md`](metodologia.md) | Descripción técnica del pipeline tal y como está implementado: datos, características, protocolo de evaluación y los tres modelos con sus hiperparámetros |
| [`correcciones.md`](correcciones.md) | Fase 1: los 13 errores de la versión original, con el código antes y después |
| [`mejoras.md`](mejoras.md) | Fase 2: las 8 mejoras de características, entrenamiento y simulación, con su interruptor y lo que se ha verificado |
| [`figures/`](figures/) | Figuras del README, generadas con [`scripts/generar_figuras.py`](../scripts/generar_figuras.py) |

## Figuras

| Fichero | Contenido |
|---|---|
| `pipeline.png` | Flujo completo: S2648 → descriptores, estructura 3D, condiciones y ProtT5 (XGBoost y FCNN; la QNN, sin ProtT5) → evaluación agrupada |
| `circuito_qnn.png` | Dos de los ocho bloques del circuito de 2 qubits con recarga de datos |
| `datos.png` | Distribución de ΔΔG y número de mutaciones por *cluster* de proteína |
| `resultados.png` | PCC y RMSE por modelo. Usa los resultados de la versión corregida si existen y, si no, los de la versión original |
