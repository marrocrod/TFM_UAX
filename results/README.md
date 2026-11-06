# Resultados

| Fichero | Contenido |
|---|---|
| [`version_original.json`](version_original.json) | Métricas de la versión original, transcritas de las salidas guardadas de [`notebook/version_original/Notebook_final.ipynb`](../notebook/version_original/Notebook_final.ipynb). Se obtuvieron con particiones por mutación y otros errores descritos en [`docs/correcciones.md`](../docs/correcciones.md): **son optimistas** y no comparables con las de la versión corregida |
| `resultados.json` | *(se genera al ejecutar el notebook corregido)* Métricas por partición, medias, prueba fija y configuración |
| `predicciones_prueba.csv` | *(se genera)* $\Delta\Delta G$ experimental y predicho por cada modelo en la prueba fija |
| `comparacion.png`, `dispersion_prueba.png` | *(se generan)* Comparación de modelos y predicho frente a experimental |
| `metricas_cv.csv`, `metricas_resumen.csv` | *(se generan con [`scripts/actualizar_resultados.py`](../scripts/actualizar_resultados.py))* Métricas por partición y resumen por modelo |

Unidades: $\Delta\Delta G$, RMSE y MAE en kcal/mol; PCC es la correlación de Pearson.
