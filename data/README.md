# Datos

**`s2648.csv`**: conjunto S2648 de mutaciones puntuales con $\Delta\Delta G$ experimental (Dehouck *et al.*, *Bioinformatics* 25 (19), 2009), en la versión distribuida con este trabajo.

| Columna | Descripción |
|---|---|
| `CLID` | *Cluster* de homología de la proteína: se usa para agrupar las particiones |
| `PDB`, `CHAIN` | Estructura y cadena de referencia |
| `MUT` | Mutación: residuo original, posición con la numeración del PDB (con código de inserción si lo hay) y residuo mutado, p. ej. `C30S` o `L27CN` |
| `DDG` | $\Delta\Delta G$ experimental (kcal/mol) |
| `pH`, `T` | Condiciones del experimento (°C) |

2 648 mutaciones · 131 estructuras · 113 *clusters*. Las estructuras se descargan del PDB al ejecutar el notebook y se guardan en `cache/pdb/`.
