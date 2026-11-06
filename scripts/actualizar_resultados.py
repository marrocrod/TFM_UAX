"""Turn the results of a full run of the corrected notebook into tidy tables and README text.

Reads results/resultados.json (written by Part VIII of notebook/estabilidad_proteica.ipynb) and:
  - writes results/metricas_cv.csv (one row per model and fold) and results/metricas_resumen.csv;
  - replaces the results table of README.md between the markers
    <!-- RESULTADOS:INICIO --> and <!-- RESULTADOS:FIN -->;
  - redraws the figures (scripts/generar_figuras.py picks the corrected results up automatically).

Usage (from the repository root, after copying the notebook outputs into results/):
    python scripts/actualizar_resultados.py
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"
START, END = "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIN -->"


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def main():
    path = RESULTS / "resultados.json"
    if not path.exists():
        sys.exit("results/resultados.json no existe: ejecuta el notebook completo (RUN_MODE = 'full') primero.")
    data = json.load(open(path, encoding="utf-8"))
    if data.get("run_mode") != "full":
        sys.exit(f"results/resultados.json viene de RUN_MODE = '{data.get('run_mode')}', no de una ejecución completa.")
    models = data["modelos"]

    with open(RESULTS / "metricas_cv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "particion", "PCC", "RMSE", "MSE", "MAE"])
        for m, r in models.items():
            for i, fold in enumerate(r["cv_folds"], 1):
                w.writerow([m, i, fold["PCC"], fold["RMSE"], fold["MSE"], fold["MAE"]])
            w.writerow([m, "prueba fija", r["test"]["PCC"], r["test"]["RMSE"], r["test"]["MSE"], r["test"]["MAE"]])

    with open(RESULTS / "metricas_resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "PCC_cv_media", "PCC_cv_desv", "RMSE_cv_media", "RMSE_cv_desv",
                    "MAE_cv_media", "MAE_cv_desv", "PCC_prueba", "RMSE_prueba", "MAE_prueba"])
        for m, r in models.items():
            cv = r["cv"]
            w.writerow([m, *cv["PCC"], *cv["RMSE"], *cv["MAE"], r["test"]["PCC"], r["test"]["RMSE"], r["test"]["MAE"]])

    d = data["datos"]
    lines = [START, "",
             f"**Versión corregida**: validación cruzada agrupada por proteína (`GroupKFold(5)`) y prueba fija con "
             f"{d['prueba_mutaciones']} mutaciones de {d['prueba_clusters']} *clusters* no vistos "
             f"({d['mutaciones']} mutaciones válidas en total).", "",
             "| Modelo | PCC (CV) | RMSE (CV) | MAE (CV) | PCC (prueba) | RMSE (prueba) | MAE (prueba) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m, r in models.items():
        cv = r["cv"]
        lines.append(f"| {m} | {fmt(cv['PCC'][0])} ± {fmt(cv['PCC'][1])} | {fmt(cv['RMSE'][0])} ± {fmt(cv['RMSE'][1])} "
                     f"| {fmt(cv['MAE'][0])} ± {fmt(cv['MAE'][1])} | {fmt(r['test']['PCC'])} | {fmt(r['test']['RMSE'])} "
                     f"| {fmt(r['test']['MAE'])} |")
    lines += ["", "<sub>RMSE y MAE en kcal/mol. Detalle por partición en "
              "[`results/metricas_cv.csv`](results/metricas_cv.csv).</sub>", "", END]

    readme = README.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        sys.exit("README.md no tiene los marcadores de resultados.")
    readme = re.sub(re.escape(START) + r".*?" + re.escape(END), "\n".join(lines), readme, flags=re.S)
    README.write_text(readme, encoding="utf-8")

    subprocess.run([sys.executable, str(ROOT / "scripts" / "generar_figuras.py")], check=True)
    print("Actualizado: README.md, results/metricas_cv.csv, results/metricas_resumen.csv y docs/figures/resultados.png")


if __name__ == "__main__":
    main()
