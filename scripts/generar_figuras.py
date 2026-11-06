"""Draw the README figures in docs/figures/.

- pipeline.png        : data → features → three models → grouped evaluation
- circuito_qnn.png    : the 2-qubit re-uploading circuit (Qiskit drawing)
- datos.png           : ΔΔG distribution and mutations per protein cluster in S2648
- resultados.png      : PCC and RMSE per model. Uses results/resultados.json (corrected
                        version) when it exists, otherwise results/version_original.json.

Run from the repository root:
    python scripts/generar_figuras.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, TEAL, GRAY, LGRAY = "#219BC3", "#00566C", "#858591", "#C0C0C8"
QUANT, TINT_Q, TINT_B, TINT_G = "#7D3C98", "#F3EAF7", "#E8F5FA", "#F1F2F4"
MODEL_COLORS = {"XGBoost": BLUE, "FCNN": TEAL, "QNN": QUANT}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": LGRAY,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ·", name)


def box(ax, x, y, w, h, title, sub="", fc=TINT_B, tc=TEAL, sc=GRAY, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h * (0.64 if sub else 0.5), title, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", color=sc, fontsize=fs - 2.5)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
    box(ax, 0.1, 1.45, 2.0, 1.3, "S2648", "2 648 mutaciones\n131 estructuras PDB")
    arrow(ax, 2.15, 2.1, 2.55, 2.1)
    box(ax, 2.6, 2.35, 2.5, 1.15, "Descriptores (42)", "residuos · vecinos ±7\nΔ volumen, hidrof., flexib.", fs=10.5)
    box(ax, 2.6, 0.7, 2.5, 1.15, "ProtT5-XL (1 024)", "embedding del\nresiduo mutado", fs=10.5)
    arrow(ax, 5.15, 2.9, 5.55, 2.25); arrow(ax, 5.15, 1.3, 5.55, 1.95)
    box(ax, 5.6, 1.45, 1.6, 1.3, "X", "1 066 dim.\n(QNN: 68)", fc=TINT_G)
    models = [("XGBoost", "400 árboles", TINT_B, TEAL, GRAY), ("FCNN", "320-384-256 · kNN", TINT_B, TEAL, GRAY),
              ("QNN", "68 var. sin ProtT5 → 2 qubits", QUANT, "white", "white")]
    for k, (t, s, fc, tc, sc) in enumerate(models):
        y = 2.95 - k * 1.25
        arrow(ax, 7.25, 2.1, 7.75, y + 0.5)
        box(ax, 7.8, y, 2.1, 1.0, t, s, fc, tc, sc, fs=10.5)
        arrow(ax, 9.95, y + 0.5, 10.45, 2.1)
    box(ax, 10.5, 1.2, 2.4, 1.8, "Evaluación", "GroupKFold(5) por proteína\n+ prueba fija 20 %\nPCC · RMSE · MAE", fc=TINT_G)
    save(fig, "pipeline.png")


def fig_circuito():
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    n_q, blocks = 2, 2
    x = ParameterVector("x", blocks * n_q); wz = ParameterVector("wz", blocks * n_q); wy = ParameterVector("wy", blocks * n_q)
    qc = QuantumCircuit(n_q)
    for b in range(blocks):
        for q in range(n_q):
            qc.ry(x[b * n_q + q], q)
        qc.cx(0, 1)
        for q in range(n_q):
            qc.rz(wz[b * n_q + q], q); qc.ry(wy[b * n_q + q], q)
        qc.barrier()
    style = {"displaycolor": {"ry": ("#219BC3", "#FFFFFF"), "rz": ("#7D3C98", "#FFFFFF"),
                              "cx": ("#00566C", "#FFFFFF")}, "fontsize": 12, "subfontsize": 9}
    fig = qc.draw(output="mpl", style=style, fold=-1, plot_barriers=True)
    fig.suptitle("Bloques 1 y 2 de 8 (se repiten con x[4…15])", x=0.02, ha="left", color=TEAL, fontsize=12)
    save(fig, "circuito_qnn.png")


def fig_datos():
    df = pd.read_csv(ROOT / "data" / "s2648.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    a1.hist(df["DDG"], bins=40, color=BLUE, alpha=0.9)
    a1.axvline(0, color=GRAY, ls="--", lw=1)
    a1.set_xlabel("ΔΔG experimental (kcal/mol)"); a1.set_ylabel("Mutaciones")
    a1.set_title(f"Distribución de ΔΔG · {(df['DDG'] < 0).mean() * 100:.0f} % con ΔΔG < 0",
                 color=TEAL, fontweight="bold", loc="left")
    per = df.groupby("CLID").size().sort_values(ascending=False)
    top = per.head(15)
    a2.barh(range(len(top))[::-1], top.values, color=[QUANT if i == 0 else TEAL for i in range(len(top))])
    a2.set_yticks(range(len(top))[::-1]); a2.set_yticklabels(top.index, fontsize=8.5)
    a2.set_xlabel("Mutaciones")
    a2.set_title(f"Mutaciones por cluster ({per.size} clusters)", color=TEAL, fontweight="bold", loc="left")
    save(fig, "datos.png")


def fig_resultados():
    corrected = ROOT / "results" / "resultados.json"
    if corrected.exists():
        data, label = json.load(open(corrected, encoding="utf-8")), "versión corregida · GroupKFold por proteína"
    else:
        data, label = json.load(open(ROOT / "results" / "version_original.json", encoding="utf-8")), \
            "versión original · KFold por mutación (optimista)"
    models = list(data["modelos"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["PCC", "RMSE"]):
        cv_m, cv_s, te = [], [], []
        for m in models:
            r = data["modelos"][m]
            if metric in r["cv"]:
                mu, sd = r["cv"][metric]
            else:                                   # original runs reported MSE: RMSE per fold
                vals = [f["MSE"] ** 0.5 for f in r["cv_folds"]]
                mu, sd = float(np.mean(vals)), float(np.std(vals))
            cv_m.append(mu); cv_s.append(sd); te.append(r["test"][metric])
        xs = np.arange(len(models))
        cols = [MODEL_COLORS.get(m, GRAY) for m in models]
        ax.bar(xs - 0.18, cv_m, 0.34, yerr=cv_s, capsize=4, color=cols, label="validación cruzada")
        ax.bar(xs + 0.18, te, 0.34, color=cols, alpha=0.45, label="prueba fija")
        for x, v, sd in zip(xs - 0.18, cv_m, cv_s):
            ax.text(x, v + sd + 0.02, f"{v:.2f}".replace(".", ","), ha="center", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(models)
        ax.set_title(metric + (" (kcal/mol)" if metric == "RMSE" else ""), color=TEAL, fontweight="bold", loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(label, x=0.01, ha="left", color=GRAY, fontsize=10, style="italic")
    save(fig, "resultados.png")


if __name__ == "__main__":
    print("Figuras en docs/figures/:")
    for f in (fig_pipeline, fig_circuito, fig_datos, fig_resultados):
        f()
