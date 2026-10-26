"""Compact single-panel figures made for the LaTeX report (written to results/figures/ and
copied to docs/report/figures/).  Usage: python experiments/report_figures.py"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import lip  # noqa: E402
from core.menu import optimal_sd_choice  # noqa: E402

FIG = ROOT / "results" / "figures"
REPORT_FIG = ROOT / "docs" / "report" / "figures"
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})


def optimal_menu_vs_lip():
    """What the dictator of type x receives: Theorem-C menu vs LIP(a*)."""
    a_star = json.loads((ROOT / "results/data/lip_alpha_opt.json").read_text())["a_star"]
    x = np.linspace(0, 1, 2001)
    m1, m2 = optimal_sd_choice(x)
    q1, q2 = lip.bundle_additive(x, a_star)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(x, m1, c="C0", lw=1.6, label="Theorem C menu")
    ax.plot(x, m2, c="C1", lw=1.6)
    ax.plot(x, q1, c="C0", lw=1.1, ls="--", label=r"LIP($a^\star$)")
    ax.plot(x, q2, c="C1", lw=1.1, ls="--")
    ax.text(0.97, 0.93, "item 1", color="C0", ha="right", va="top", fontsize=9, transform=ax.transAxes)
    ax.text(0.03, 0.93, "item 2", color="C1", ha="left", va="top", fontsize=9, transform=ax.transAxes)
    ax.set(xlabel=r"dictator's type $x$", ylabel="share received", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.66), framealpha=0.95)
    plt.tight_layout()
    name = "02_optimal_menu.png"
    plt.savefig(FIG / name, dpi=220)
    plt.close(fig)
    return name


if __name__ == "__main__":
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    for name in (optimal_menu_vs_lip(),):
        shutil.copy(FIG / name, REPORT_FIG / name)
        print("wrote", name)
