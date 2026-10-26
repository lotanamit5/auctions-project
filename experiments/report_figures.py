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
    """What the dictator of type x receives: optimal menu vs LIP(a*)."""
    a_star = json.loads((ROOT / "results/data/lip_alpha_opt.json").read_text())["a_star"]
    x = np.linspace(0, 1, 2001)
    m1, m2 = optimal_sd_choice(x)
    q1, q2 = lip.bundle_additive(x, a_star)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(x, m1, c="C0", lw=1.6, label="optimal menu")
    ax.plot(x, m2, c="C1", lw=1.6)
    ax.plot(x, q1, c="C0", lw=1.1, ls="--", label=r"LIP($a^\star$)")
    ax.plot(x, q2, c="C1", lw=1.1, ls="--")
    ax.text(0.97, 0.93, "item 1", color="C0", ha="right", va="top", fontsize=9, transform=ax.transAxes)
    ax.text(0.03, 0.93, "item 2", color="C1", ha="left", va="top", fontsize=9, transform=ax.transAxes)
    ax.set(xlabel=r"dictator's type $x$", ylabel="share received", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.66), framealpha=0.95)
    plt.tight_layout()
    name = "02_optimal_menu"
    plt.savefig(FIG / f"{name}.png", dpi=220)
    plt.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    return name


def smd_vs_bounds():
    """Section 4: SMD against the menu bound (fixed-menu bound) and the LP bound on all SP mechanisms."""
    import pandas as pd
    from core import split

    lpc = pd.read_csv(ROOT / "results/data/lp_eps_curve.csv")
    lp_best = lpc.sort_values("N").groupby("eps", as_index=False).last().sort_values("eps")
    lp_best = lp_best[lp_best.eps <= 1]
    nes = pd.read_csv(ROOT / "results/data/eps_lip_best_nested.csv").drop_duplicates("eps", keep="last")
    nes = nes[nes.eps <= 1].sort_values("eps")
    e = np.linspace(0.001, 1, 400)
    menu = np.minimum(np.where(e <= 0.5, (5 + 2 * e) / 6, np.inf), (1 + e) / (1 + 2 * e - e * e))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(lp_best.eps, lp_best.alpha, "k-", lw=1.6, label="all SP (LP bound)")
    ax.plot(e, menu, "--", c="C3", lw=1.4, label="fixed menus (upper bound)")
    ax.plot(e, split.ratio_closed_form(e), c="C2", lw=1.6, label="SMD")
    ax.plot(nes.eps, nes.lower, "s", c="C1", ms=3.5, label=r"nested LIP($a^\star$)")
    ax.set(xlabel=r"$\varepsilon$", ylabel="competitive ratio", xlim=(0, 1), ylim=(0.70, 1.01))
    ax.set_title(r"Competitive ratio vs. $\varepsilon$", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    plt.tight_layout()
    name = "04_smd"
    plt.savefig(FIG / f"{name}.png", dpi=220)
    plt.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    return name


if __name__ == "__main__":
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    for name in (optimal_menu_vs_lip(), smd_vs_bounds()):
        shutil.copy(FIG / f"{name}.pdf", REPORT_FIG / f"{name}.pdf")
        print("wrote", name)
