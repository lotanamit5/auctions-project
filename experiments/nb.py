"""Tiny helper to write notebooks from Python scripts and execute them in place.

Usage in experiments/nb_XX_*.py:
    from nb import NB
    nb = NB("01_replication")
    nb.md("# Title ...")
    nb.code('''import numpy as np''')
    nb.save(execute=True)
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

PREAMBLE = """
import sys, warnings
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / 'core').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.figsize': (7, 4)})
FIG = ROOT / 'results' / 'figures'; DATA = ROOT / 'results' / 'data'
"""


class NB:
    def __init__(self, name: str):
        self.name = name
        self.cells = []
        self.code(PREAMBLE)

    def md(self, text: str):
        self.cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(self, src: str):
        self.cells.append(nbformat.v4.new_code_cell(textwrap.dedent(src).strip()))

    def save(self, execute: bool = True, timeout: int = 3600):
        nb = nbformat.v4.new_notebook(cells=self.cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB_DIR / f"{self.name}.ipynb"
        if execute:
            client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                                    resources={"metadata": {"path": str(NB_DIR)}})
            client.execute()
        nbformat.write(nb, path)
        print("wrote", path, file=sys.stderr)
        return path
