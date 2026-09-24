"""The notebook must read its audit gates from analysis/cells.py, never redefine them.

cells.py holds the single definition of the three audit levels. If the notebook
set a band of its own, the same quantity could be computed two ways and the two
would drift apart silently.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))


def test_gates_are_the_published_ones():
    import cells

    assert cells.EOD_BAND == 0.1
    assert cells.DI_BAND == (0.8, 1.2)
    assert cells.SUPPORT_MARGIN == 0.01
    assert cells.COLLAPSE_FPR_MARGIN == 0.1


def test_notebook_reads_the_shared_module():
    source = "\n".join("".join(cell["source"])
                       for cell in json.loads((ROOT / "analysis.ipynb").read_text())["cells"]
                       if cell["cell_type"] == "code")
    assert "import cells as C" in source
    for gate in ("EOD_BAND", "DI_BAND", "SUPPORT_MARGIN", "COLLAPSE_FPR_MARGIN"):
        assert not re.search(rf"^{gate}\s*=", source, flags=re.M), \
            f"the notebook redefines {gate} instead of reading it from cells.py"


def test_collapse_gate_rejects_a_degenerate_cell():
    """A cell answering one label throughout is collapsed, however fair it looks."""
    import numpy as np
    import cells

    y_true = np.array([1, 0] * 50)
    sensitive = np.array([1, 1, 0, 0] * 25)
    always_positive = np.ones((1, 100), dtype=int)

    _, raw, collapsed, genuine = cells.genuine_matrix(y_true, sensitive, 1, always_positive)
    assert raw[0], "a constant predictor scores equal rates, so it passes the raw bands"
    assert collapsed[0], "a constant predictor must be flagged as collapsed"
    assert not genuine[0], "non-collapsed GF must reject it"
