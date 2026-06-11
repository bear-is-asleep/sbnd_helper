"""cosmic_data used for event_cv in compute_covariances, not metadata nominal."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

_here = Path(__file__).resolve()
_repo = _here.parents[2]
_cafpyana_wd = _repo.parents[1]
for p in (str(_repo), str(_cafpyana_wd)):
    if p not in sys.path:
        sys.path.insert(0, p)

for _mod in ("matplotlib", "matplotlib.pyplot", "numba", "tqdm"):
    sys.modules.setdefault(_mod, MagicMock())
_plotlib = types.ModuleType("sbnd.plotlibrary")
_plotlib.makeplot = MagicMock()
sys.modules["sbnd.plotlibrary"] = _plotlib
sys.modules["sbnd.plotlibrary.makeplot"] = _plotlib.makeplot
_general = types.ModuleType("sbnd.general")
_general.plotters = MagicMock()
_general.utils = MagicMock()
sys.modules["sbnd.general"] = _general
sys.modules["sbnd.general.plotters"] = _general.plotters
sys.modules["sbnd.general.utils"] = _general.utils

from sbnd.stats.systematics import Systematics


def test_compute_covariances_uses_cosmic_data_cv() -> None:
    bins = np.linspace(0, 1, 4)
    n = 3
    s = Systematics.__new__(Systematics)
    s.variable_name = "costheta"
    s.bins = bins
    s.sel = np.ones(n) * 100.0
    s.sel_background = np.ones(n) * 50.0
    s.sigma_tilde = None
    cosmic_cv_sel = np.ones(n) * 10.0
    cosmic_cv_bg = np.ones(n) * 5.0
    variation = np.ones(n) * 12.0
    s.systematics = {
        "cosmic": {
            "name": "cosmic",
            "type": "cosmic",
            "variation": "unisim",
            "rank": 1,
            "sel": [variation],
            "sel_background": [],
            "sigma_tilde": [],
            "response": [],
            "eff_truth": [],
            "cv_sel": None,
            "cv_sigma_tilde": None,
            "cols": None,
            "col_names": None,
            "label": "cosmic",
            "description": "",
            "order": None,
            "color": None,
        },
        "cosmic_data": {
            "name": "cosmic_data",
            "type": "RW",
            "variation": "self",
            "sel": cosmic_cv_sel,
            "sel_background": cosmic_cv_bg,
            "sigma_tilde": None,
        },
    }

    captured: dict = {}

    def fake_construct_covariance(cv_evts, var_evts, **kwargs):
        captured["cv"] = np.asarray(cv_evts, dtype=float).copy()
        captured["var"] = np.asarray(var_evts, dtype=float).copy()
        n_bins = len(cv_evts)
        z = np.zeros((n_bins, n_bins))
        frac = np.full(n_bins, 0.1)
        return z, z, z, z, z, frac, None

    with patch("sbnd.stats.systematics.construct_covariance", side_effect=fake_construct_covariance):
        s.compute_covariances(keys=["cosmic"], compute_xsec_cov=False)

    expected_cv = cosmic_cv_sel + cosmic_cv_bg
    assert np.allclose(captured["cv"], expected_cv)
    assert not np.allclose(captured["cv"], s.sel + s.sel_background)
    assert np.allclose(captured["var"], variation)
