"""Tests for slim CV pinning before nominal metadata promote."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
_repo = _here.parents[2]
_cafpyana_wd = _repo.parents[1]
for p in (str(_repo), str(_cafpyana_wd)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sbnd.stats.systematics import Systematics

# Import after path setup
from scripts.build_detsys_universes import SLIM_KEYS, _pin_slim_cv_sel


def test_pin_slim_cv_sel_survives_nominal_overwrite() -> None:
    bins = np.linspace(0, 1, 4)
    s = Systematics.for_chunked_build(
        "costheta",
        bins,
        list(SLIM_KEYS),
        pattern=SLIM_KEYS,
        stype="RW",
    )
    cv = s._chunk_acc["cv"]
    cv["sel"] = np.array([1.0, 2.0, 3.0])
    cv["sel_background"] = np.array([0.5, 0.5, 0.5])
    s.finalize_chunked_build()
    _pin_slim_cv_sel(s)
    slim_tot = np.array([1.5, 2.5, 3.5])
    for key in SLIM_KEYS:
        assert key in s.systematics
        assert np.allclose(s.systematics[key]["cv_sel"], slim_tot)

    s.sel = np.zeros(3)
    s.sel_background = np.zeros(3)
    for key in SLIM_KEYS:
        assert np.allclose(s.systematics[key]["cv_sel"], slim_tot)


def test_pin_slim_cv_sel_does_not_clobber_existing() -> None:
    bins = np.linspace(0, 1, 4)
    s = Systematics.for_chunked_build(
        "costheta",
        bins,
        ["xsec"],
        pattern=["xsec"],
        stype="RW",
    )
    cv = s._chunk_acc["cv"]
    cv["sel"] = np.ones(3)
    cv["sel_background"] = np.ones(3)
    s.finalize_chunked_build()
    existing = np.array([9.0, 8.0, 7.0])
    s.systematics["xsec"]["cv_sel"] = existing.copy()
    _pin_slim_cv_sel(s, keys=("xsec",))
    assert np.allclose(s.systematics["xsec"]["cv_sel"], existing)
