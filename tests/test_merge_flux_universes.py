"""Tests for CAFSlice.merge_flux_universes and the flux preprocess stage."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve()
_repo = _here.parents[2]
_cafpyana_wd = _repo.parents[1]
for p in (str(_repo), str(_cafpyana_wd)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sbnd.cafclasses.slice import CAFSlice, run_preprocess

NUNIV = 100
N_COMPONENTS = 2
COMPONENTS = ("expskin_Flux", "kplus_Flux")


def _col(name: str, univ: int) -> tuple:
    return ("truth", name, f"univ_{univ}", "")


def _make_slice(weight: float = 2.0, n_events: int = 2) -> CAFSlice:
    cols = [("reco", "genweight", "", "")]
    for comp in COMPONENTS:
        for u in range(NUNIV):
            cols.append(_col(comp, u))
    index = pd.MultiIndex.from_tuples(
        [(0, 0, i) for i in range(n_events)],
        names=["__ntuple", "entry", "subentry"],
    )
    data = {}
    for col in cols:
        if col[0] == "reco":
            data[col] = np.ones(n_events)
        else:
            data[col] = np.full(n_events, weight)
    df = pd.DataFrame(data, index=index)
    df.columns = pd.MultiIndex.from_tuples(cols)
    return CAFSlice(df, duplicate_ok=True)


def test_merge_flux_universes_drops_source_and_adds_combine() -> None:
    slc = _make_slice(weight=2.0)
    n_before = slc.data.shape[1]
    expected_after = n_before - (N_COMPONENTS * NUNIV) + NUNIV

    slc.merge_flux_universes()

    assert slc.data.shape[1] == expected_after
    assert not any("_Flux" in col[1] and col[1] != "Flux_combine" for col in slc.data.columns)
    assert sum(col[1] == "Flux_combine" for col in slc.data.columns) == NUNIV

    combine_col = ("truth", "Flux_combine", "univ_0", "")
    expected = 2.0 ** N_COMPONENTS
    assert np.allclose(slc.data.loc[:, combine_col].values, expected)


def test_merge_flux_universes_idempotent() -> None:
    slc = _make_slice()
    slc.merge_flux_universes()
    n_after = slc.data.shape[1]
    slc.merge_flux_universes()
    assert slc.data.shape[1] == n_after


def test_run_preprocess_flux_stage() -> None:
    slc = _make_slice()
    n_before = slc.data.shape[1]
    run_preprocess(slc, stages=("flux",))
    assert slc.data.shape[1] == n_before - (N_COMPONENTS * NUNIV) + NUNIV


def test_load_preprocess_flux_string_stage() -> None:
    """preprocess=('flux') is a str in Python; load must not treat it as True -> det only."""
    from unittest.mock import patch

    slc = _make_slice()
    flux_stage = ("flux")  # str, not tuple — notebook footgun
    assert isinstance(flux_stage, str)
    with patch("sbnd.cafclasses.parent.CAF._load_combined", return_value=slc):
        loaded = CAFSlice.load("dummy.h5", preprocess=flux_stage)
    assert sum(col[1] == "Flux_combine" for col in loaded.data.columns) == NUNIV


if __name__ == "__main__":
    test_merge_flux_universes_drops_source_and_adds_combine()
    test_merge_flux_universes_idempotent()
    test_run_preprocess_flux_stage()
    test_load_preprocess_flux_string_stage()
    print("all tests passed")
