"""Unit tests for CAF.apply_cut_chain."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))

from sbnd.cafclasses.parent import CAF


def _make_caf_df(rows: list[dict]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [
            np.arange(len(rows), dtype=int),
            np.zeros(len(rows), dtype=int),
            np.arange(len(rows), dtype=int),
        ],
        names=["run", "subrun", "__ntuple"],
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("cut", "fv"),
            ("cut", "muon"),
            ("x", ""),
        ],
        names=["0", "1"],
    )
    data = {
        ("cut", "fv"): [r["fv"] for r in rows],
        ("cut", "muon"): [r["muon"] for r in rows],
        ("x", ""): np.arange(len(rows), dtype=float),
    }
    return pd.DataFrame(data, index=idx, columns=cols)


class _TestCAF(CAF):
    def key_length(self):
        return 2

    def check_key_structure(self):
        return True

    def clean(self):
        pass


def test_apply_cut_chain_and_semantics() -> None:
    df = _make_caf_df(
        [
            {"fv": True, "muon": True},
            {"fv": True, "muon": False},
            {"fv": False, "muon": True},
        ]
    )
    slc = _TestCAF(df, duplicate_ok=True)
    slc.apply_cut_chain(["precut", "fv", "muon"], verbose=False)
    assert len(slc.data) == 1
    assert slc.data[("x", "")].iloc[0] == 0.0


def test_apply_cut_chain_skips_precut_only() -> None:
    df = _make_caf_df([{"fv": True, "muon": True}, {"fv": False, "muon": True}])
    slc = _TestCAF(df, duplicate_ok=True)
    slc.apply_cut_chain(["precut"], verbose=False)
    assert len(slc.data) == 2


if __name__ == "__main__":
    test_apply_cut_chain_and_semantics()
    test_apply_cut_chain_skips_precut_only()
    print("apply_cut_chain tests OK")
