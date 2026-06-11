"""Regression: nominal MC concat must not duplicate aux rows across chunks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))

from sbnd.general.utils import concat_with_ntuple_offset, offset_ntuple_index


def _make_slice_df(ntuple_vals: list[int]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [
            np.zeros(len(ntuple_vals), dtype=int),
            np.zeros(len(ntuple_vals), dtype=int),
            np.array(ntuple_vals, dtype=int),
        ],
        names=["run", "subrun", "__ntuple"],
    )
    return pd.DataFrame({"x": np.arange(len(ntuple_vals), dtype=float)}, index=idx)


def test_mc_only_concat_is_unique() -> None:
    parts = [_make_slice_df([1, 2, 3]), _make_slice_df([1, 2, 3])]
    combined = concat_with_ntuple_offset(parts)
    assert not combined.index.duplicated().any()


def test_aux_once_after_mc_concat_is_unique() -> None:
    mc = concat_with_ntuple_offset([_make_slice_df([10, 11]), _make_slice_df([10, 11])])
    aux = offset_ntuple_index(_make_slice_df([100, 101]), int(1e7))
    combined = pd.concat([mc, aux])
    assert not combined.index.duplicated().any()


def test_aux_per_chunk_before_concat_duplicates() -> None:
    bad_parts = []
    for _ in range(2):
        mc = _make_slice_df([10, 11])
        aux = offset_ntuple_index(_make_slice_df([100, 101]), int(1e7))
        bad_parts.append(pd.concat([mc, aux]))
    combined = pd.concat(bad_parts)
    assert combined.index.duplicated().any()


def test_global_file_offset_avoids_chunk_reset_collisions() -> None:
    """Simulate two chunk loads that reset per-chunk file offsets without a global base."""
    chunk_size = 2

    def _chunk_part(file_base: int) -> pd.DataFrame:
        parts = [
            offset_ntuple_index(_make_slice_df([5, 6]), file_base + i)
            for i in range(chunk_size)
        ]
        return pd.concat(parts)

    bad = pd.concat([_chunk_part(0), _chunk_part(0)])
    assert bad.index.duplicated().any()

    good = pd.concat([_chunk_part(0), _chunk_part(chunk_size)])
    assert not good.index.duplicated().any()


if __name__ == "__main__":
    test_mc_only_concat_is_unique()
    test_aux_once_after_mc_concat_is_unique()
    test_aux_per_chunk_before_concat_duplicates()
    test_global_file_offset_avoids_chunk_reset_collisions()
    print("nominal index offset tests OK")
