"""Tests for per-variation det event matching (triple-key: file_index, __ntuple, entry)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve()
_repo = _here.parents[2]
_cafpyana_wd = _repo.parents[1]
for p in (str(_repo), str(_cafpyana_wd)):
    if p not in sys.path:
        sys.path.insert(0, p)

from detsys_config import build_config, build_file_map, subsample_file_map
from detsys_det_match import (
    EVENT_CSV_COLUMNS,
    NTUPLE_FILE_STEP,
    build_variation_scaling_entry,
    filter_slice_to_events,
    pairwise_common_events,
    pot_scaling_path,
    prepare_mcnu_event_index,
    runs_csv_path,
    write_pot_scaling_json,
    write_runs_csv,
)
from sbnd.cafclasses.slice import CAFSlice


class _FakeNU:
    def __init__(self, data: pd.DataFrame, key_depth: int = 2):
        self.data = data
        self.key_depth = key_depth

    def copy(self, deep=True, duplicate_ok=False):
        out = _FakeNU(self.data.copy(deep=deep), key_depth=self.key_depth)
        return out

    def get_key(self, key):
        if isinstance(key, list):
            return list(key)
        return [key]


def _mcnu_from_triples(rows: list[tuple[int, int, int]]) -> _FakeNU:
    """rows: (file_index, __ntuple, entry) with __ntuple = file_index * step + local."""
    tuples = [(nt, ent) for _, nt, ent in rows]
    index = pd.MultiIndex.from_tuples(tuples, names=["__ntuple", "entry"])
    return _FakeNU(pd.DataFrame({"x": np.arange(len(rows))}, index=index))


def _hdr_stub(mcnu: _FakeNU) -> _FakeNU:
    return _FakeNU(pd.DataFrame(index=mcnu.data.index[:0]))


def test_file_index_from_ntuple():
    assert 1005 // NTUPLE_FILE_STEP == 1
    assert 0 // NTUPLE_FILE_STEP == 0


def test_prepare_mcnu_event_index_dedupes_triples():
    # Same __ntuple/entry would duplicate; different entry under same ntuple kept
    nom = _mcnu_from_triples([(0, 0, 0), (0, 0, 1), (1, 1000, 0)])
    indexed = prepare_mcnu_event_index(nom, _hdr_stub(nom))
    assert list(indexed.data.columns) == EVENT_CSV_COLUMNS
    assert len(indexed.data) == 3


def test_pairwise_common_events_synthetic():
    nom = _mcnu_from_triples([(0, 0, 0), (0, 0, 1), (1, 1000, 0)])
    var = _mcnu_from_triples([(0, 0, 0), (1, 1000, 0)])
    events, n_nom, n_var, n_common = pairwise_common_events(
        nom, _hdr_stub(nom), var, _hdr_stub(var)
    )
    assert list(events.columns) == EVENT_CSV_COLUMNS
    assert n_nom == 3
    assert n_var == 2
    assert n_common == 2
    assert len(events) == 2


def test_build_variation_scaling_entry():
    entry = build_variation_scaling_entry(
        var="wiremodxtheta",
        n_nominal=100,
        n_variation=80,
        n_common=50,
        pot_nominal_full=1e20,
        pot_det_full=9e19,
        runs_csv="det_var/wiremod/v8/wiremodxtheta_runs.csv",
    )
    assert entry["POT_NOMINAL_FILTERED"] == 0.5e20
    assert entry["POT_DET_SCALED"] == 50 / 80 * 9e19


def test_subsample_file_map_aux_div_only():
    fm = {
        "MC_SLIM_FNAMES": [f"slim{i}" for i in range(100)],
        "MC_NOMINAL_FNAMES": [f"nom{i}" for i in range(50)],
        "MC_LOWE_FNAMES": [f"lowe{i}" for i in range(40)],
        "DATA_OFFBEAM_FNAMES": [f"off{i}" for i in range(30)],
        "OFFBEAM_FNAMES": [f"mcoff{i}" for i in range(20)],
    }
    subsample_file_map(fm, aux_div=10)
    assert len(fm["MC_SLIM_FNAMES"]) == 100
    assert len(fm["MC_NOMINAL_FNAMES"]) == 50
    assert len(fm["OFFBEAM_FNAMES"]) == 2
    assert len(fm["MC_LOWE_FNAMES"]) == 4
    assert len(fm["DATA_OFFBEAM_FNAMES"]) == 3


def test_subsample_file_map_small_aux_not_double_sliced():
    fm = {
        "MC_NOMINAL_FNAMES": [f"nom{i}" for i in range(50)],
        "OFFBEAM_FNAMES": [f"mcoff{i}" for i in range(30)],
        "MC_LOWE_FNAMES": [f"lowe{i}" for i in range(30)],
        "DATA_OFFBEAM_FNAMES": [f"dataoff{i}" for i in range(30)],
    }
    subsample_file_map(fm, sample_div=5, aux_div=10)
    assert len(fm["MC_NOMINAL_FNAMES"]) == 10
    assert len(fm["OFFBEAM_FNAMES"]) == 3
    assert len(fm["MC_LOWE_FNAMES"]) == 3
    assert len(fm["DATA_OFFBEAM_FNAMES"]) == 3


def test_det_pot_eff_length_ratio_only():
    """Det scaling uses full sample POT times n_after/n_before, not event_ratio."""
    sample_pot = 1.5e5
    n_before, n_after = 1000, 287
    pot_eff = sample_pot * (n_after / n_before)
    assert abs(pot_eff - sample_pot * 0.287) < 1e-6
    assert abs(pot_eff - sample_pot * 0.698) > 1e3


def _pad_col(name: str, depth: int = 9) -> tuple:
    parts = name.split(".")
    return tuple(parts + [""] * (depth - len(parts)))


def test_filter_slice_to_events_triple_key():
    events = pd.DataFrame(
        [(0, 0, 0), (1, 1005, 2)],
        columns=EVENT_CSV_COLUMNS,
    )
    idx = pd.MultiIndex.from_tuples(
        [(0, 0, 0), (0, 1, 1), (1005, 2, 0), (2000, 3, 0)],
        names=["__ntuple", "entry", "rec.slc..index"],
    )
    depth = 9
    df = pd.DataFrame(
        {
            _pad_col("run", depth): [1, 1, 2, 2],
            _pad_col("subrun", depth): [0, 0, 0, 0],
            _pad_col("evt", depth): [10, 10, 20, 20],
            _pad_col("truth.E", depth): [0.5, 0.5, 2.0, 2.0],
        },
        index=idx,
    )
    slc = CAFSlice(df, pot=1.0)
    slc.key_depth = depth
    filtered = filter_slice_to_events(slc, events)
    # (0,0,0) and (1005,2,*) file_index=1; (0,1,1) same RSE as row0 but wrong entry
    assert len(filtered.data) == 2
    kept = set(filtered.data.index)
    assert (0, 0, 0) in kept
    assert (1005, 2, 0) in kept


def test_filter_slice_to_events_drops_wrong_entry_despite_same_rse():
    events = pd.DataFrame([(0, 0, 0)], columns=EVENT_CSV_COLUMNS)
    depth = 9
    idx = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1)],
        names=["__ntuple", "entry"],
    )
    df = pd.DataFrame(
        {
            _pad_col("run", depth): [1, 1],
            _pad_col("subrun", depth): [0, 0],
            _pad_col("evt", depth): [10, 10],
            _pad_col("truth.E", depth): [0.5, 0.5],
        },
        index=idx,
    )
    slc = CAFSlice(df, pot=1.0)
    slc.key_depth = depth
    filtered = filter_slice_to_events(slc, events)
    assert len(filtered.data) == 1
    assert filtered.data.index[0] == (0, 0)


def test_slice_count_parity_optional():
    if os.environ.get("DET_EVENT_LISTS_INTEGRATION") != "1":
        return
    cfg = build_config(small=True, ncpu=1)
    file_map = build_file_map(cfg)
    var = "wiremodxtheta"
    if var not in file_map["DET_VARS"]:
        return
    var_idx = file_map["DET_VARS"].index(var)
    flist = file_map["DET_FNAMES"][var_idx]
    if not flist or not file_map["MC_NOMINAL_FNAMES"]:
        return

    from sbnd.cafclasses.nu import NU
    from sbnd.cafclasses.parent import CAF

    nom_mcnu = NU.load(file_map["MC_NOMINAL_FNAMES"], key="mcnu*", ncpu=1, show_progress=False)
    nom_hdr = CAF.load(file_map["MC_NOMINAL_FNAMES"], key="hdr*", ncpu=1, show_progress=False)
    var_mcnu = NU.load(flist, key="mcnu*", ncpu=1, show_progress=False)
    var_hdr = CAF.load(flist, key="hdr*", ncpu=1, show_progress=False)
    events, _, _, _ = pairwise_common_events(nom_mcnu, nom_hdr, var_mcnu, var_hdr)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = build_config(small=True, ncpu=1, data_dir=tmp)
        write_runs_csv(events, runs_csv_path(cfg, var))
        payload = {
            "version": cfg.version,
            "POT_NOMINAL_FULL": 1.0,
            "variations": {
                var: build_variation_scaling_entry(
                    var=var,
                    n_nominal=10,
                    n_variation=10,
                    n_common=len(events),
                    pot_nominal_full=1.0,
                    pot_det_full=1.0,
                    runs_csv=str(runs_csv_path(cfg, var).name),
                ),
            },
        }
        write_pot_scaling_json(cfg, payload)

        slc_nom = CAFSlice.load(
            file_map["MC_NOMINAL_FNAMES"][:1], key="evt_pand*", ncpu=1, show_progress=False
        )
        slc_var = CAFSlice.load(flist[:1], key="evt_pand*", ncpu=1, show_progress=False)
        slc_nom = filter_slice_to_events(slc_nom, events)
        slc_var = filter_slice_to_events(slc_var, events)

        cols = slc_nom.get_key(["run", "subrun", "evt"])
        unique_runs = slc_nom.data.loc[:, cols].drop_duplicates()
        bad = 0
        for _, row in unique_runs.iterrows():
            run, subrun, evt = row
            counts = []
            for s in (slc_nom, slc_var):
                m1 = (s.data[cols[0]] == run) & (s.data[cols[1]] == subrun) & (s.data[cols[2]] == evt)
                counts.append(len(s.data.loc[m1]))
            if max(counts) - min(counts) > 2:
                bad += 1
        assert bad == 0, f"{bad} run/subrun/evt groups exceed count tolerance"


if __name__ == "__main__":
    test_file_index_from_ntuple()
    test_prepare_mcnu_event_index_dedupes_triples()
    test_pairwise_common_events_synthetic()
    test_build_variation_scaling_entry()
    test_filter_slice_to_events_triple_key()
    test_filter_slice_to_events_drops_wrong_entry_despite_same_rse()
    test_slice_count_parity_optional()
    print("det event list tests OK")
