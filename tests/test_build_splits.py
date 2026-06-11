"""Tests for three-way full build file splits and selective save."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

_here = Path(__file__).resolve()
_repo = _here.parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "scripts"))

from detsys_config import (
    _FULL_COSMIC_DATA_OFFBEAM_DIV,
    _FULL_COSMIC_LOWE_DIV,
    _FULL_COSMIC_NOMINAL_DIV,
    _FULL_COSMIC_OFFBEAM_MC_DIV,
    _DEFAULT_SLIM_DIV,
    _FULL_PARTIAL_AUX_DIV,
    _FULL_TEST_DIV,
    _SINGLE_FILE_DIV,
    _apply_group_div,
    apply_build_mode_splits,
    build_config,
)
from sbnd.stats.systematics import Systematics


def _fake_file_map(n: int = 100) -> dict:
    files = [f"/fake/f{i}.df" for i in range(n)]
    return {
        "MC_SLIM_FNAMES": list(files),
        "MC_NOMINAL_FNAMES": list(files),
        "OFFBEAM_FNAMES": list(files),
        "DATA_OFFBEAM_FNAMES": list(files),
        "DATA_FNAMES": list(files),
        "MC_LOWE_FNAMES": list(files),
        "DET_VARS": ["pmtgain", "nosce"],
        "DET_FNAMES": [list(files), list(files)],
        "CALO_VARS": [],
        "CALO_SUFFIXES": [],
    }


def test_apply_group_div_zero_is_single_file() -> None:
    files = [f"a{i}" for i in range(20)]
    assert _apply_group_div(files, _SINGLE_FILE_DIV) == ["a0"]


def test_apply_group_div_one_is_full() -> None:
    files = [f"a{i}" for i in range(20)]
    assert _apply_group_div(files, 1) == files


def test_apply_group_div_ten() -> None:
    files = [f"a{i}" for i in range(100)]
    assert len(_apply_group_div(files, 10)) == 10


def test_full_slim_splits() -> None:
    fm = _fake_file_map()
    apply_build_mode_splits(fm, "full_slim")
    assert len(fm["MC_SLIM_FNAMES"]) == 100
    assert fm["MC_NOMINAL_FNAMES"] == []
    assert fm["OFFBEAM_FNAMES"] == []
    assert all(fl == [] for fl in fm["DET_FNAMES"])
    assert len(fm["MC_LOWE_FNAMES"]) == 100 // _FULL_PARTIAL_AUX_DIV
    assert len(fm["DATA_OFFBEAM_FNAMES"]) == 100 // _FULL_PARTIAL_AUX_DIV


def test_full_det_splits() -> None:
    fm = _fake_file_map()
    apply_build_mode_splits(fm, "full_det")
    assert fm["MC_SLIM_FNAMES"] == []
    assert len(fm["MC_NOMINAL_FNAMES"]) == 100
    assert len(fm["DET_FNAMES"][0]) == 100
    assert fm["OFFBEAM_FNAMES"] == []
    assert len(fm["MC_LOWE_FNAMES"]) == 100 // _FULL_PARTIAL_AUX_DIV


def test_full_cosmic_splits() -> None:
    fm = _fake_file_map()
    apply_build_mode_splits(fm, "full_cosmic")
    assert fm["MC_SLIM_FNAMES"] == []
    assert len(fm["MC_NOMINAL_FNAMES"]) == 100 // _FULL_COSMIC_NOMINAL_DIV
    assert all(fl == [] for fl in fm["DET_FNAMES"])
    assert len(fm["OFFBEAM_FNAMES"]) == 100 // _FULL_COSMIC_OFFBEAM_MC_DIV
    assert len(fm["MC_LOWE_FNAMES"]) == 100 // _FULL_COSMIC_LOWE_DIV
    assert len(fm["DATA_OFFBEAM_FNAMES"]) == 100 // _FULL_COSMIC_DATA_OFFBEAM_DIV


def test_full_slim_test_at_one_percent() -> None:
    fm = _fake_file_map(1000)
    apply_build_mode_splits(fm, "full_slim_test")
    assert len(fm["MC_SLIM_FNAMES"]) == 1000 // _FULL_TEST_DIV
    assert len(fm["MC_LOWE_FNAMES"]) == 1000 // _FULL_TEST_DIV
    assert len(fm["DATA_OFFBEAM_FNAMES"]) == 1000 // _FULL_TEST_DIV


def test_full_det_test_at_one_percent() -> None:
    fm = _fake_file_map(1000)
    apply_build_mode_splits(fm, "full_det_test")
    assert len(fm["MC_NOMINAL_FNAMES"]) == 1000 // _FULL_TEST_DIV
    assert len(fm["DET_FNAMES"][0]) == 1000 // _FULL_TEST_DIV
    assert len(fm["MC_LOWE_FNAMES"]) == 1000 // _FULL_TEST_DIV


def test_full_cosmic_test_at_one_percent() -> None:
    fm = _fake_file_map(1000)
    apply_build_mode_splits(fm, "full_cosmic_test")
    assert len(fm["MC_NOMINAL_FNAMES"]) == 1000 // _FULL_TEST_DIV
    assert len(fm["OFFBEAM_FNAMES"]) == 1000 // _FULL_TEST_DIV
    assert len(fm["MC_LOWE_FNAMES"]) == 1000 // _FULL_TEST_DIV
    assert len(fm["DATA_OFFBEAM_FNAMES"]) == 1000 // _FULL_TEST_DIV


def test_full_test_mode_properties() -> None:
    cfg = build_config(build_mode="full_cosmic_test")
    assert cfg.full_cosmic is True
    assert cfg.full_test is True
    assert cfg.save_build_mode == "full_cosmic"
    assert cfg.runs_cosmic is True
    assert cfg.runs_det is False
    assert cfg.runs_slim is False


def test_resolve_full_cosmic_test() -> None:
    import argparse

    from build_detsys_universes import _resolve_build_mode

    args = argparse.Namespace(
        tiny=False,
        small=False,
        full_slim=False,
        full_slim_test=False,
        full_det=False,
        full_det_test=False,
        full_cosmic=False,
        full_cosmic_test=True,
    )
    assert _resolve_build_mode(args) == "full_cosmic_test"


def test_default_slim_div() -> None:
    fm = _fake_file_map(200)
    apply_build_mode_splits(fm, "default")
    assert len(fm["MC_SLIM_FNAMES"]) == 200 // _DEFAULT_SLIM_DIV


def test_resolve_build_mode_exclusive() -> None:
    import argparse

    from build_detsys_universes import _resolve_build_mode

    args = argparse.Namespace(
        tiny=False,
        small=False,
        full_slim=True,
        full_slim_test=False,
        full_det=True,
        full_det_test=False,
        full_cosmic=False,
        full_cosmic_test=False,
    )
    try:
        _resolve_build_mode(args)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_selective_save_merge() -> None:
    bins = np.linspace(0.0, 1.0, 4)
    sys_obj = Systematics.for_chunked_build("costheta", bins, [], pattern=None, stype="RW")
    sys_obj.sel = np.array([1.0, 2.0, 3.0])
    sys_obj.sel_background = np.array([0.5, 0.5, 0.5])
    sys_obj.systematics = {
        "pmtgain": {
            "cols": None,
            "col_names": [],
            "type": "Det",
            "name": "pmtgain",
            "label": "pmtgain",
            "description": "",
            "variation": "Det",
            "order": None,
            "sel": [np.array([0.4, 0.5, 0.6])],
        },
        "xsec": {
            "cols": None,
            "col_names": [],
            "type": "RW",
            "name": "xsec",
            "label": "xsec",
            "description": "",
            "variation": "RW",
            "order": None,
            "sel": [np.array([0.1, 0.2, 0.3])],
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        cut_dir = Path(tmp)
        sys_obj.save(
            str(cut_dir),
            metadata_dir="metadata_detsys",
            include_sys_keys=["pmtgain"],
        )
        meta = cut_dir / "metadata_detsys" / "sys_dict.json"
        saved = json.loads(meta.read_text())
        assert "pmtgain" in saved
        assert "xsec" not in saved

        slim = Systematics.for_chunked_build("costheta", bins, [], pattern=None, stype="RW")
        slim.systematics = {"xsec": sys_obj.systematics["xsec"]}
        slim.save(
            str(cut_dir),
            metadata_dir="metadata_detsys",
            include_sys_keys=["xsec"],
            skip_metadata=True,
            merge_sys_dict=True,
        )
        merged = json.loads(meta.read_text())
        assert "pmtgain" in merged
        assert "xsec" in merged


if __name__ == "__main__":
    for _name in sorted(n for n in dir() if n.startswith("test_")):
        globals()[_name]()
        print(f"ok {_name}")
    print("all passed")
