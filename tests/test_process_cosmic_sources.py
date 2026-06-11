"""Cosmic CV uses data offbeam; variation uses MC offbeam (notebook parity)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

_here = Path(__file__).resolve()
_repo = _here.parents[2]
_cafpyana_wd = _repo.parents[1]
for p in (str(_repo), str(_cafpyana_wd)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.build_detsys_universes import _process_cosmic


def test_process_cosmic_cv_vs_variation_backgrounds() -> None:
    nom_reco = np.array([1.0, 2.0])
    data_reco = np.array([10.0, 20.0])
    mc_reco = np.array([100.0, 200.0])
    nom_gen = np.array([1.0, 1.0])
    data_gen = np.array([2.0, 2.0])
    mc_gen = np.array([3.0, 3.0])

    def fake_get_values(slc, key):
        if key == "mu.pfp.trk.costheta":
            if slc is nominal:
                return nom_reco
            if slc is data_ob:
                return data_reco
            if slc is mc_ob:
                return mc_reco
        if key == "genweight":
            if slc is nominal:
                return nom_gen
            if slc is data_ob:
                return data_gen
            if slc is mc_ob:
                return mc_gen
        return None

    nominal = object()
    data_ob = object()
    mc_ob = object()

    captured: dict = {}

    class FakeCosmic:
        def __init__(self, _name, _bins, _reco_sel, reco_sel_background, _gw_sig, _gw_sel, genweights_sel_background, **_kw):
            captured["init_reco_bg"] = reco_sel_background
            captured["init_gen_bg"] = genweights_sel_background

        def process_det_systematics(
            self, _reco_sel_vars, reco_sel_background_vars, _gw_sel_vars, genweights_sel_background_vars, **_kw
        ):
            captured["var_reco_bg"] = reco_sel_background_vars[0]
            captured["var_gen_bg"] = genweights_sel_background_vars[0]

        def compute_covariances(self, **kw):
            pass

    system = MagicMock()
    system._data = None
    system._genweights_data = None
    cfg = MagicMock()
    cfg.cuts = ["precut"]

    with (
        patch("scripts.build_detsys_universes._get_values", side_effect=fake_get_values),
        patch("scripts.build_detsys_universes.Systematics", FakeCosmic),
        patch.object(system, "combine"),
    ):
        _process_cosmic(
            system,
            "costheta",
            np.linspace(0, 1, 3),
            "mu.pfp.trk.costheta",
            None,
            nominal,
            mc_ob,
            data_ob,
            "precut",
            cfg,
        )

    assert np.array_equal(captured["init_reco_bg"], data_reco)
    assert np.array_equal(captured["init_gen_bg"], data_gen)
    assert np.array_equal(captured["var_reco_bg"], mc_reco)
    assert np.array_equal(captured["var_gen_bg"], mc_gen)
