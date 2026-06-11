"""Tests for det xsec covariance scaling from event fractional cov."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))

from sbnd.stats.systematics import (
    Systematics,
    scale_xsec_cov_from_event_fraccov,
    warn_if_event_xsec_fracunc_mismatch,
    xsec_cv_from_event_scale,
)
from sbnd.stats.stats import covariance_from_fraccov


def _make_systematics(variable_name: str = "costheta") -> Systematics:
    bins = np.linspace(-1.0, 1.0, 5)
    n = 12
    reco_sel = np.linspace(-0.9, 0.9, n)
    reco_bg = np.linspace(-0.8, 0.8, n)
    gen = np.ones(n)
    sys_obj = Systematics(
        variable_name,
        bins,
        reco_sel,
        reco_bg,
        gen,
        gen,
        gen,
    )
    sys_obj.xsec_unit = 2.0
    sys_obj.sigma_tilde = np.full(len(bins) - 1, 0.25)
    return sys_obj


def _add_det_key(sys_obj: Systematics, key: str = "pmtgain") -> None:
    n_bins = len(sys_obj.bins) - 1
    cv = sys_obj.sel + sys_obj.sel_background
    u1 = cv * 1.02
    u2 = cv * 0.98
    template = Systematics._get_dict_template()
    sys_obj.systematics[key] = {
        **template,
        "type": "pds",
        "name": key,
        "variation": "unisim",
        "rank": 1,
        "sel": [u1, u2],
        "sigma_tilde": [],
    }


def _add_rw_key(sys_obj: Systematics, key: str = "xsec") -> None:
    n_bins = len(sys_obj.bins) - 1
    cv = sys_obj.sel + sys_obj.sel_background
    template = Systematics._get_dict_template()
    sys_obj.systematics[key] = {
        **template,
        "type": "xsec",
        "name": key,
        "variation": "multisim",
        "rank": 1,
        "sel": [cv * 1.01, cv * 0.99],
        "sigma_tilde": [],
    }


def test_xsec_cv_from_event_scale() -> None:
    sel = np.array([10.0, 20.0])
    bg = np.array([1.0, 2.0])
    out = xsec_cv_from_event_scale(sel, bg, 3.0)
    assert np.allclose(out, np.array([33.0, 66.0]))


def test_det_empty_sigma_tilde_uses_scaled_xsec_cov() -> None:
    sys_obj = _make_systematics()
    _add_det_key(sys_obj)

    sys_obj.compute_covariances(keys=["pmtgain"], compute_xsec_cov=True)
    sd = sys_obj.systematics["pmtgain"]

    assert sd["xsec_cov"] is not None
    assert sd["xsec_fraccov"] is sd["event_fraccov"]
    assert np.allclose(sd["xsec_fracunc"], sd["event_fracunc"])
    expected_cv = xsec_cv_from_event_scale(sys_obj.sel, sys_obj.sel_background, sys_obj.xsec_unit)
    assert np.allclose(sd["xsec_cov"], covariance_from_fraccov(sd["event_fraccov"], expected_cv))
    assert warn_if_event_xsec_fracunc_mismatch(
        "pmtgain", "costheta", sd["event_fracunc"], sd["xsec_fracunc"]
    ) < 1e-4


def test_rw_empty_sigma_tilde_raises() -> None:
    sys_obj = _make_systematics()
    _add_rw_key(sys_obj)

    try:
        sys_obj.compute_covariances(keys=["xsec"], compute_xsec_cov=True)
    except ValueError as exc:
        assert "RW systematic" in str(exc)
    else:
        raise AssertionError("expected ValueError for RW key with empty sigma_tilde")


def test_scaled_fallback_only_on_xsec_variables() -> None:
    sys_obj = _make_systematics(variable_name="flashpe")
    _add_det_key(sys_obj)

    sys_obj.compute_covariances(keys=["pmtgain"], compute_xsec_cov=True)
    sd = sys_obj.systematics["pmtgain"]
    assert sd["xsec_cov"] is None


def test_scale_xsec_cov_from_event_fraccov_helper() -> None:
    fraccov = np.eye(2) * 0.01
    fracunc = np.array([0.1, 0.1])
    out = scale_xsec_cov_from_event_fraccov(
        fraccov,
        fraccov,
        np.eye(2),
        fracunc,
        fracunc * 10.0,
        np.array([10.0, 20.0]),
        np.array([0.0, 0.0]),
        2.0,
    )
    assert np.allclose(out["xsec_fracunc"], fracunc)
    assert np.allclose(out["xsec_cov"], covariance_from_fraccov(fraccov, np.array([20.0, 40.0])))


if __name__ == "__main__":
    test_xsec_cv_from_event_scale()
    test_det_empty_sigma_tilde_uses_scaled_xsec_cov()
    test_rw_empty_sigma_tilde_raises()
    test_scaled_fallback_only_on_xsec_variables()
    test_scale_xsec_cov_from_event_fraccov_helper()
    print("all tests passed")
