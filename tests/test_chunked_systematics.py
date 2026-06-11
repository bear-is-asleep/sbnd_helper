"""Parity: chunked Systematics accumulation vs one-shot process_systematics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))

from sbnd.stats.systematics import Systematics

RW_COLS = [("truth", "g4", "w1"), ("truth", "g4", "w2")]
RW_KEYS = [("g4", "w1"), ("g4", "w2")]


def _frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({c: rng.uniform(0.9, 1.1, n) for c in RW_COLS}, columns=pd.Index(RW_COLS))


def test_chunked_rw_parity() -> None:
    rng = np.random.default_rng(0)
    bins = np.linspace(0.0, 1.0, 6)
    n_sig, n_sel, n_bg = 20, 40, 20

    reco_sel = np.linspace(0.05, 0.95, n_sel)
    reco_bg = np.linspace(0.1, 0.9, n_bg)
    gen_sig = np.ones(n_sig)
    gen_sel = np.ones(n_sel)
    gen_bg = np.ones(n_bg)

    sig_all = _frame(n_sig, rng)
    sel_all = _frame(n_sel, rng)
    bg_all = _frame(n_bg, rng)

    ref = Systematics(
        "costheta",
        bins,
        reco_sel,
        reco_bg,
        gen_sig,
        gen_sel,
        gen_bg,
        keys=RW_KEYS,
        pattern=["g4"],
        stype="RW",
    )
    ref.process_systematics(sig_all, sel_all, bg_all)

    chk = Systematics.for_chunked_build(
        "costheta",
        bins,
        RW_KEYS,
        pattern=["g4"],
        stype="RW",
    )
    chunks = [
        (slice(0, 10), slice(0, 15), slice(0, 10)),
        (slice(10, 20), slice(15, 40), slice(10, 20)),
    ]
    for sig_sl, sel_sl, bg_sl in chunks:
        chk.accumulate_cv_chunk(
            reco_sel[sel_sl],
            reco_bg[bg_sl],
            gen_sel[sel_sl],
            gen_bg[bg_sl],
        )
        chk.process_systematics_chunk(sig_all[sig_sl], sel_all[sel_sl], bg_all[bg_sl])
    chk.finalize_chunked_build()

    for key in ref.systematics:
        rsel = np.array(ref.systematics[key]["sel"])
        csel = np.array(chk.systematics[key]["sel"])
        assert np.allclose(rsel, csel, rtol=0, atol=1e-9), (
            f"sel mismatch for {key}: max diff {np.max(np.abs(rsel - csel))}"
        )

    assert np.allclose(ref.sel, chk.sel, rtol=0, atol=1e-9), (
        f"CV sel mismatch: max diff {np.max(np.abs(ref.sel - chk.sel))}"
    )


def test_chunked_det_parity() -> None:
    bins = np.linspace(0.0, 1.0, 6)
    reco_sel = np.linspace(0.05, 0.95, 40)
    reco_bg = np.linspace(0.1, 0.9, 30)
    gen_sel = np.ones(40)
    gen_bg = np.ones(30)

    ref = Systematics(
        "costheta",
        bins,
        reco_sel,
        reco_bg,
        gen_sel,
        gen_sel,
        gen_bg,
        keys=["detA"],
        stype="Det",
        pattern=None,
    )
    ref.process_det_systematics(
        [reco_sel],
        [reco_bg],
        [gen_sel],
        [gen_bg],
        sys_names=["detA"],
    )

    chk = Systematics(
        "costheta",
        bins,
        reco_sel[:20],
        reco_bg[:15],
        gen_sel[:20],
        gen_sel[:20],
        gen_bg[:15],
        keys=["detA"],
        stype="Det",
        pattern=None,
    )
    chk.process_det_systematics(
        [reco_sel[:20]],
        [reco_bg[:15]],
        [gen_sel[:20]],
        [gen_bg[:15]],
        sys_names=["detA"],
        accumulate=False,
    )
    chk.process_det_systematics(
        [reco_sel[20:]],
        [reco_bg[15:]],
        [gen_sel[20:]],
        [gen_bg[15:]],
        sys_names=["detA"],
        accumulate=True,
    )

    assert np.allclose(ref.systematics["detA"]["sel"][0], chk.systematics["detA"]["sel"][0], rtol=0, atol=1e-9)


if __name__ == "__main__":
    test_chunked_rw_parity()
    test_chunked_det_parity()
    print("chunked parity OK")
