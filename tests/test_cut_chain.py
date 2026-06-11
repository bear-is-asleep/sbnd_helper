"""cuts_for_mode and cut_chain_for_output."""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[2]))

from detsys_config import (
    CUTS_BEFORE_MUON,
    DETSYS_CUTS_ALL,
    DETSYS_CUTS_MUON,
    cut_chain_for_output,
    cuts_for_mode,
)
from naming import PAND_CUTS_CONT


def test_cuts_for_mode() -> None:
    assert cuts_for_mode("all") == DETSYS_CUTS_ALL
    assert cuts_for_mode("muon") == DETSYS_CUTS_MUON


def test_muon_mode_prepends_upstream() -> None:
    chain = cut_chain_for_output("muon", DETSYS_CUTS_MUON)
    assert chain == list(CUTS_BEFORE_MUON) + ["muon"]


def test_muon_mode_cont() -> None:
    chain = cut_chain_for_output("cont", DETSYS_CUTS_MUON)
    assert chain == list(CUTS_BEFORE_MUON) + ["muon", "cont_full", "cont"]


def test_all_mode_cont() -> None:
    chain = cut_chain_for_output("cont", DETSYS_CUTS_ALL)
    assert chain == list(PAND_CUTS_CONT)


if __name__ == "__main__":
    test_cuts_for_mode()
    test_muon_mode_prepends_upstream()
    test_muon_mode_cont()
    test_all_mode_cont()
    print("all tests passed")
