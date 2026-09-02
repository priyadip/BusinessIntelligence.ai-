"""
Path resolution, so the prototype runs from either the development tree or the packaged
deliverable folder without editing anything.

Development layout          Deliverable layout
  casefile/contracts/...      config/kpi_contract.yaml
  data/warehouse/...          data/vantage.duckdb
  out/                        results/
"""
from __future__ import annotations
import os
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = Path(os.environ.get("CASEFILE_ROOT", _HERE.parents[1]))


def _first(*cands: Path) -> Path:
    for c in cands:
        if c.exists(): return c
    return cands[0]


def contract() -> Path:
    return _first(ROOT / "casefile/contracts/kpi_contract.yaml",
                  ROOT / "config/kpi_contract.yaml",
                  ROOT.parent / "config/kpi_contract.yaml",
                  _HERE.parent / "contracts/kpi_contract.yaml")


def routing() -> Path:
    return _first(_HERE.parent / "llm/routing.yaml",
                  ROOT / "config/routing.yaml",
                  ROOT.parent / "config/routing.yaml")


def warehouse() -> Path:
    return _first(ROOT / "data/warehouse/vantage.duckdb",
                  ROOT / "data/vantage.duckdb",
                  ROOT.parent / "data/vantage.duckdb")


def ground_truth() -> Path:
    return _first(ROOT / "data/warehouse/ground_truth.json",
                  ROOT / "data/ground_truth.json",
                  ROOT.parent / "data/ground_truth.json")


def corpus() -> Path:
    return _first(ROOT / "data/corpus/documents.jsonl",
                  ROOT / "data/documents.jsonl",
                  ROOT.parent / "data/documents.jsonl")


def out() -> Path:
    # the packaged deliverable keeps artifacts in ../results; the development tree uses ./out.
    # results/ wins when present so a stray empty out/ cannot shadow the shipped artifacts.
    for c in (ROOT.parent / "results", ROOT / "results", ROOT / "out"):
        if c.exists(): return c
    d = ROOT / "out"; d.mkdir(parents=True, exist_ok=True); return d


def likelihood_table() -> Path:
    return _first(ROOT / "data/warehouse/likelihood_table.json",
                  ROOT / "data/likelihood_table.json",
                  ROOT.parent / "data/likelihood_table.json")
