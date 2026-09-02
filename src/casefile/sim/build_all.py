"""Regenerate the entire synthetic world from scratch. Deterministic given the seed."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from casefile.sim.scenarios import build_ground_truth
from casefile.sim.corpus import build_corpus
from casefile.sim.warehouse import build as build_warehouse


def main(seed: int = 20260901):
    t0 = time.time()
    (ROOT / "data/warehouse").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/corpus").mkdir(parents=True, exist_ok=True)
    print("1/3 ground truth (exact Shapley over counterfactual re-simulation)...")
    gt = build_ground_truth(str(ROOT / "data/warehouse/ground_truth.json"), seed, workers=32)
    print(f"    {gt['n_simulations']} counterfactual worlds")
    print("2/3 unstructured corpus...")
    cs = build_corpus(str(ROOT / "data/corpus/documents.jsonl"), seed)
    print(f"    {cs['total']} documents, {100*cs['noise_ratio']:.0f}% unrelated to any incident")
    print("3/3 warehouse...")
    w = build_warehouse(str(ROOT / "data/warehouse/vantage.duckdb"),
                        str(ROOT / "data/corpus/documents.jsonl"), seed)
    for k, v in w["tables"].items(): print(f"    {k:34s} {v:>10,}")
    print(f"    ledger reconciliation delta {w['reconciliation_delta_pct']}%")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20260901)
