#!/usr/bin/env python3
"""Calibrate evidence likelihood ratios from simulated incidents, with held-out evaluation."""
from __future__ import annotations
import json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from casefile.sim.model import World
from casefile.sim.random_incident import draw
from casefile.engine.likelihood import ALL_HYPS
from casefile.paths import ROOT as PROJ_ROOT

TRAIN_SEEDS = range(30000, 30500)      # calibration only
TEST_SEEDS = range(9000, 9300)         # evaluation only, never seen here
assert not (set(TRAIN_SEEDS) & set(TEST_SEEDS)), "train and test seeds must be disjoint"

DRIVER_COLS = {"ontime": "ontime_pct", "checkout": "checkout_error_rate",
               "price": "price_index", "promo": "promo_depth",
               "stockout": "stockout_rate", "competitor": "competitor_gap",
               "marketing": "marketing_mult"}
# "promo" includes txt_price: a shopper cannot tell a price rise from an ended discount,
# so both produce the same complaint theme and the calibration must reflect that.
DRIVER_TESTS = {
    "ontime":     ["ontime_drop", "carrier_shift", "txt_delivery_late"],
    "checkout":   ["checkout_errors", "txt_payment_fail", "doc_release_checkout"],
    "price":      ["price_move", "txt_price", "doc_release_price"],
    "promo":      ["promo_move", "doc_release_promo", "txt_price"],
    "stockout":   ["stockout_rise", "txt_stock"],
    "competitor": ["txt_competitor"],
    "marketing":  ["mix_shift"],
}
FIRE_THRESHOLD = 0.15


def _observe(seed: int):
    """Return (fired test ids, true hypothesis ids) for one simulated world."""
    inc = draw(seed)
    try:
        panel = World(inc["interventions"], seed=seed).simulate()
    except Exception:
        return None
    panel["d"] = pd.to_datetime(panel["d"]).dt.date
    lo, hi = inc["window"]
    from datetime import timedelta
    pre_hi = lo - timedelta(days=3); pre_lo = pre_hi - timedelta(days=21)
    pre = panel[(panel.d >= pre_lo) & (panel.d <= pre_hi)]
    post = panel[(panel.d >= lo) & (panel.d <= hi)]
    if pre.empty or post.empty: return None
    fired = set()
    for k, col in DRIVER_COLS.items():
        a, b = pre[col].mean(), post[col].mean()
        if not np.isfinite(a) or a == 0: continue
        strength = float(np.clip(abs((b - a) / abs(a)) / 0.06, 0, 1))
        if strength >= FIRE_THRESHOLD:
            fired.update(DRIVER_TESTS[k])
    return sorted(fired), sorted({t["hypothesis"] for t in inc["truth"]})


def _retemper(p_top: float, T: float) -> float:
    """Re-apply a temperature to a top-class probability under a softmax over K classes."""
    K = len(ALL_HYPS)
    p_top = float(np.clip(p_top, 1e-9, 1 - 1e-9))
    logit = np.log(p_top / (1 - p_top)) + np.log(K - 1)
    z = logit / T
    return float(1.0 / (1.0 + (K - 1) * np.exp(-z)))


def _posterior_point(seed: int):
    """(top posterior, did it name the dominant true cause) for one calibration incident."""
    try:
        sys.path.insert(0, str(ROOT / "eval"))
        from run_batch import one
        r = one(seed)
    except Exception:
        return None
    if not r or not r.get("cf_named"): return None
    return float(r["cf_conf"]), bool(r["cf_hit_dominant"])


def main(workers: int = 48, out: Path | None = None):
    seeds = list(TRAIN_SEEDS)
    with Pool(workers) as p:
        obs = [o for o in p.map(_observe, seeds) if o]
    tests = sorted({t for f, _ in obs for t in f} | {t for v in DRIVER_TESTS.values() for t in v})

    fire_h = defaultdict(int); n_h = defaultdict(int)
    fire_nh = defaultdict(int); n_nh = defaultdict(int)
    for fired, truth in obs:
        fs = set(fired)
        for h in ALL_HYPS:
            if h == "H_NULL": continue
            true_here = h in truth
            for t in tests:
                if true_here:
                    n_h[(t, h)] += 1; fire_h[(t, h)] += 1 if t in fs else 0
                else:
                    n_nh[(t, h)] += 1; fire_nh[(t, h)] += 1 if t in fs else 0

    lr, support = {}, {}
    for t in tests:
        row, srow = {}, {}
        for h in ALL_HYPS:
            if h == "H_NULL": continue
            if n_h[(t, h)] < 8:        # too few positive examples to estimate anything
                continue
            p_h = (fire_h[(t, h)] + 0.5) / (n_h[(t, h)] + 1.0)      # Jeffreys
            p_nh = (fire_nh[(t, h)] + 0.5) / (n_nh[(t, h)] + 1.0)
            raw = p_h / max(p_nh, 1e-9)
            ratio = float(np.clip(raw, 0.05, 50.0))
            if abs(ratio - 1.0) < 0.08:      # indistinguishable from uninformative
                continue
            row[h] = round(ratio, 4)
            srow[h] = {"n_positive": n_h[(t, h)], "fired_when_true": fire_h[(t, h)],
                       "p_fire_given_h": round(p_h, 4), "p_fire_given_not_h": round(p_nh, 4),
                       "raw_ratio": round(float(raw), 2), "clipped": bool(raw > 50.0)}
        if row:
            lr[t] = row; support[t] = srow

    payload = {"lr": lr, "support": support, "n_incidents": len(obs),
               "train_seeds": [min(seeds), max(seeds)],
               "test_seeds_held_out": [min(TEST_SEEDS), max(TEST_SEEDS)],
               "method": "Jeffreys-smoothed P(evidence|hypothesis) / P(evidence|not hypothesis) "
                         "over simulated incidents; ratios clipped to [0.05, 50]",
               "fire_threshold": FIRE_THRESHOLD,
               "min_positive_examples": 8,
               "clip_note": "In this synthetic world the mapping from cause to observable "
                            "evidence is close to deterministic, so many raw ratios are very "
                            "large and saturate the clip. The raw estimate is retained in "
                            "`support` for audit. The clip is kept deliberately: real evidence "
                            "is far noisier, and no single test should be able to manufacture "
                            "certainty on its own.",
               "correlated_tests_note": "Tests driven by the same observable (for example "
                                        "ontime_drop, carrier_shift and txt_delivery_late) fire "
                                        "together and therefore receive identical ratios here. "
                                        "Double counting is prevented at inference time by "
                                        "source-group collapsing in verdict.build_posterior.",
               "dropped_uninformative": "ratios within 8% of 1.0 are omitted so uninformative "
                                        "evidence contributes exactly nothing"}
    # ---- fit the temperature on the SAME calibration seeds, then lock it ----
    out = out or (PROJ_ROOT / "data/warehouse/likelihood_table.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["temperature"] = 1.0
    out.write_text(json.dumps(payload, indent=1))          # write untempered first so the
                                                           # fitting pass loads these ratios
    with Pool(workers) as p2:
        pts = [r for r in p2.map(_posterior_point, seeds) if r]
    if pts:
        conf = np.array([c for c, _ in pts]); hit = np.array([float(h) for _, h in pts])
        best_T, best_brier = 1.0, float(np.mean((conf - hit) ** 2))
        for T in np.arange(1.0, 12.05, 0.25):
            c2 = np.array([_retemper(c, T) for c in conf])
            b = float(np.mean((c2 - hit) ** 2))
            if b < best_brier: best_T, best_brier = float(T), b
        payload["temperature"] = round(best_T, 3)
        payload["temperature_fit"] = {
            "n_calibration_points": len(pts),
            "brier_before": round(float(np.mean((conf - hit) ** 2)), 4),
            "brier_after": round(best_brier, 4),
            "method": "single scalar temperature on the summed log-likelihood-ratio, grid "
                      "searched to minimise Brier score on the calibration seeds only; "
                      "locked before any evaluation seed is touched. Ranking is unaffected."}
        print(f"\ntemperature {best_T:.2f}  Brier {np.mean((conf-hit)**2):.4f} -> {best_brier:.4f} "
              f"on {len(pts)} calibration points")
    out.write_text(json.dumps(payload, indent=1))
    print(f"calibrated on {len(obs)} incidents from seeds {min(seeds)}..{max(seeds)}")
    print(f"held-out evaluation seeds {min(TEST_SEEDS)}..{max(TEST_SEEDS)} were never used here")
    print(f"{len(lr)} evidence tests carry informative ratios -> {out}\n")
    for t in sorted(lr):
        top = sorted(lr[t].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {t:22s} " + "  ".join(f"{h.replace('H_',''):18s} LR={v:6.2f}" for h, v in top))
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args(); main(a.workers)
