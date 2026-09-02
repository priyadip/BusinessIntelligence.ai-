"""Test suite. Run: python3 -m pytest tests -q"""
import json, sys, math
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd, pytest, yaml
ROOT = Path(__file__).resolve().parents[1]
for c in (ROOT, ROOT / "src"):
    if (c / "casefile").exists(): sys.path.insert(0, str(c)); break

from casefile.security.policy import PolicyEngine
from casefile.engine import contribution as C, verdict as VD, narrative as NR, likelihood as LK
from casefile.engine import baseline as B

from casefile.paths import contract as _contract
CONTRACT = yaml.safe_load(open(_contract()))


# ---------------------------------------------------------------- contract
def test_contract_is_complete():
    for k, v in CONTRACT["kpis"].items():
        for need in ("definition", "measure", "thresholds", "lineage", "access", "grain"):
            assert need in v, f"{k} missing {need}"
    assert 3 <= len(CONTRACT["kpis"]) <= 5
    assert 2 <= len(CONTRACT["sources"]) <= 3
    assert len({s["refresh_cadence_minutes"] for s in CONTRACT["sources"].values()}) > 1
    assert len({k["grain"] for k in CONTRACT["kpis"].values()}) > 1


def test_every_hypothesis_resolves_to_a_lever():
    for h in CONTRACT["hypothesis_library"]:
        assert h["lever"] in CONTRACT["levers"], h["id"]


# ---------------------------------------------------------------- decompositions
def test_lmdi_is_exact():
    pre = {"a": 1_000_000.0, "b": 0.0567, "c": 2400.0}
    post = {"a": 980_000.0, "b": 0.0505, "c": 2455.0}
    cs, tot, resid = C.lmdi_factors(pre, post)
    assert abs(resid) < 1e-6 * abs(tot)
    assert abs(sum(c.value for c in cs) - tot) < 1e-6 * abs(tot)


def test_ratio_decomposition_is_exact():
    segs = ["W", "N", "S", "E"]
    pre = pd.DataFrame({"k": segs, "numerator": [5200, 5900, 5800, 3100],
                        "denominator": [91000, 101000, 98000, 54000]})
    post = pd.DataFrame({"k": segs, "numerator": [4700, 5600, 5700, 2850],
                         "denominator": [90000, 100500, 98500, 53500]})
    cs, tot, resid = C.ratio_segments(pre, post, "k")
    assert abs(resid) < 1e-12
    exact = post.numerator.sum()/post.denominator.sum() - pre.numerator.sum()/pre.denominator.sum()
    assert abs(tot - exact) < 1e-12


# ---------------------------------------------------------------- security
def test_row_and_column_policy():
    pe = PolicyEngine(CONTRACT)
    ops, cd = pe.principal("ops_manager"), pe.principal("category_director")
    assert set(ops.regions) == {"WEST"}
    assert not pe.may_read(ops, "gross_margin_pct")
    assert pe.may_read(cd, "gross_margin_pct")


def test_render_fails_closed_on_derived_values():
    pe = PolicyEngine(CONTRACT); ops = pe.principal("ops_manager")
    with pytest.raises(PermissionError):
        pe.assert_renderable(ops, {"m": {"value": 0.31, "derived_from": ["gross_margin_pct"]}})


def test_pii_shield_redacts():
    pe = PolicyEngine(CONTRACT); ops = pe.principal("ops_manager")
    t, red = pe.redact("Customer Aarav Sharma (a1@mail.example, +91-9876543210): late.",
                       "support_tickets", ops)
    assert red and "mail.example" not in t and "9876543210" not in t


# ---------------------------------------------------------------- verdict
def _hyps():
    return [VD.Hypothesis("H_PRICE_RISE", "price", .2, lever="price_change",
                          control_available=False, rung="R2"),
            VD.Hypothesis("H_PROMO_WITHDRAWAL", "promo", .2, lever="promo_depth",
                          control_available=False, rung="R2"),
            VD.Hypothesis("H_NULL", "outside", .06)]


def test_non_diagnostic_evidence_is_ignored():
    h = _hyps()
    ev = [{"id": "flat", "lr": {x.id: 1.05 for x in h}, "source_group": "g1"}]
    meta = VD.build_posterior(h, ev, CONTRACT)
    assert len(meta["evidence_ignored_non_diagnostic"]) == 1


def test_alibi_screen_eliminates_out_of_window_causes():
    h = _hyps()
    VD.alibi_screen(h, {"H_PRICE_RISE": date(2026, 1, 1)}, (date(2026, 8, 3), date(2026, 8, 6)))
    assert not h[0].alibi_ok


def test_collinear_causes_abstain():
    h = _hyps()
    ev = [{"id": "a", "lr": {"H_PRICE_RISE": 6.0, "H_PROMO_WITHDRAWAL": 1.0, "H_NULL": 1.0}, "source_group": "g1"},
          {"id": "b", "lr": {"H_PRICE_RISE": 1.0, "H_PROMO_WITHDRAWAL": 5.6, "H_NULL": 1.0}, "source_group": "g2"}]
    VD.build_posterior(h, ev, CONTRACT)
    v = VD.decide("T", "net_revenue", h, CONTRACT, 250000.0)
    assert v.decision == "ABSTAIN" and v.abstain_type == "collinear_causes"


def test_h_null_always_present_and_can_win():
    h = [VD.Hypothesis("H_X", "x", .2, lever="price_change"), VD.Hypothesis("H_NULL", "outside", .8)]
    VD.build_posterior(h, [], CONTRACT)
    v = VD.decide("T", "net_revenue", h, CONTRACT, 1000.0)
    assert v.abstain_type == "out_of_library"


def test_stale_source_blocks_before_anything_else():
    v = VD.decide("T", "gross_margin_pct", _hyps(), CONTRACT, 1000.0, stale=True)
    assert v.decision == "ABSTAIN" and v.abstain_type == "stale_source"


def test_cheapest_decisive_test_wins_on_enbs():
    h = _hyps(); VD.build_posterior(h, [
        {"id": "a", "lr": {"H_PRICE_RISE": 6.0, "H_PROMO_WITHDRAWAL": 1.0, "H_NULL": 1.0}, "source_group": "g1"},
        {"id": "b", "lr": {"H_PRICE_RISE": 1.0, "H_PROMO_WITHDRAWAL": 5.6, "H_NULL": 1.0}, "source_group": "g2"}],
        CONTRACT)
    cheap = VD.CandidateTest("cheap", "cheap holdout", ["H_PROMO_WITHDRAWAL"], 96, 24, 10, 0, 4200,
                             .02, "weekly", "category_director", power=0.86)
    dear = VD.CandidateTest("dear", "expensive revert", ["H_PRICE_RISE"], 96, 24, 14, 340000, 5200,
                            .02, "poor", "category_director", power=0.84)
    ranked, evpi = VD.rank_tests(h, [dear, cheap], 259000.0)
    assert ranked[0].id == "cheap"
    assert evpi >= max(t.evsi for t in ranked) - 1e-6     # EVPI bounds every EVSI


# ---------------------------------------------------------------- narrative
def test_closure_check_catches_free_numerals():
    facts = {"delta": -7.98, "n": 22}
    assert NR.closure_check("down 7.98% over 22 days", facts)["passed"]
    bad = NR.closure_check("down 8.4% over 22 days", facts)
    assert not bad["passed"] and "8.4%" in bad["free_numerals"]


def test_closure_check_survives_nan():
    NR.closure_check("nothing here", {"x": float("nan")})


# ---------------------------------------------------------------- likelihood
def test_lr_scales_with_strength_and_is_neutral_where_uninformed():
    """On-target ratios rise with strength; hypotheses the table is silent about get exactly
    1.0. A CALIBRATED entry below 1.0 is not a bug: it is measured evidence AGAINST that
    hypothesis, which the hand-written prior table could not express."""
    t, _ = LK.load_table()
    full = LK.lr_vector("carrier_shift", 1.0, t)
    half = LK.lr_vector("carrier_shift", 0.3, t)
    assert full["H_CARRIER_DEGRADE"] > half["H_CARRIER_DEGRADE"] > 1.0
    row = t.get("carrier_shift", {})
    silent = [h for h in LK.ALL_HYPS if h not in row]
    assert silent, "expected at least one hypothesis the table is silent about"
    for h in silent:
        assert full[h] == 1.0, f"{h} is not in the table but did not get a neutral ratio"
    for h, v in row.items():
        assert 0.05 <= full[h] <= 50.0, f"{h} ratio {full[h]} escaped the clip"


def test_calibrated_table_is_locked_and_held_out():
    t, src = LK.load_table()
    import json as _j
    if not LK.CALIBRATED_PATH.exists():
        return
    d = _j.loads(LK.CALIBRATED_PATH.read_text())
    tr = set(range(d["train_seeds"][0], d["train_seeds"][1] + 1))
    te = set(range(d["test_seeds_held_out"][0], d["test_seeds_held_out"][1] + 1))
    assert not (tr & te), "calibration and evaluation seeds overlap"
    assert d.get("temperature", 1.0) >= 1.0
    assert "CALIBRATED" in src


def test_alibi_screen_never_eliminates_every_evidenced_hypothesis():
    """The screen must disable itself rather than hand the verdict to hypotheses that carry
    no evidence at all. A change-point off by two weeks once did exactly that."""
    h = [VD.Hypothesis("H_PRICE_RISE", "price", .3, lever="price_change"),
         VD.Hypothesis("H_PROMO_WITHDRAWAL", "promo", .3, lever="promo_depth"),
         VD.Hypothesis("H_STOCKOUT", "stock", .3, lever="inventory_replenish")]
    meta = VD.alibi_screen(
        h, {"H_PRICE_RISE": date(2026, 1, 1), "H_PROMO_WITHDRAWAL": date(2026, 1, 2)},
        (date(2026, 8, 3), date(2026, 8, 6)), onset_confidence=1.0,
        supported={"H_PRICE_RISE", "H_PROMO_WITHDRAWAL"})
    assert not meta["applied"] and meta["disabled_reason"]
    assert all(x.alibi_ok for x in h)


def test_alibi_penalty_is_soft_when_the_onset_is_uncertain():
    h = [VD.Hypothesis("H_PRICE_RISE", "price", .3, lever="price_change"),
         VD.Hypothesis("H_STOCKOUT", "stock", .3, lever="inventory_replenish")]
    VD.alibi_screen(h, {"H_PRICE_RISE": date(2026, 1, 1)},
                    (date(2026, 8, 3), date(2026, 8, 6)), onset_confidence=0.05,
                    supported={"H_STOCKOUT"})
    assert h[0].alibi_ok, "a weak change-point must not eliminate anything"
    assert h[0].alibi_penalty < 2.0


# ---------------------------------------------------------------- baseline
def test_sparse_history_refuses_rather_than_guessing():
    s = pd.Series(np.random.default_rng(0).normal(100, 5, 12),
                  index=pd.date_range("2026-08-01", periods=12))
    bl = B.fit("net_revenue", s, CONTRACT, test_start=pd.Timestamp("2026-08-09"))
    assert bl.method == "INSUFFICIENT_HISTORY"
    assert not bl.causal_claims_allowed


def test_an_uncoverable_declared_cycle_is_refused_not_guessed():
    """The failure this catches is silence, not error.

    A declared seasonal cycle the training block cannot cover is dropped by MSTL and the
    baseline still returns a number. On real retail data that number is confident and
    wrong through the seasonal peak. The engine must notice and abstain.
    """
    import numpy as np, pandas as pd
    from casefile.engine import baseline as B, verdict as VD
    rng = np.random.default_rng(3)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    s = pd.Series(1000 * (1 + 0.05 * np.sin(2 * np.pi * np.arange(n) / 7))
                  * rng.lognormal(0, 0.02, n), index=idx)
    C = {"kpis": {"net_revenue": {"min_history_days": 120,
                                  "seasonality": {"weekly": True, "yearly": True}}},
         "decision_policy": {**CONTRACT["decision_policy"]}}

    bl = B.fit("net_revenue", s, C, test_start=idx[280])
    assert bl.method == "MSTL_PROJECTED", "the baseline still fits, which is the danger"
    assert 365 in bl.dropped_periods, "the uncoverable annual cycle was not recorded"
    assert not bl.seasonal_coverage_ok

    h = [VD.Hypothesis("H_A", "a", 0.9), VD.Hypothesis("H_NULL", "none", 0.1)]
    v = VD.decide("T", "net_revenue", h, C, 1e5,
                  incomplete_seasonal_cycle=not bl.seasonal_coverage_ok,
                  unmodelled_periods=bl.dropped_periods)
    assert v.decision == "ABSTAIN"
    assert v.abstain_type == "incomplete_seasonal_cycle"
    assert "730" in v.reason, "the refusal must say how much history it would need"


def test_a_covered_cycle_still_answers():
    """The gate must not fire when the history is long enough. Otherwise it is not a gate."""
    import numpy as np, pandas as pd
    from casefile.engine import baseline as B
    rng = np.random.default_rng(4)
    n = 900
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    s = pd.Series(1000 * (1 + 0.05 * np.sin(2 * np.pi * np.arange(n) / 7))
                  * rng.lognormal(0, 0.02, n), index=idx)
    C = {"kpis": {"net_revenue": {"min_history_days": 180,
                                  "seasonality": {"weekly": True, "yearly": True}}},
         "decision_policy": {**CONTRACT["decision_policy"]}}
    bl = B.fit("net_revenue", s, C, test_start=idx[880])
    assert bl.seasonal_coverage_ok, bl.note
    assert bl.dropped_periods == ()


def test_every_abstain_type_in_code_is_declared_in_the_contract():
    """The taxonomy is governed. Code that can emit a type the contract does not list is a
    governance hole, and this test previously would have failed."""
    from casefile.engine import verdict as VD
    declared = set(CONTRACT["decision_policy"]["abstain_types"])
    emitted = {t for t in VD.ABSTAIN_TYPES
               if t not in ("budget_exceeded_latency", "entitlement_limited")}
    missing = emitted - declared
    assert not missing, f"code can emit {sorted(missing)} but the contract does not declare it"


def test_the_batch_evaluation_never_reads_the_ground_truth_flag_to_decide():
    """The circularity guard.

    The harness used to set the proof rung from inc["identifiable_by_construction"], which
    is the simulator's own answer. With R3 required to name a cause, that made abstention
    on unidentifiable incidents a property of the harness rather than a measurement. The
    rung must now come from the estimator.
    """
    import inspect, sys
    sys.path.insert(0, str(ROOT / "src"))
    from eval import run_batch as RB

    import ast, textwrap

    def body_without_docstring(fn):
        """The docstrings here discuss the flag by name, so compare code, not prose."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return ast.unparse(node)

    est = body_without_docstring(RB._estimated_rungs)
    obs = body_without_docstring(RB._units_where_driver_moved)
    assert "identifiable" not in est, "the estimator consults the ground-truth flag"
    assert "identifiable" not in obs, "the scope observer consults the ground-truth flag"
    assert "CA.estimate" in est, "the estimator does not actually run synthetic control"

    sig = inspect.signature(RB.one)
    assert sig.parameters["rung_mode"].default == "estimated", (
        "the oracle path must not be the default")
