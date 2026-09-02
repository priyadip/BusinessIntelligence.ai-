"""
Materialise the simulated world into a DuckDB warehouse with THREE schemas that behave
like three different source systems: different grains, different refresh cadences,
different lag, different quality tiers, and a deliberately DIFFERENT revenue definition
in finance so the reconciliation layer has real work to do.
"""
from __future__ import annotations
import json, math
from datetime import date, timedelta, datetime
import numpy as np, pandas as pd, duckdb
from .model import World, REGIONS, CHANNELS, CATEGORIES, CARRIERS, END
from .scenarios import incident_interventions

HOT_WINDOW_DAYS = 940           # order-grain detail is kept hot; older history is aggregated
AS_OF = datetime(2026, 8, 31, 23, 45)


def build(db_path: str, corpus_path: str, seed: int = 20260901) -> dict:
    ivs = incident_interventions()
    panel = World(ivs, seed=seed).simulate()
    panel["d"] = pd.to_datetime(panel["d"])
    rng = np.random.default_rng(seed)

    hot_from = pd.Timestamp(END - timedelta(days=HOT_WINDOW_DAYS))
    hot = panel[panel.d >= hot_from].copy()

    # ---------------------------------------------------------- commerce.sessions
    sess = hot[["d","region","channel","category","sessions","orders"]].copy()
    sess["landed"] = sess.sessions.round().astype(int)
    sess["converted"] = sess.orders.round().astype(int)
    sessions = (sess.groupby(["d","region","channel","category"], as_index=False)
                    [["landed","converted"]].sum())
    sessions["session_grain_note"] = "pre-aggregated from session events at load time"

    # ---------------------------------------------------------- commerce.orders (order grain)
    orders_rows = []
    for row in hot.itertuples(index=False):
        n = int(round(row.orders))
        if n <= 0:
            continue
        n_emit = min(n, 20)                       # cap per cell/day, scale value to preserve totals
        scale = n / n_emit
        jitter = rng.lognormal(0, 0.42, n_emit); jitter /= jitter.mean()
        for k in range(n_emit):
            orders_rows.append((
                f"O{row.d.strftime('%y%m%d')}{abs(hash((row.region,row.channel,row.category,k)))%10**7:07d}",
                row.d, row.region, row.channel, row.category,
                str(rng.choice(["NEW","RETURNING","VIP"], p=[0.31,0.55,0.14])),
                f"SITE-{rng.integers(10,99)}",
                float(row.gross_amount    / n * scale * jitter[k]),
                float(row.discount_amount / n * scale * jitter[k]),
                float(row.returns_amount  / n * scale * jitter[k]),
                float(row.cogs            / n * scale * jitter[k]),
                int(max(1, round(scale))),
            ))
    orders = pd.DataFrame(orders_rows, columns=[
        "order_id","order_ts","region","channel","category","customer_segment","site_id",
        "gross_amount","discount_amount","returns_amount","cogs_amount","order_multiplicity"])

    # ---------------------------------------------------------- commerce.checkout_events
    ck = hot.groupby(["d","region","channel"], as_index=False).agg(
        checkout_error_rate=("checkout_error_rate","mean"), sessions=("sessions","sum"))
    ck["attempts"] = (ck.sessions * 0.34).round().astype(int)
    ck["errors"] = (ck.attempts * ck.checkout_error_rate * rng.uniform(0.9,1.1,len(ck))).round().astype(int)
    checkout_events = ck[["d","region","channel","attempts","errors"]]

    # ---------------------------------------------------------- commerce.price_promo_daily
    pp = hot.groupby(["d","region","category"], as_index=False).agg(
        list_price_index=("price_index","mean"), promo_depth=("promo_depth","mean"),
        gross=("gross_amount","sum"), disc=("discount_amount","sum"))
    pp["realised_discount_pct"] = pp.disc / pp.gross.clip(lower=1)
    price_promo_daily = pp[["d","region","category","list_price_index","promo_depth","realised_discount_pct"]]

    # ---------------------------------------------------------- ops_voice.shipment_events
    ship = []
    carrier_p = {"C_SWIFT":0.34,"C_METRO":0.30,"C_BLUEDART_LIKE":0.29,"C_NEWCO":0.07}
    ont = hot.set_index(["d","region"]).ontime_pct.groupby(level=[0,1]).mean().to_dict()
    dom = hot.set_index(["d","region"]).dominant_carrier.groupby(level=[0,1]).first().to_dict()
    for i, o in enumerate(orders.itertuples(index=False)):
        if i % 3:                                  # sample: telemetry covers a subset, realistic
            continue
        key = (o.order_ts, o.region)
        p = ont.get(key, 0.94)
        carrier = dom.get(key) if rng.random() < 0.55 else str(rng.choice(CARRIERS, p=list(carrier_p.values())))
        promised = o.order_ts + timedelta(days=3)
        late_h = 0 if rng.random() < p else float(rng.gamma(2.2, 14))
        ship.append((f"E{i:08d}", f"S{i:08d}", o.order_id, o.region, carrier,
                     promised, promised + timedelta(hours=late_h),
                     "DELIVERED" if rng.random() > 0.012 else "IN_TRANSIT"))
    shipment_events = pd.DataFrame(ship, columns=[
        "event_id","shipment_id","order_id","region","carrier_id","promised_ts","delivered_ts","status"])

    # ---------------------------------------------------------- ops_voice.inventory_daily
    inv = hot.groupby(["d","region","category"], as_index=False).agg(
        stockout_rate=("stockout_rate","mean"), orders=("orders","sum"))
    inv["sku_id"] = "SKU-AGG-" + inv.category
    inv["on_hand"] = (inv.orders * rng.uniform(3, 9, len(inv))).round().astype(int)
    inv["oos_minutes"] = (inv.stockout_rate * 1440).round().astype(int)
    inventory_daily = inv[["sku_id","d","region","on_hand","oos_minutes"]]

    # ---------------------------------------------------------- finance_erp (WEEKLY grain, lagged)
    fin_cut = pd.Timestamp(END - timedelta(days=3))         # 3-day publication lag
    fp = panel[panel.d <= fin_cut].copy()
    fp["iso_week"] = fp.d.dt.strftime("%G-W%V")
    fin_rev = fp.groupby(["region","iso_week"], as_index=False).agg(
        net_revenue_commerce=("net_revenue","sum"), cogs=("cogs","sum"))
    # ALTERNATE DEFINITION: finance recognises at ship date and nets supplier rebates.
    # This produces a small, real, explainable delta the reconciler must surface.
    rebate = rng.uniform(0.0022, 0.0061, len(fin_rev))
    shift  = rng.normal(0.0, 0.0018, len(fin_rev))
    fin_rev["net_revenue_fin"] = fin_rev.net_revenue_commerce * (1 - rebate + shift)
    fin_revenue = fin_rev[["region","iso_week","net_revenue_fin"]]

    skuw = fp.groupby(["category","iso_week"], as_index=False).agg(
        cogs=("cogs","sum"), orders=("orders","sum"))
    skuw["sku_id"] = "SKU-AGG-" + skuw.category
    skuw["unit_cost"]   = skuw.cogs / skuw.orders.clip(lower=1)
    skuw["landed_cost"] = skuw.unit_cost * rng.uniform(1.03, 1.09, len(skuw))
    skuw["supplier_id"] = "SUP-" + (skuw.index % 7).astype(str)
    sku_week_cost = skuw[["sku_id","iso_week","unit_cost","landed_cost","supplier_id","category"]]

    promo = fp.groupby(["region","category","iso_week"], as_index=False).agg(
        promo_depth_pct=("promo_depth","mean"), nr=("net_revenue","sum"))
    promo["campaign_id"] = "CMP-" + promo.category.str[:3] + "-" + promo.iso_week.str[-3:]
    promo["spend"] = promo.nr * (0.048 + promo.promo_depth_pct.abs() * 0.5)
    promo_spend = promo[["campaign_id","iso_week","category","region","spend","promo_depth_pct"]]

    # ---------------------------------------------------------- freshness watermarks
    watermarks = pd.DataFrame([
        # source, table, max_event_ts, loaded_at, sla_minutes  -> the engine computes staleness
        ("commerce","orders",           str(END),                 str(AS_OF),                     90),
        ("commerce","sessions",         str(END),                 str(AS_OF),                     90),
        ("commerce","checkout_events",  str(END),                 str(AS_OF),                     90),
        ("commerce","price_promo_daily",str(END),                 str(AS_OF),                     90),
        ("ops_voice","shipment_events", str(END),                 str(AS_OF - timedelta(minutes=52)), 240),
        ("ops_voice","inventory_daily", str(END),                 str(AS_OF - timedelta(minutes=95)), 240),
        ("finance_erp","fin_revenue",   str(END - timedelta(days=3)), str(AS_OF - timedelta(days=3)), 11520),
        ("finance_erp","sku_week_cost", str(END - timedelta(days=18)), str(AS_OF - timedelta(days=18)), 11520),  # STALE on purpose
        ("finance_erp","promo_spend",   str(END - timedelta(days=3)), str(AS_OF - timedelta(days=3)), 11520),
    ], columns=["source","table_name","max_event_ts","loaded_at","sla_minutes"])

    corpus = pd.read_json(corpus_path, lines=True)
    corpus["meta"] = corpus["meta"].apply(json.dumps)

    # ---------------------------------------------------------- write
    con = duckdb.connect(db_path)
    for s in ("commerce","finance_erp","ops_voice","meta"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
    def put(schema, name, df):
        con.register("_t", df)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{name} AS SELECT * FROM _t")
        con.unregister("_t")
    put("commerce","orders",orders); put("commerce","sessions",sessions)
    put("commerce","checkout_events",checkout_events)
    put("commerce","price_promo_daily",price_promo_daily)
    put("ops_voice","shipment_events",shipment_events); put("ops_voice","inventory_daily",inventory_daily)
    put("ops_voice","documents",corpus)
    put("finance_erp","fin_revenue",fin_revenue); put("finance_erp","sku_week_cost",sku_week_cost)
    put("finance_erp","promo_spend",promo_spend)
    put("meta","watermarks",watermarks)
    put("meta","daily_panel_truth", panel)          # latent drivers, for evaluation ONLY

    stats = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in [
        "commerce.orders","commerce.sessions","commerce.checkout_events",
        "commerce.price_promo_daily","ops_voice.shipment_events",
        "ops_voice.inventory_daily","ops_voice.documents","finance_erp.fin_revenue",
        "finance_erp.sku_week_cost","finance_erp.promo_spend","meta.watermarks"]}
    recon = con.execute("""
        SELECT SUM(f.net_revenue_fin) AS fin, SUM(x.nr) AS com
        FROM finance_erp.fin_revenue f
        JOIN (SELECT region, strftime(order_ts,'%G-W%V') AS iso_week,
                     SUM(gross_amount-discount_amount-returns_amount) AS nr
              FROM commerce.orders GROUP BY 1,2) x
          ON x.region=f.region AND x.iso_week=f.iso_week
    """).fetchone()
    con.close()
    return {"tables": stats,
            "reconciliation_delta_pct": round(100*(recon[0]-recon[1])/recon[1], 3) if recon[1] else None}
