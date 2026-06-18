"""
Verification harness: compare regenerated revision JSONs against the values
recorded in RESULTS_SUMMARY.md, and report acceptance criteria for the Phase-2
strengthening experiments. Read-only; prints a PASS/FAIL table + overall verdict.

Usage:
  python scripts/verify_reproduction.py
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(_HERE), "results")

TOL = 5e-3        # deterministic point values (WM deterministic=True)
TOL_OPE = 1.2e-2  # ope_validation uses a sampling predict path

rows = []  # (name, status, detail)


def _add(name, ok, detail):
    rows.append((name, "PASS" if ok else ("SKIP" if ok is None else "FAIL"), detail))


def _load(fname):
    p = os.path.join(RES, fname)
    return json.load(open(p)) if os.path.exists(p) else None


def near(a, b, tol=TOL):
    return a is not None and b is not None and abs(a - b) <= tol


def pareto_front(pts):
    P = sorted(pts, key=lambda p: (p["mae"], p["ce"]))
    fr, best = [], float("inf")
    for p in P:
        if p["ce"] < best - 1e-12:
            fr.append(p); best = p["ce"]
    return fr


def check_e1():
    # (file, sarc_mae, sarc_ce, dom_native, dom_matched)
    exp = {
        "cmp2": (0.233, 0.079, False, True),
        "cmp1": (0.492, 0.121, True, True),
        "sim_mild": (0.269, 0.321, False, False),
        "sim_medium": (0.333, 0.334, True, False),
    }
    for tb, (m, c, dn, dm) in exp.items():
        j = _load(f"e1_baseline_pareto_{tb}.json")
        if j is None:
            _add(f"E1 {tb}", None, "json missing"); continue
        s = j["sarc"]
        ok_pt = near(s["mae"], m) and near(s["ce"], c)
        # dominance: prefer persisted field, else recompute
        dom = j.get("dominance") or {
            f"box_{b}": all((s["mae"] <= p["mae"] and s["ce"] <= p["ce"])
                            for p in pareto_front(j[f"box_{b}"])) for b in (3.0, 1.0)}
        ok_dom = (bool(dom["box_3.0"]) == dn) and (bool(dom["box_1.0"]) == dm)
        _add(f"E1 {tb}", ok_pt and ok_dom,
             f"SARC {s['mae']:.3f}/{s['ce']:.3f} (exp {m}/{c}); "
             f"dom native={dom['box_3.0']}(exp {dn}) matched={dom['box_1.0']}(exp {dm})")


def check_e2_ope():
    exp = {"cmp2": (0.469, 0.087, 0.0234), "cmp1": (0.105, 0.0035, 0.070)}
    for tb, (one, sens, ood) in exp.items():
        j = _load(f"e2_ope_{tb}.json")
        if j is None:
            _add(f"E2 ope {tb}", None, "json missing"); continue
        ac = j["action_conditional"]
        ok = (near(ac["one_step_rr_mae"], one, TOL_OPE)
              and near(ac["action_sensitivity_per_unit"], sens, TOL_OPE)
              and near(j["ood_action"]["SARC"]["mean"], ood, TOL_OPE))
        _add(f"E2 ope {tb}", ok,
             f"1-step={ac['one_step_rr_mae']:.3f}(exp {one}) "
             f"sens={ac['action_sensitivity_per_unit']:.4f}(exp {sens}) "
             f"OOD_SARC={j['ood_action']['SARC']['mean']:.4f}(exp {ood})")


def check_e3_age():
    exp = {"cmp2": 0.298, "cmp1": 0.502}
    for tb, m in exp.items():
        j = _load(f"e3_age_dewma_{tb}.json")
        if j is None:
            _add(f"E3 age {tb}", None, "json missing"); continue
        got = j["age_D-EWMA_bestMAE"]["mae"]
        sarc = j["SARC"]["mae"]
        _add(f"E3 age {tb}", near(got, m, 1e-2) and sarc < got,
             f"age-D-EWMA={got:.3f}(exp {m}), SARC={sarc:.3f} (<age? {sarc < got})")


def check_e4():
    j = _load("e4_nu_cmp2.json")
    if j is None:
        _add("E4 NU cmp2", None, "json missing"); return
    s, d, k = j["SARC"]["NU"], j["D-EWMA"]["NU"], j["Kalman"]["NU"]
    ok = near(s, 0.3425, 1e-2) and s < d and s < k
    _add("E4 NU cmp2", ok, f"SARC={s:.4f} < D-EWMA={d:.4f} < Kalman={k:.4f}")


def check_stats():
    j = _load("stats_all.json") or _load("stats_e1_matched.json")
    if j is None:
        _add("stats E1 matched", None, "stats json missing"); return
    want = {"E1 matched(±1) cmp2": ("***", "***"),
            "E1 matched(±1) cmp1": ("***", "***")}
    for label, (mae_sig, ce_sig) in want.items():
        d = j.get(label)
        if d is None:
            _add(f"stats {label}", None, "label missing"); continue
        mae_ok = any(d[k]["sig"] == mae_sig for k in d if k.startswith("MAE"))
        ce_ok = any(d[k]["sig"] == ce_sig for k in d if k.startswith("CE"))
        _add(f"stats {label}", mae_ok and ce_ok, f"MAE/CE both {mae_sig}? {mae_ok and ce_ok}")
    # sim CE must be ns (trade-off)
    for reg in ("sim_mild", "sim_medium"):
        d = j.get(f"E1 matched(±1) {reg}")
        if d is None:
            _add(f"stats {reg} CE", None, "label missing"); continue
        ce_ns = all(d[k]["sig"] == "ns" for k in d if k.startswith("CE"))
        _add(f"stats {reg} CE=ns", ce_ns, f"CE ns (trade-off)? {ce_ns}")


def main():
    check_e1(); check_e2_ope(); check_e3_age(); check_e4(); check_stats()
    print(f"\n{'='*78}\nREPRODUCTION VERIFICATION\n{'='*78}")
    w = max(len(n) for n, _, _ in rows)
    npass = nfail = nskip = 0
    for name, status, detail in rows:
        print(f"  [{status:4}] {name:<{w}}  {detail}")
        npass += status == "PASS"; nfail += status == "FAIL"; nskip += status == "SKIP"
    print(f"{'-'*78}\n  {npass} PASS / {nfail} FAIL / {nskip} SKIP")
    print("  VERDICT:", "ALL CHECKS PASS" if nfail == 0 else f"{nfail} FAILURE(S) -- investigate")


if __name__ == "__main__":
    main()
