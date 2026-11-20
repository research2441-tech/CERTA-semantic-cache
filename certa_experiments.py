"""
certa_experiments.py
Runs every experiment reported in the paper and writes results/*.json.
E1 main comparison, E2 ablation, E3 sensitivity (alpha, V), E4 robustness (attack rate,
Byzantine cells, sybil budget / aged accounts), E5 generalisation (PAWS shift, workload
shift), E6 efficiency, E7 statistics, E8 risk-control calibration, E9 failure analysis.
"""
import os, json, time, itertools
import numpy as np
from multiprocessing import Pool
from scipy import stats
from certa_sim import (Assets, run_one, paws_shift, METHODS, ABLATIONS, DEFAULT_CFG)

OUT = "results"
FAST = os.environ.get("CERTA_FAST", "0") == "1"      # quick smoke run: fewer seeds
SEEDS10 = [1, 2, 3] if FAST else list(range(1, 11))
SEEDS5 = [1, 2] if FAST else list(range(1, 6))
_A = None


def _init():
    global _A
    _A = Assets()


def _job(args):
    method, seed, cfg_over, flag_over = args
    t = time.perf_counter()
    r = run_one(_A, method, seed, cfg_over=cfg_over, flag_over=flag_over)
    r["wall"] = time.perf_counter() - t
    for k in ("lat_sample", "tseries", "grp_n", "grp_fr", "taus_series", "adv_cells_list"):
        if k in r and method not in ("CERTA", "Static-C"):
            r.pop(k)
    return r


def run_jobs(jobs, procs=2):
    with Pool(procs, initializer=_init) as p:
        return p.map(_job, jobs, chunksize=1)


def agg(rs, keys=("valid_hit", "frr", "asr", "poison_rate", "mean_lat", "p95_lat", "energy",
                  "llm_per_req", "cloud_share", "remote_share", "verif_per_req", "mean_tau", "p99_lat")):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rs if k in r and r[k] is not None], float)
        v = v[~np.isnan(v)] if len(v) else v
        if len(v):
            boot = np.random.default_rng(0).choice(v, size=(2000, len(v))).mean(1)
            out[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                          lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                          vals=v.tolist())
    return out


def worst_group(r, min_n=200):
    n = np.array(r["grp_n"]); f = np.array(r["grp_fr"]); m = n >= min_n
    return float((f[m] / n[m]).max())


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * pvals[k]); adj[k] = min(1.0, run)
    return adj


def extra_experiments(procs=2):
    out = {}
    pols = [("Static-C", None), ("CERTA w/o probation", dict(probation=False)), ("CERTA", None)]
    jobs = [("Static-C" if nm == "Static-C" else "CERTA", s, None, fo) for nm, fo in pols for s in SEEDS10]
    res = run_jobs(jobs, procs)
    keys = ("adv_inserted", "adv_activated", "adv_removed_trust", "adv_served", "adv_cells_mean",
            "adv_cells_max", "adv_entries_served", "asr", "adv_occupancy", "valid_hit")
    out["E10_spread"] = {nm: agg(res[i * len(SEEDS10):(i + 1) * len(SEEDS10)], keys) for i, (nm, _) in enumerate(pols)}
    gs = [0.99, 0.995, 0.999, 0.9995]
    jobs = [("CERTA", s, dict(gamma=gm), None) for gm in gs for s in SEEDS5]
    res = run_jobs(jobs, procs)
    out["E3_gamma"] = {str(gm): agg(res[i * len(SEEDS5):(i + 1) * len(SEEDS5)]) for i, gm in enumerate(gs)}
    # imperfect answer-comparison judge
    jud = [("none", dict()), ("fa 0.10", dict(judge_fa=0.10)), ("fa 0.20", dict(judge_fa=0.20)),
           ("fd 0.10", dict(judge_fd=0.10)), ("fd 0.10, corrected", dict(judge_fd=0.10, judge_correct=True)),
           ("fd 0.20", dict(judge_fd=0.20)), ("fd 0.20, corrected", dict(judge_fd=0.20, judge_correct=True)),
           ("fa 0.10 + fd 0.10", dict(judge_fa=0.10, judge_fd=0.10)),
           ("fa 0.10 + fd 0.10, corrected", dict(judge_fa=0.10, judge_fd=0.10, judge_correct=True))]
    jobs = [("CERTA", s, co, None) for _, co in jud for s in SEEDS5]
    res = run_jobs(jobs, procs)
    out["E11_judge"] = {nm: agg(res[i * len(SEEDS5):(i + 1) * len(SEEDS5)]) for i, (nm, _) in enumerate(jud)}
    # adaptive forgery
    jobs = [("CERTA", s, dict(byz_cells=f, byz_mode="adaptive"), fo) for f in (1, 2)
            for fo in (None, dict(agg="mean")) for s in SEEDS5]
    res = run_jobs(jobs, procs)
    out["E12_adaptive_byz"] = {}
    k = 0
    for f in (1, 2):
        for nm in ("trimmed", "mean"):
            out["E12_adaptive_byz"][f"{nm} f={f}"] = agg(res[k:k + len(SEEDS5)]); k += len(SEEDS5)
    return out


def main(procs=2):
    os.makedirs(OUT, exist_ok=True)
    R = {}
    t0 = time.time()
    # ---------------- E1 main comparison ----------------
    meths = list(METHODS.keys())
    jobs = [(m, s, None, None) for m in meths for s in SEEDS10]
    res = run_jobs(jobs, procs)
    E1 = {m: [r for (mm, _, _, _), r in zip(jobs, res) if mm == m] for m in meths}
    R["E1"] = {m: agg(E1[m]) for m in meths}
    for m in ("CERTA", "Static-C"):
        R["E1"][m]["worst_group"] = [worst_group(r) for r in E1[m]]
        R["E1"][m]["lat_sample"] = E1[m][0]["lat_sample"]
        R["E1"][m]["tseries"] = E1[m][0]["tseries"]
        R["E1"][m]["taus_series"] = E1[m][0]["taus_series"]
        R["E1"][m]["grp"] = [E1[m][0]["grp_n"], E1[m][0]["grp_fr"]]
    for m in ("Exact", "Static-L", "CRC-Off", "PE-Online"):
        pass
    R["E1"]["CRC-Off"]["crc_tau"] = [r["crc_tau"] for r in E1["CRC-Off"]]
    print(f"E1 done {time.time()-t0:.0f}s", flush=True)

    # ---------------- E7 statistics (paired by seed) ----------------
    st = {}
    for metric, better in [("frr", "lower"), ("asr", "lower"), ("valid_hit", "higher"),
                           ("p95_lat", "lower"), ("energy", "lower")]:
        rows, ps = [], []
        for m in meths:
            if m == "CERTA":
                continue
            a = np.array([r[metric] for r in E1["CERTA"]]); b = np.array([r[metric] for r in E1[m]])
            d = a - b
            if np.allclose(d, 0):
                p, W = 1.0, 0.0
            else:
                W, p = stats.wilcoxon(a, b, zero_method="zsplit")
            npos = (d > 0).sum(); nneg = (d < 0).sum()
            rbc = (npos - nneg) / len(d)          # matched-pairs rank-biserial (sign form)
            ranks = stats.rankdata(np.abs(d)); rp = ranks[d > 0].sum(); rn = ranks[d < 0].sum()
            rbc = float((rp - rn) / ranks.sum()) if ranks.sum() > 0 else 0.0
            rows.append(dict(baseline=m, mean_diff=float(d.mean()), W=float(W), p=float(p), rbc=rbc))
            ps.append(p)
        adj = holm(np.array(ps))
        for r_, pa in zip(rows, adj):
            r_["p_holm"] = float(pa)
        st[metric] = rows
    R["E7"] = st

    # ---------------- E2 ablation ----------------
    ab_jobs = [("CERTA", s, None, None) for s in SEEDS10]
    for name, fo in ABLATIONS.items():
        ab_jobs += [("CERTA", s, None, fo) for s in SEEDS10]
    # robust aggregation also under one forged-report cell
    ab_jobs += [("CERTA", s, dict(byz_cells=1), None) for s in SEEDS10]
    ab_jobs += [("CERTA", s, dict(byz_cells=1), dict(agg="mean")) for s in SEEDS10]
    res = run_jobs(ab_jobs, procs)
    names = ["CERTA (full)"] + list(ABLATIONS.keys()) + ["CERTA (full), f=1", "w/o robust aggregation, f=1"]
    E2 = {}
    for i, nm in enumerate(names):
        rs = res[i * len(SEEDS10):(i + 1) * len(SEEDS10)]
        E2[nm] = agg(rs)
        if "grp_n" in rs[0]:
            E2[nm]["worst_group"] = [worst_group(r) for r in rs]
    R["E2"] = E2
    print(f"E2 done {time.time()-t0:.0f}s", flush=True)

    # ---------------- E3 + E8 alpha sweep (risk-control calibration) ----------------
    alphas = [0.01, 0.02, 0.05, 0.10]
    jobs = [(m, s, dict(alpha=a), None) for a in alphas for m in ("CERTA", "CRC-Off", "PE-Online")
            for s in SEEDS5]
    res = run_jobs(jobs, procs)
    E8 = {}
    for (m, s, co, _), r in zip(jobs, res):
        E8.setdefault(m, {}).setdefault(str(co["alpha"]), []).append(r)
    R["E8"] = {m: {a: agg(v) for a, v in d.items()} for m, d in E8.items()}
    # verification-rate, margin-gain and trust-threshold sensitivity
    sens = [("p_v", [0.02, 0.05, 0.10, 0.20]), ("W_R", [50.0, 200.0, 800.0, 3200.0]),
            ("th_act", [0.4, 0.6, 0.8, 1.0, 1.2])]
    jobs = []
    for key, vals in sens:
        for v in vals:
            co = {key: v, "th_share": v} if key == "th_act" else {key: v}
            if key == "th_act":
                co.update(sybils=8)
            jobs += [("CERTA", s, co, None) for s in SEEDS5]
    res = run_jobs(jobs, procs)
    E3 = {}; k = 0
    for key, vals in sens:
        for v in vals:
            E3.setdefault(key, {})[str(v)] = agg(res[k:k + len(SEEDS5)]); k += len(SEEDS5)
    R["E3"] = E3
    # tuned static-threshold frontier (strongest fixed-threshold competitor)
    taus = [0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96]
    jobs = [("Static-C", s, dict(tau_static=t), None) for t in taus for s in SEEDS5]
    res = run_jobs(jobs, procs)
    R["E1_static_sweep"] = {str(t): agg(res[i * len(SEEDS5):(i + 1) * len(SEEDS5)]) for i, t in enumerate(taus)}
    print(f"E3 done {time.time()-t0:.0f}s", flush=True)

    # ---------------- E4 robustness ----------------
    rates = [0.0, 0.02, 0.05, 0.10, 0.20]
    rm = ["Static-C", "CRC-Off", "PE-Online", "CERTA", "CERTA-noprob"]
    jobs = []
    for a in rates:
        for m in rm:
            for s in SEEDS5:
                if m == "CERTA-noprob":
                    jobs.append(("CERTA", s, dict(attack_frac=a), dict(probation=False)))
                else:
                    jobs.append((m, s, dict(attack_frac=a), None))
    res = run_jobs(jobs, procs)
    E4a = {}
    k = 0
    for a in rates:
        for m in rm:
            E4a.setdefault(m, {})[str(a)] = agg(res[k:k + len(SEEDS5)]); k += len(SEEDS5)
    R["E4_attack"] = E4a
    jobs = [("CERTA", s, dict(byz_cells=f), fo) for f in (0, 1, 2) for fo in (None, dict(agg="mean"))
            for s in SEEDS5]
    res = run_jobs(jobs, procs)
    E4b = {}; k = 0
    for f in (0, 1, 2):
        for nm in ("trimmed", "mean"):
            E4b.setdefault(nm, {})[str(f)] = agg(res[k:k + len(SEEDS5)]); k += len(SEEDS5)
    R["E4_byz"] = E4b
    syb = [2, 4, 6, 8, 12]
    trusts = [None, 0.5, 0.9]
    jobs = [("CERTA", s, dict(sybils=sb, attacker_trust=tr), None) for tr in trusts for sb in syb
            for s in SEEDS5]
    res = run_jobs(jobs, procs)
    E9 = {}; k = 0
    for tr in trusts:
        for sb in syb:
            E9.setdefault(str(tr), {})[str(sb)] = agg(res[k:k + len(SEEDS5)]); k += len(SEEDS5)
    R["E9_sybil"] = E9
    print(f"E4/E9 done {time.time()-t0:.0f}s", flush=True)

    # ---------------- E5 generalisation ----------------
    A = Assets()
    E5 = {}
    for a in (0.02, 0.05, 0.10):
        crc = np.mean(R["E8"]["CRC-Off"][str(a)]["mean_tau"]["vals"]) if False else None
        taus = [r for r in E8["CRC-Off"][str(a)]]
        crc_tau = float(np.mean([r["crc_tau"] for r in taus]))
        reps = [paws_shift(A, alpha=a, tau_crc=crc_tau, seed=s) for s in SEEDS5]
        E5[str(a)] = {m: dict(frr=float(np.mean([x[m]["frr"] for x in reps])),
                              frr_sd=float(np.std([x[m]["frr"] for x in reps], ddof=1)),
                              valid_hit=float(np.mean([x[m]["valid_hit"] for x in reps])),
                              tau=float(np.mean([x[m]["tau"] for x in reps])))
                      for m in reps[0]}
    R["E5_paws"] = E5
    shifts = [("zipf 0.7", dict(zipf_s=0.7)), ("zipf 1.1", dict(zipf_s=1.1)),
              ("4 cells", dict(n_cells=4)), ("10 cells", dict(n_cells=10, req_per_cell=4000)),
              ("cache 500", dict(cache_cap=500)), ("local mix 0.7", dict(local_mix=0.7))]
    jobs = [(m, s, co, None) for _, co in shifts for m in ("Static-C", "CRC-Off", "CERTA") for s in SEEDS5]
    res = run_jobs(jobs, procs)
    E5b = {}; k = 0
    for nm, _ in shifts:
        for m in ("Static-C", "CRC-Off", "CERTA"):
            E5b.setdefault(nm, {})[m] = agg(res[k:k + len(SEEDS5)]); k += len(SEEDS5)
    R["E5_workload"] = E5b
    # oracle-tuned static threshold (0.92 matched CERTA risk in-distribution): does it transfer?
    jobs = [("Static-C", s, dict(co, tau_static=0.92), None) for _, co in shifts for s in SEEDS5]
    res = run_jobs(jobs, procs)
    for i, (nm, _) in enumerate(shifts):
        R["E5_workload"][nm]["Static-C (0.92)"] = agg(res[i * len(SEEDS5):(i + 1) * len(SEEDS5)])
    sP, lP = A.paws_sim.astype(float), A.paws_lab.astype(bool)
    R["E5_paws_static"] = {str(t): dict(frr=float(np.mean((sP >= t) & ~lP)), valid_hit=float(np.mean((sP >= t) & lP)))
                           for t in (0.80, 0.92, 0.96, 0.98, 0.99)}
    print(f"E5 done {time.time()-t0:.0f}s", flush=True)

    # ---------------- E10 cross-cell spread of planted entries; decay sweep ----------------
    R.update(extra_experiments(procs))

    # ---------------- E6 efficiency ----------------
    E6 = {}
    for m in meths:
        w = [r["wall"] for r in E1[m]]
        n_req = DEFAULT_CFG["n_cells"] * DEFAULT_CFG["req_per_cell"] * 1.05
        E6[m] = dict(us_per_req=float(np.mean(w) / n_req * 1e6), wall=float(np.mean(w)))
    R["E6"] = E6
    R["runtime_s"] = time.time() - t0
    R["cfg"] = DEFAULT_CFG
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"))
    print(f"ALL DONE {time.time()-t0:.0f}s", flush=True)
    return R


if __name__ == "__main__":
    main()
