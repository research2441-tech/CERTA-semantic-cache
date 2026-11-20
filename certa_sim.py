"""
certa_sim.py
Event-driven multi-cell simulator for cooperative semantic caching with tiered LLM
inference offloading, plus the CERTA controller and all baselines.

Request service options: local cache hit, neighbouring-cell cache hit, edge LLM, cloud LLM.
Ground truth: a cached answer is valid for a request iff both belong to the same
paraphrase intent and the entry was not produced by an attacker query.
"""
import math
from collections import OrderedDict
import numpy as np

BIN_LO, BIN_W, NBIN = 0.50, 0.005, 100
GRID = BIN_LO + BIN_W * np.arange(NBIN)          # candidate thresholds


def s2bin(s):
    return int(min(NBIN - 1, max(0, math.floor((s - BIN_LO) / BIN_W))))


DEFAULT_CFG = dict(
    n_cells=6, req_per_cell=6000, rate=8.0, warmup_frac=0.25, cache_cap=1500,
    zipf_s=0.9, local_mix=0.3, users_per_cell=400, user_zipf=1.0,
    alpha=0.05, tau_static=0.80,
    attack_frac=0.05, sybils=4, attacker_trust=None, target_rank=(10, 600),
    byz_cells=0,
    # LLM tier profiles (assumed hardware profiles; see paper Table IV)
    radio_lat=0.010, lookup_lat=0.002, remote_lat=0.008,
    e_radio=0.5, e_lookup=0.05, e_remote=0.2,
    edge_servers=12, edge_power=350.0, cloud_power=5600.0, cloud_conc=48,
    wan_rtt=0.09, wan_energy=2.0,
    edge_ttft=0.04, edge_prefill=4000.0, edge_decode=60.0,
    cloud_ttft=0.15, cloud_prefill=3000.0, cloud_decode=35.0,
    # CERTA
    p_v=0.05, gamma=0.999, n_min=50, trim=1, W_R=200.0, eta_max_frac=0.5,
    u0=0.10, u_slope=0.02, tenure_median=30.0, th_act=0.8, th_share=0.8, th_del=-0.5, lam_d=1.0, penalty=0.25,
    calib_requests=4000, judge_fa=0.0, judge_fd=0.0, judge_correct=False, byz_mode="zero",
)

METHODS = {
    # name: flags
    "Exact":        dict(coop=False, thr="exact"),
    "Static-L":     dict(coop=False, thr="static"),
    "Static-C":     dict(coop=True,  thr="static"),
    "CRC-Off":      dict(coop=True,  thr="crc_off"),
    "PE-Online":    dict(coop=True,  thr="pe"),
    "CERTA":        dict(coop=True,  thr="certa", probation=True, verify=True, ipw=True,
                         groupwise=False, margin=True, agg="trim"),
}
ABLATIONS = {
    "+ group-conditional thresholds": dict(groupwise=True),
    "w/o IPW labels":         dict(ipw=False),
    "w/o risk-deficit margin": dict(margin=False),
    "w/o probation trust":    dict(probation=False),
    "w/o robust aggregation": dict(agg="mean"),
}


def method_flags(name, overrides=None):
    base = dict(coop=True, thr="static", probation=False, verify=False, ipw=True,
                groupwise=True, margin=False, agg="trim")
    base.update(METHODS.get(name, METHODS["CERTA"]) if name in METHODS else METHODS["CERTA"])
    if overrides:
        base.update(overrides)
    return base


class Assets:
    def __init__(self, path="certa_assets.npz"):
        z = np.load(path)
        self.nb_idx, self.nb_sim = z["nb_idx"], z["nb_sim"]
        self.n_benign = int(z["n_benign"]); self.n_pool = self.nb_idx.shape[0]
        self.n_intents = int(z["n_intents"]); self.n_para = int(z["n_para_intents"])
        intent = z["intent"].astype(np.int64).copy()
        self.adv_target = z["adv_target_intent"]; self.adv_variant = z["adv_variant"]
        # attacker answers for the same target are mutually consistent
        intent[self.n_benign:] = -1 - self.adv_target.astype(np.int64)
        self.intent = intent
        self.group = z["group"].astype(np.int64)
        self.G = int(self.group.max()) + 1
        self.ctx, self.gen = z["az_ctx"].astype(float), z["az_gen"].astype(float)
        self.ia = z["az_ia"].astype(float)
        self.paws_sim, self.paws_lab, self.paws_group = z["paws_sim"], z["paws_lab"], z["paws_group"]
        self.n_suffix = int(z["n_suffix"])
        mem = [[] for _ in range(self.n_intents)]
        for pid in range(self.n_benign):
            mem[intent[pid]].append(pid)
        self.members = [np.array(m, dtype=np.int64) for m in mem]
        adv_of = {}
        for k in range(len(self.adv_target)):
            adv_of.setdefault(int(self.adv_target[k]), []).append(self.n_benign + k)
        self.adv_of = adv_of


def service_times(A, cfg, pid):
    c, g = A.ctx[pid], A.gen[pid]
    s_edge = cfg["edge_ttft"] + c / cfg["edge_prefill"] + g / cfg["edge_decode"]
    s_cloud = cfg["cloud_ttft"] + c / cfg["cloud_prefill"] + g / cfg["cloud_decode"]
    e_edge = cfg["edge_power"] * s_edge / cfg["edge_servers"]
    e_cloud = cfg["cloud_power"] / cfg["cloud_conc"] * s_cloud + cfg["wan_energy"]
    return s_edge, s_cloud + cfg["wan_rtt"], e_edge, e_cloud


# ------------------------------------------------------------------ workload
def make_workload(A, cfg, seed, attack=True):
    rng = np.random.default_rng(seed)
    C, Nc = cfg["n_cells"], cfg["req_per_cell"]
    nI = A.n_intents
    w = 1.0 / np.arange(1, nI + 1) ** cfg["zipf_s"]; w /= w.sum()
    grank = rng.permutation(nI)                       # grank[r] = intent at rank r
    rank_of = np.empty(nI, np.int64); rank_of[grank] = np.arange(nI)
    times, cells, pids, users, is_att = [], [], [], [], []
    uw = 1.0 / np.arange(1, cfg["users_per_cell"] + 1) ** cfg["user_zipf"]; uw /= uw.sum()
    for c in range(C):
        lrank = rng.permutation(nI)
        r = rng.choice(nI, size=Nc, p=w)
        loc = rng.random(Nc) < cfg["local_mix"]
        it = np.where(loc, lrank[r], grank[r])
        p = np.array([A.members[i][rng.integers(len(A.members[i]))] for i in it])
        start = rng.integers(0, len(A.ia) - Nc - 1)
        ia = A.ia[start:start + Nc].copy()
        ia *= (1.0 / cfg["rate"]) / ia.mean()
        t = np.cumsum(ia)
        u = c * cfg["users_per_cell"] + rng.choice(cfg["users_per_cell"], size=Nc, p=uw)
        times.append(t); cells.append(np.full(Nc, c)); pids.append(p); users.append(u)
        is_att.append(np.zeros(Nc, bool))
    T_end = max(t[-1] for t in times)
    targets = []
    if attack and cfg["attack_frac"] > 0:
        lo, hi = cfg["target_rank"]
        cand = [ti for ti in A.adv_of if lo <= rank_of[ti] <= hi]
        if not cand:
            cand = sorted(A.adv_of, key=lambda ti: rank_of[ti])[:20]
        S = cfg["sybils"]
        n_camp = int(cfg["attack_frac"] * C * Nc / (1 + S))
        uid = 10_000_000
        for k in range(n_camp):
            ti = cand[rng.integers(len(cand))]
            targets.append(ti)
            c = rng.integers(C)
            t0 = rng.uniform(0.05 * T_end, 0.95 * T_end)
            advs = A.adv_of[ti]
            for j in range(S + 1):
                times.append(np.array([t0 + 0.4 * j]))
                cells.append(np.array([c])); pids.append(np.array([advs[j % len(advs)]]))
                users.append(np.array([uid])); is_att.append(np.array([True])); uid += 1
    t = np.concatenate(times); o = np.argsort(t, kind="stable")
    W = dict(t=t[o], cell=np.concatenate(cells)[o], pid=np.concatenate(pids)[o],
             user=np.concatenate(users)[o], att=np.concatenate(is_att)[o],
             targets=set(targets), T_end=T_end)
    return W


# ------------------------------------------------------------------ simulator
class Sim:
    def __init__(self, A, cfg, flags, seed, crc_tau=None, rng_seed=None):
        self.A, self.cfg, self.f, self.seed = A, cfg, flags, seed
        self.rng = np.random.default_rng(rng_seed if rng_seed is not None else seed + 991)
        C, P = cfg["n_cells"], A.n_pool
        self.present = np.zeros((C, P), bool)
        self.active = np.zeros((C, P), bool)
        self.shared = np.zeros((C, P), bool)
        self.shared_cnt = np.zeros(P, np.int32)
        self.lru = [OrderedDict() for _ in range(C)]
        self.meta = {}                       # (c,pid) -> dict
        self.user_n = {}; self.user_pen = {}
        G = A.G
        self.Hl = np.zeros((C, G, NBIN)); self.Hn = np.zeros((C, G))
        self.Hl_lab = np.zeros((C, G, NBIN)); self.Hn_lab = np.zeros((C, G))
        self.Hc = np.zeros((C, G, NBIN))         # weighted mass of all labelled events (for judge de-biasing)
        self.tau_g = np.ones(G)
        self.crc_tau = crc_tau
        self.ZR = 0.0; self.pv = cfg["p_v"]; self.now = 0.0
        self.jrng = np.random.default_rng((rng_seed if rng_seed is not None else seed) + 4242)
        self.edge_free = [np.zeros(cfg["edge_servers"]) for _ in range(C)]
        self.pe = {}                          # per-entry observations for PE-Online
        self.byz = set(range(cfg["byz_cells"]))
        self.slot_R = 0.0; self.slot_Ntot = 0
        self.adv = dict(inserted=0, activated=0, removed_trust=0, served=0, life=[], cells={}, t_ins={})

    # ---------------- answer-comparison judge (label noise) ----------------
    def judge(self, valid):
        if valid:
            return not (self.cfg["judge_fd"] > 0 and self.jrng.random() < self.cfg["judge_fd"])
        return bool(self.cfg["judge_fa"] > 0 and self.jrng.random() < self.cfg["judge_fa"])

    # ---------------- trust ----------------
    def init_users(self, W):
        rng = np.random.default_rng(self.seed + 7)
        ub = np.unique(W["user"][~W["att"]])
        prior = rng.lognormal(np.log(self.cfg["tenure_median"]), 1.0, size=len(ub)).astype(int)
        self.user_n = dict(zip(ub.tolist(), prior.tolist()))

    def utrust(self, user, att):
        if att and self.cfg["attacker_trust"] is not None:
            base = self.cfg["attacker_trust"]
        else:
            base = min(1.0, self.cfg["u0"] + self.cfg["u_slope"] * self.user_n.get(user, 0))
        return base * self.user_pen.get(user, 1.0)

    # ---------------- cache ops ----------------
    def _set_shared(self, c, pid, val):
        if self.shared[c, pid] != val:
            self.shared[c, pid] = val
            self.shared_cnt[pid] += 1 if val else -1

    def evict(self, c, pid, reason="lru"):
        if pid >= self.A.n_benign and self.present[c, pid] and (c, pid) in self.adv["t_ins"]:
            self.adv["life"].append((self.now - self.adv["t_ins"].pop((c, pid)), reason))
        self._evict(c, pid)

    def _evict(self, c, pid):
        self.present[c, pid] = False; self.active[c, pid] = False
        self._set_shared(c, pid, False)
        self.lru[c].pop(pid, None); self.meta.pop((c, pid), None); self.pe.pop((c, pid), None)

    def insert(self, c, pid, user, utr):
        if self.present[c, pid]:
            self.lru[c].move_to_end(pid); return
        if len(self.lru[c]) >= self.cfg["cache_cap"]:
            old, _ = self.lru[c].popitem(last=False)
            self.evict(c, old)
        self.lru[c][pid] = None
        self.present[c, pid] = True
        if pid >= self.A.n_benign:
            self.adv["inserted"] += 1; self.adv["t_ins"][(c, pid)] = self.now
        if self.f["probation"]:
            m = dict(T=utr, origin=user, conf=set())
            self.meta[(c, pid)] = m
            self.active[c, pid] = utr >= self.cfg["th_act"]
            if self.f["coop"]:
                self._set_shared(c, pid, utr >= self.cfg["th_share"])
        else:
            self.active[c, pid] = True
            if self.f["coop"]:
                self._set_shared(c, pid, True)

    def update_entry(self, c, pid, consistent, user, utr):
        m = self.meta.get((c, pid))
        if m is None:
            return
        cfg = self.cfg
        if consistent:
            if user != m["origin"] and user not in m["conf"]:
                m["T"] += utr; m["conf"].add(user)
        else:
            m["T"] -= cfg["lam_d"] * utr
        if m["T"] <= cfg["th_del"]:
            for u in [m["origin"], *m["conf"]]:
                self.user_pen[u] = self.user_pen.get(u, 1.0) * cfg["penalty"]
            self.evict(c, pid, "trust"); return
        if pid >= self.A.n_benign and not self.active[c, pid] and m["T"] >= cfg["th_act"]:
            self.adv["activated"] += 1
        self.active[c, pid] = m["T"] >= cfg["th_act"]
        if self.f["coop"]:
            self._set_shared(c, pid, m["T"] >= cfg["th_share"])

    # ---------------- lookup ----------------
    def candidates(self, c, q):
        nb = self.A.nb_idx[q]; ns = self.A.nb_sim[q]
        la = self.active[c, nb]
        il = int(np.argmax(la)) if la.any() else -1
        ir = -1
        if self.f["coop"]:
            ra = (self.shared_cnt[nb] - self.shared[c, nb]) > 0
            ir = int(np.argmax(ra)) if ra.any() else -1
        ip = -1
        if self.f["probation"]:
            pa = self.present[c, nb] & ~self.active[c, nb]
            ip = int(np.argmax(pa)) if pa.any() else -1
        best = None
        if il >= 0:
            best = (float(ns[il]), int(nb[il]), "L")
        if ir >= 0 and (best is None or ns[ir] > best[0] + 1e-9):
            best = (float(ns[ir]), int(nb[ir]), "R")
        prob = (float(ns[ip]), int(nb[ip])) if ip >= 0 else None
        return best, prob

    def threshold(self, q, entry_key):
        thr = self.f["thr"]
        if thr == "exact":
            return 0.9999
        if thr == "static":
            return self.cfg["tau_static"]
        if thr == "crc_off":
            return self.crc_tau
        if thr == "pe":
            ob = self.pe.get(entry_key)
            return 1.0 if ob is None else ob["t"]
        return self.tau_g[self.A.group[q]] if self.f["groupwise"] else self.tau_g[0]

    def holder(self, c, pid):
        for cc in range(self.cfg["n_cells"]):
            if cc != c and self.shared[cc, pid]:
                return cc
        return None

    # ---------------- PE-Online (per-entry online threshold) ----------------
    def pe_observe(self, key, s, valid):
        ob = self.pe.setdefault(key, dict(s=[], v=[], t=1.0))
        ob["s"].append(s); ob["v"].append(valid)
        if len(ob["s"]) < 5:
            return
        s_arr = np.array(ob["s"]); v_arr = np.array(ob["v"], float)
        o = np.argsort(-s_arr); s_arr, v_arr = s_arr[o], v_arr[o]
        n = np.arange(1, len(s_arr) + 1); err = np.cumsum(1 - v_arr)
        # Wilson upper bound on the error of reusing at threshold s_arr[k]
        z = 1.645; ph = err / n
        ub = (ph + z * z / (2 * n) + z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / (1 + z * z / n)
        ok = np.where((ub <= self.cfg["alpha"]) & (n >= 5))[0]
        ob["t"] = float(s_arr[ok.max()]) if len(ok) else 1.0

    # ---------------- calibration (CERTA) ----------------
    def record(self, c, g, s, valid, w, labelled=True):
        self.Hn[c, g] += 1.0
        if s is not None and labelled:
            self.Hc[c, g, s2bin(s)] += w
        if s is not None and not valid and labelled:
            self.Hl[c, g, s2bin(s)] += w
        if labelled:
            self.Hn_lab[c, g] += 1.0
            if s is not None and not valid:
                self.Hl_lab[c, g, s2bin(s)] += 1.0

    def recalibrate(self):
        cfg, f = self.cfg, self.f
        C = cfg["n_cells"]
        if f["ipw"]:
            Hl, Hn = self.Hl.copy(), self.Hn.copy()
            Hc = self.Hc.copy()
        else:
            Hl, Hn = self.Hl_lab.copy(), self.Hn_lab.copy()
        hon = [c for c in range(C) if c not in self.byz]
        for c in self.byz:
            if cfg["byz_mode"] == "zero":          # forged reports: zero loss, inflated count
                Hl[c] = 0.0; Hn[c] = 1e6
            else:                                   # adaptive: plausible curve below the honest range
                Hn[c] = np.median(Hn[hon], axis=0)
                r_min = (Hl[hon] / np.maximum(Hn[hon][..., None], 1.0)).min(0)
                Hl[c] = 0.5 * r_min * Hn[c][..., None]
        if not f["groupwise"]:
            Hl = Hl.sum(1, keepdims=True); Hn = Hn.sum(1, keepdims=True)
        cum = np.flip(np.cumsum(np.flip(Hl, -1), -1), -1)          # loss mass with s >= grid
        if f["ipw"] and cfg["judge_correct"] and (cfg["judge_fa"] > 0 or cfg["judge_fd"] > 0):
            # de-bias judged labels with known error rates: E[observed dispute] = (1-v)(1-fa-fd) + fd
            if not f["groupwise"]:
                Hc = Hc.sum(1, keepdims=True)
            cumc = np.flip(np.cumsum(np.flip(Hc, -1), -1), -1)
            for c in self.byz:
                cumc[c] = 0.0
            cum = np.maximum(cum - cfg["judge_fd"] * cumc, 0.0) / (1.0 - cfg["judge_fa"] - cfg["judge_fd"])
        B = 1.0 / cfg["p_v"] if f["ipw"] else 1.0
        eta = min(cfg["eta_max_frac"] * cfg["alpha"], self.ZR / cfg["W_R"]) if f["margin"] else 0.0
        a_eff = cfg["alpha"] - eta
        G = Hl.shape[1]
        tau = np.ones(G)
        for g in range(G):
            if f["agg"] == "mean":
                N = Hn[:, g].sum(); R = cum[:, g].sum(0) / max(N, 1.0)
            else:
                r = cum[:, g] / np.maximum(Hn[:, g:g + 1], 1.0)
                k = cfg["trim"]
                rs = np.sort(r, axis=0)
                R = rs[k:C - k].mean(0) if C - 2 * k > 0 else np.median(rs, 0)
                N = C * float(np.median(Hn[:, g]))
            if N < cfg["n_min"]:
                tau[g] = 1.0; continue
            bound = (N * R + B) / (N + 1.0)
            ok = np.where(bound <= a_eff)[0]
            tau[g] = GRID[ok.min()] if len(ok) else 1.0
        self.tau_g = tau if f["groupwise"] else np.full(self.A.G, tau[0])
        self.Hl *= cfg["gamma"]; self.Hn *= cfg["gamma"]
        self.Hl_lab *= cfg["gamma"]; self.Hn_lab *= cfg["gamma"]; self.Hc *= cfg["gamma"]

    # ---------------- slot-level Lyapunov updates ----------------
    def end_slot(self):
        cfg = self.cfg
        # risk-deficit virtual queue (network-wide, IPW-estimated false reuse)
        self.ZR = max(self.ZR + self.slot_R - cfg["alpha"] * self.slot_Ntot, 0.0)
        self.slot_R = 0.0; self.slot_Ntot = 0
        if self.f["thr"] == "certa":
            self.recalibrate()

    # ---------------- routing ----------------
    def route(self, c, t, q):
        cfg = self.cfg
        s_e, s_c, e_e, e_c = service_times(self.A, cfg, q)
        fr = self.edge_free[c]; k = int(np.argmin(fr)); wait = max(0.0, fr[k] - t)
        d_e, d_c = wait + s_e, s_c
        use_edge = d_e <= d_c                     # latency-greedy tier selection (all methods)
        if use_edge:
            fr[k] = t + wait + s_e
            return d_e, e_e, "edge"
        return d_c, e_c, "cloud"

    # ---------------- main loop ----------------
    def run(self, W, collect=True):
        A, cfg, f = self.A, self.cfg, self.f
        C = cfg["n_cells"]
        t_warm = cfg["warmup_frac"] * W["T_end"]
        n = len(W["t"])
        self.init_users(W)
        out = dict(lat=[], energy=0.0, n=0, hit_valid=0, hit_invalid=0, poison=0, tgt_req=0,
                   tgt_poison=0, llm_calls=0, cloud=0, miss=0, remote_hits=0, verif=0,
                   grp_n=np.zeros(A.G), grp_fr=np.zeros(A.G), taus=[], tseries=[])
        next_slot = 1.0
        win = [0, 0]
        warm_reset = False
        for i in range(n):
            t = W["t"][i]; self.now = t
            while t >= next_slot:
                self.end_slot(); next_slot += 1.0
                if collect and int(next_slot) % 50 == 0:
                    out["taus"].append(float(np.mean(self.tau_g)))
                    out["tseries"].append((next_slot, win[0], win[1])); win = [0, 0]
            if not warm_reset and t >= t_warm:
                self.ZR = 0.0; warm_reset = True                       # end of cold start
            c, q, user, att = int(W["cell"][i]), int(W["pid"][i]), int(W["user"][i]), bool(W["att"][i])
            g = int(A.group[q])
            utr = self.utrust(user, att)
            best, prob = self.candidates(c, q)
            hit = False
            if best is not None:
                s, e, where = best
                key = (c, e) if where == "L" else (self.holder(c, e), e)
                tau = self.threshold(q, key)
                hit = s >= tau
            measured = (not att) and t >= t_warm
            lat = cfg["radio_lat"] + cfg["lookup_lat"]; en = cfg["e_radio"] + cfg["e_lookup"]
            if hit:
                valid = A.intent[q] == A.intent[e]
                if where == "L":
                    self.lru[c].move_to_end(e)
                else:
                    lat += cfg["remote_lat"]; en += cfg["e_remote"]
                    hc = key[0]
                    if hc is not None:
                        self.lru[hc].move_to_end(e)
                if f["verify"]:
                    if self.rng.random() < self.pv:
                        _, _, e_v, _ = service_times(A, cfg, q); en += e_v   # low-priority edge re-generation
                        jv = self.judge(valid)
                        self.record(c, g, s, jv, 1.0 / self.pv, True)
                        if not jv:
                            self.slot_R += 1.0 / self.pv
                        if f["probation"] and key[0] is not None:
                            self.update_entry(key[0], e, jv, user, utr)
                        if measured:
                            out["verif"] += 1; out["llm_calls"] += 1
                    else:
                        self.record(c, g, s, valid, 0.0, False)
                if measured:
                    out["hit_valid" if valid else "hit_invalid"] += 1
                    if e >= A.n_benign:
                        out["poison"] += 1
                        hc_ = c if where == "L" else key[0]
                        self.adv["served"] += 1
                        self.adv["cells"].setdefault((hc_, e), set()).add(c)
                    if where == "R":
                        out["remote_hits"] += 1
                    out["grp_n"][g] += 1; out["grp_fr"][g] += 0 if valid else 1
                    win[0] += 1; win[1] += 0 if valid else 1
            else:
                d, e_l, tier = self.route(c, t, q)
                lat += d; en += e_l
                cand_valid = None
                if best is not None:
                    s, e, where = best
                    cand_valid = A.intent[q] == A.intent[e]
                    if f["thr"] == "pe" and key[0] is not None:
                        self.pe_observe(key, s, bool(cand_valid))
                if f["verify"]:
                    self.record(c, g, best[0] if best else None, self.judge(bool(cand_valid)) if best else True, 1.0, True)
                if f["probation"] and prob is not None:
                    tau_p = self.threshold(q, None)
                    if prob[0] >= tau_p:
                        self.update_entry(c, prob[1], self.judge(bool(A.intent[q] == A.intent[prob[1]])), user, utr)
                self.insert(c, q, user, utr)
                if measured:
                    out["miss"] += 1; out["llm_calls"] += 1
                    out["cloud"] += tier == "cloud"
                    out["grp_n"][g] += 1
                    win[0] += 1
            self.slot_Ntot += 1
            self.user_n[user] = self.user_n.get(user, 0) + 1
            if measured:
                out["n"] += 1; out["lat"].append(lat); out["energy"] += en
                if A.intent[q] in W["targets"]:
                    out["tgt_req"] += 1
                    if hit and e >= A.n_benign:
                        out["tgt_poison"] += 1
        return self.summarize(out)

    def summarize(self, o):
        n = max(o["n"], 1); lat = np.array(o["lat"])
        res = dict(
            n=o["n"], valid_hit=o["hit_valid"] / n, frr=o["hit_invalid"] / n,
            poison_rate=o["poison"] / n, asr=o["tgt_poison"] / max(o["tgt_req"], 1),
            tgt_req=o["tgt_req"], mean_lat=float(lat.mean()), p95_lat=float(np.percentile(lat, 95)),
            p99_lat=float(np.percentile(lat, 99)), energy=o["energy"] / n,
            llm_per_req=o["llm_calls"] / n, cloud_share=o["cloud"] / max(o["miss"], 1),
            remote_share=o["remote_hits"] / max(o["hit_valid"] + o["hit_invalid"], 1),
            verif_per_req=o["verif"] / n, mean_tau=float(np.mean(o["taus"])) if o["taus"] else float("nan"),
            grp_n=o["grp_n"].tolist(), grp_fr=o["grp_fr"].tolist(), tseries=o["tseries"], taus_series=o["taus"],
            lat_sample=np.sort(lat)[:: max(1, len(lat) // 400)].tolist(),
            adv_occupancy=float(self.present[:, self.A.n_benign:].sum() / (self.cfg["n_cells"] * self.cfg["cache_cap"])),
            adv_inserted=self.adv["inserted"], adv_activated=self.adv["activated"],
            adv_removed_trust=sum(1 for _, r in self.adv["life"] if r == "trust"),
            adv_served=self.adv["served"],
            adv_cells_mean=float(np.mean([len(v) for v in self.adv["cells"].values()])) if self.adv["cells"] else 0.0,
            adv_cells_max=int(max([len(v) for v in self.adv["cells"].values()])) if self.adv["cells"] else 0,
            adv_entries_served=len(self.adv["cells"]),
            adv_life_med=float(np.median([l for l, _ in self.adv["life"]])) if self.adv["life"] else float("nan"))
        return res


# ------------------------------------------------------------------ helpers
def offline_crc_threshold(A, cfg, seed):
    """Conventional split-CRC calibration on a clean episode (no attack): every request is
    answered by the LLM, and the validity of its nearest cooperative candidate is labelled."""
    c2 = dict(cfg); c2["req_per_cell"] = max(200, cfg["calib_requests"] // cfg["n_cells"])
    W = make_workload(A, c2, seed + 50_000, attack=False)
    sim = Sim(A, c2, method_flags("Static-C"), seed)
    S, V = [], []
    for i in range(len(W["t"])):
        c, q = int(W["cell"][i]), int(W["pid"][i])
        best, _ = sim.candidates(c, q)
        if best is not None:
            S.append(best[0]); V.append(A.intent[q] == A.intent[best[1]])
        else:
            S.append(0.0); V.append(True)
        sim.insert(c, q, 0, 1.0)
    S, V = np.array(S), np.array(V)
    n = len(S)
    for tau in GRID:
        if ((np.sum((S >= tau) & ~V) + 1.0) / (n + 1.0)) <= cfg["alpha"]:
            return float(tau)
    return 1.0


def run_one(A, method, seed, cfg_over=None, flag_over=None, attack=True):
    cfg = dict(DEFAULT_CFG)
    if cfg_over:
        cfg.update(cfg_over)
    flags = method_flags(method if method in METHODS else "CERTA", flag_over)
    W = make_workload(A, cfg, seed, attack=attack)
    crc = offline_crc_threshold(A, cfg, seed) if flags["thr"] == "crc_off" else None
    sim = Sim(A, cfg, flags, seed, crc_tau=crc)
    r = sim.run(W)
    r["crc_tau"] = crc
    return r


def paws_shift(A, alpha=0.05, tau_static=0.80, tau_crc=None, seed=0, p_v=0.05,
               gamma=0.999, n_min=50, W_R=200.0):
    """Distribution-shift study: cached key = PAWS sentence1, query = sentence2."""
    rng = np.random.default_rng(seed)
    o = rng.permutation(len(A.paws_sim))
    s, lab = A.paws_sim[o].astype(float), A.paws_lab[o].astype(bool)
    res = {}
    for name, tau in [("Static-C", tau_static), ("CRC-Off", tau_crc)]:
        hit = s >= tau
        res[name] = dict(frr=float(np.mean(hit & ~lab)), valid_hit=float(np.mean(hit & lab)), tau=tau)
    # CERTA-style online recalibration with IPW labels and risk-deficit margin
    Hl = np.zeros(NBIN); Hn = 0.0; ZR = 0.0; tau = 1.0
    hits = inval = 0; taus = []; pv = p_v
    B = 1.0 / p_v
    for i in range(len(s)):
        h = s[i] >= tau
        if h:
            hits += 1; inval += int(not lab[i])
            if rng.random() < pv:
                if not lab[i]:
                    Hl[s2bin(s[i])] += 1.0 / pv; ZR += 1.0 / pv
        else:
            if not lab[i]:
                Hl[s2bin(s[i])] += 1.0
        Hn += 1.0
        ZR = max(ZR - alpha, 0.0)
        if i % 20 == 19:
            cum = np.flip(np.cumsum(np.flip(Hl)))
            eta = min(0.5 * alpha, ZR / W_R)
            bound = (Hn * cum / max(Hn, 1) + B) / (Hn + 1)
            ok = np.where(bound <= alpha - eta)[0]
            tau = GRID[ok.min()] if (len(ok) and Hn >= n_min) else 1.0
            Hl *= gamma ** 20; Hn *= gamma ** 20
            taus.append(tau)
    n = len(s)
    res["CERTA"] = dict(frr=inval / n, valid_hit=(hits - inval) / n, tau=float(np.mean(taus[len(taus)//4:])))
    return res
