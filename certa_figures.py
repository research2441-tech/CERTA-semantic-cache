"""
certa_figures.py - grayscale result figures generated from results/results.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7, "lines.linewidth": 1.1,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5, "legend.frameon": True,
    "legend.edgecolor": "0.6", "legend.framealpha": 1.0, "savefig.dpi": 400})

STY = {  # method -> (color, marker, linestyle)
    "CERTA": ("black", "o", "-"), "Static-C": ("0.35", "s", "--"), "CRC-Off": ("0.55", "^", "-."),
    "PE-Online": ("0.2", "D", ":"), "Exact": ("0.6", "v", "-"), "Static-L": ("0.45", "<", "--"),
    "CERTA-noprob": ("0.5", "P", (0, (5, 1, 1, 1))),
}
LBL = {"CERTA-noprob": "CERTA w/o probation"}


def m(d, k):
    return d[k]["mean"], d[k]["sd"]


def legend_top(ax, ncol, y=1.02, **kw):
    return ax.legend(loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, borderaxespad=0.3,
                     columnspacing=1.0, handlelength=2.2, **kw)


def fig_risk_control(R, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.05))
    al = [0.01, 0.02, 0.05, 0.10]
    for meth in ("CERTA", "CRC-Off", "PE-Online"):
        c, mk, ls = STY[meth]
        fr = [m(R["E8"][meth][str(a)], "frr") for a in al]
        vh = [m(R["E8"][meth][str(a)], "valid_hit") for a in al]
        axs[0].errorbar(al, [x[0] for x in fr], yerr=[x[1] for x in fr], color=c, marker=mk, ls=ls, ms=3.5,
                        capsize=2, label=meth)
        axs[1].errorbar(al, [x[0] for x in vh], yerr=[x[1] for x in vh], color=c, marker=mk, ls=ls, ms=3.5,
                        capsize=2)
    axs[0].plot([0, 0.105], [0, 0.105], color="0.6", lw=0.8, ls=(0, (2, 2)), label=r"$y=\alpha$")
    axs[0].set_xlabel(r"target risk $\alpha$"); axs[0].set_ylabel("realised false-reuse rate")
    axs[1].set_xlabel(r"target risk $\alpha$"); axs[1].set_ylabel("valid hit rate")
    axs[0].set_title("(a) risk control", pad=2); axs[1].set_title("(b) utility", pad=2)
    axs[1].set_ylim(0.4, 0.9)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.2)
    fig.savefig(path); plt.close(fig)


def fig_frontier(R, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.15))
    ss = R["E1_static_sweep"]
    ts = sorted(ss, key=float)
    x = [ss[t]["frr"]["mean"] for t in ts]; y = [ss[t]["valid_hit"]["mean"] for t in ts]
    a = [ss[t]["asr"]["mean"] for t in ts]
    for ax, xv in ((axs[0], x), (axs[1], a)):
        ax.plot(xv, y, color="0.4", ls="--", marker="s", ms=3, label=r"Static-C, $\tau$ swept")
    for t, xi, yi in zip(ts, x, y):
        if t in ("0.8",):
            axs[0].annotate(r"$\tau$=" + t, (xi, yi), textcoords="offset points",
                            xytext=((-8, -11) if t == "0.8" else (-3, -11)), fontsize=5.8, color="0.3")
    pts = [("CERTA", R["E1"]["CERTA"]), ("CRC-Off", R["E1"]["CRC-Off"]), ("PE-Online", R["E1"]["PE-Online"]),
           ("Exact", R["E1"]["Exact"]), ("Static-L", R["E1"]["Static-L"]),
           ("CERTA-noprob", R["E2"]["w/o probation trust"])]
    for nm, d in pts:
        c, mk, _ = STY[nm]
        for ax, k in ((axs[0], "frr"), (axs[1], "asr")):
            ax.errorbar(d[k]["mean"], d["valid_hit"]["mean"], xerr=d[k]["sd"], yerr=d["valid_hit"]["sd"],
                        color=c, marker=mk, ms=4.5 if nm == "CERTA" else 3.5, ls="none", capsize=1.5,
                        mfc=("black" if nm == "CERTA" else "white"), label=LBL.get(nm, nm))
    axs[0].axvline(0.05, color="0.6", lw=0.8, ls=":")
    axs[0].text(0.052, 0.47, r"$\alpha$", fontsize=7, color="0.3")
    axs[0].set_xlabel("false-reuse rate"); axs[0].set_ylabel("valid hit rate")
    axs[1].set_xlabel("attack success rate")
    axs[0].set_title("(a) risk-utility plane", pad=2); axs[1].set_title("(b) poisoning-utility plane", pad=2)
    for ax in axs:
        ax.set_ylim(0.45, 0.88)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995), columnspacing=0.6,
               handlelength=1.4, fontsize=6.0)
    fig.tight_layout(rect=(0, 0, 1, 0.78), w_pad=1.0)
    fig.savefig(path); plt.close(fig)


def fig_attack(R, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.05))
    rates = ["0.0", "0.02", "0.05", "0.1", "0.2"]; xr = [float(r) * 100 for r in rates]
    for meth in ("Static-C", "CRC-Off", "PE-Online", "CERTA-noprob", "CERTA"):
        c, mk, ls = STY[meth]
        d = R["E4_attack"][meth]
        axs[0].errorbar(xr, [d[r]["asr"]["mean"] for r in rates], yerr=[d[r]["asr"]["sd"] for r in rates],
                        color=c, marker=mk, ls=ls, ms=3, capsize=1.5, label=LBL.get(meth, meth))
        axs[1].errorbar(xr, [d[r]["frr"]["mean"] for r in rates], yerr=[d[r]["frr"]["sd"] for r in rates],
                        color=c, marker=mk, ls=ls, ms=3, capsize=1.5)
    axs[1].axhline(0.05, color="0.6", lw=0.8, ls=":")
    for ax in axs:
        ax.set_xlabel("attack traffic (%)")
    axs[0].set_ylabel("attack success rate"); axs[1].set_ylabel("false-reuse rate")
    axs[0].set_title("(a) poisoning", pad=2); axs[1].set_title("(b) benign false reuse", pad=2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=1.2)
    fig.savefig(path); plt.close(fig)


def fig_byz_sybil(R, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.45))
    fs = ["0", "1", "2"]
    for nm, c, mk, ls, lab in (("trimmed", "black", "o", "-", "trimmed (CERTA)"), ("mean", "0.5", "s", "--", "plain mean")):
        d = R["E4_byz"][nm]
        axs[0].errorbar([int(f) for f in fs], [d[f]["frr"]["mean"] for f in fs], yerr=[d[f]["frr"]["sd"] for f in fs],
                        color=c, marker=mk, ls=ls, ms=3.5, capsize=1.5, label=lab)
    axs[0].axhline(0.05, color="0.6", lw=0.8, ls=":")
    axs[0].set_xticks([0, 1, 2]); axs[0].set_xlabel("forged-report cells $f$\n(a)")
    axs[0].set_ylabel("false-reuse rate")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=6.2, handlelength=1.8)
    sy = ["2", "4", "6", "8", "12"]
    for tr, c, mk, ls, lab in (("None", "black", "o", "-", "fresh sybils"), ("0.5", "0.4", "^", "--", r"aged, $u=0.5$"),
                               ("0.9", "0.6", "s", "-.", r"aged, $u=0.9$")):
        d = R["E9_sybil"][tr]
        axs[1].errorbar([int(s_) for s_ in sy], [d[s_]["asr"]["mean"] for s_ in sy], yerr=[d[s_]["asr"]["sd"] for s_ in sy],
                        color=c, marker=mk, ls=ls, ms=3.2, capsize=1.5, label=lab)
    axs[1].set_xlabel("sybil requests / campaign\n(b)"); axs[1].set_ylabel("attack success rate")
    axs[1].set_ylim(-0.02, 0.32)
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.0, handlelength=1.8,
                  columnspacing=0.6)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(path); plt.close(fig)


def fig_latency(R, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for meth in ("Static-C", "CERTA"):
        c, _, ls = STY[meth]
        v = np.sort(np.array(R["E1"][meth]["lat_sample"]))
        ax.plot(v, np.linspace(0, 1, len(v)), color=c, ls=ls, lw=1.3, label=meth)
    ax.set_xscale("log"); ax.set_xlabel("end-to-end response time (s)"); ax.set_ylabel("empirical CDF")
    ax.axvline(0.03, color="0.7", lw=0.6, ls=":")
    ax.text(0.033, 0.06, "cache hits", fontsize=6.3, color="0.3")
    ax.text(1.6, 0.06, "LLM-served misses", fontsize=6.3, color="0.3")
    ax.legend(loc="center left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def fig_timeseries(R, path):
    fig, ax1 = plt.subplots(figsize=(3.5, 2.0))
    ts = R["E1"]["CERTA"]["tseries"]
    t = np.array([x[0] for x in ts]); n = np.array([x[1] for x in ts]); f = np.array([x[2] for x in ts])
    fr = np.where(n > 0, f / np.maximum(n, 1), np.nan)
    ts2 = R["E1"]["Static-C"]["tseries"]
    n2 = np.array([x[1] for x in ts2]); f2 = np.array([x[2] for x in ts2]); t2 = np.array([x[0] for x in ts2])
    ax1.plot(t2, np.where(n2 > 0, f2 / np.maximum(n2, 1), np.nan), color="0.55", ls="--", marker="s", ms=2.5,
             label="Static-C false reuse")
    ax1.plot(t, fr, color="black", marker="o", ms=2.5, label="CERTA false reuse")
    ax1.axhline(0.05, color="0.6", lw=0.8, ls=":")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("windowed false-reuse rate")
    ax1.set_ylim(0, 0.16)
    lo = R["cfg"]["warmup_frac"] * t.max()
    ax1.axvspan(0, lo, color="0.92", lw=0)
    ax1.text(lo * 0.5, 0.148, "cold start", ha="center", fontsize=6.3, color="0.3")
    ax2 = ax1.twinx()
    taus = R["E1"]["CERTA"]["taus_series"] if "taus_series" in R["E1"]["CERTA"] else None
    if taus:
        ax2.step(t[:len(taus)], taus[:len(t)], where="post", color="0.3", ls="-.", lw=1.0, label=r"threshold $\hat{\tau}_t$")
        ax2.set_ylabel(r"threshold $\hat{\tau}_t$"); ax2.set_ylim(0.75, 1.0); ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=6.0,
               handlelength=1.8, columnspacing=0.7)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def fig_sensitivity(R, path):
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.7))
    th = ["0.4", "0.6", "0.8", "1.0", "1.2"]; xv = [float(t) for t in th]
    d = R["E3"]["th_act"]
    a = axs[0]; a2 = a.twinx()
    a.errorbar(xv, [d[t]["valid_hit"]["mean"] for t in th], yerr=[d[t]["valid_hit"]["sd"] for t in th], color="black",
               marker="o", ms=3, capsize=1.5, label="valid hit rate (left)")
    a2.errorbar(xv, [d[t]["asr"]["mean"] for t in th], yerr=[d[t]["asr"]["sd"] for t in th], color="0.5", ls="--",
                marker="^", ms=3, capsize=1.5, label="attack success rate (right)")
    a.set_xlabel(r"activation threshold $\theta$ (8 sybil requests per campaign)    (a)")
    a.set_ylabel("valid hit rate"); a2.set_ylabel("attack success rate"); a2.grid(False)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.2, handlelength=1.8)
    pv = ["0.02", "0.05", "0.1", "0.2"]; xp = [float(p) for p in pv]
    d = R["E3"]["p_v"]
    b = axs[1]; b2 = b.twinx()
    b.errorbar(xp, [d[p]["frr"]["mean"] for p in pv], yerr=[d[p]["frr"]["sd"] for p in pv], color="black",
               marker="o", ms=3, capsize=1.5, label="false-reuse rate (left)")
    b2.errorbar(xp, [d[p]["energy"]["mean"] for p in pv], yerr=[d[p]["energy"]["sd"] for p in pv], color="0.5",
                ls="--", marker="s", ms=3, capsize=1.5, label="energy per request (right)")
    b.set_xlabel("shadow-verification probability $p$    (b)"); b.set_ylabel("false-reuse rate")
    b2.set_ylabel("energy (J)"); b.set_ylim(0, 0.05); b2.grid(False)
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.2, handlelength=1.8)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(path); plt.close(fig)


def fig_paws(R, path):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.1))
    names = ["Static 0.80", "Static 0.92", "Static 0.99", "CRC-Off", "CERTA"]
    hatches = ["", "////", "xxxx", "....", ""]
    fcs = ["0.85", "0.7", "white", "white", "black"]
    for j, a in enumerate(["0.05", "0.1"]):
        P = R["E5_paws"][a]; S = R["E5_paws_static"]
        fr = [S["0.8"]["frr"], S["0.92"]["frr"], S["0.99"]["frr"], P["CRC-Off"]["frr"], P["CERTA"]["frr"]]
        vh = [S["0.8"]["valid_hit"], S["0.92"]["valid_hit"], S["0.99"]["valid_hit"], P["CRC-Off"]["valid_hit"],
              P["CERTA"]["valid_hit"]]
        for i in range(5):
            axs[j].bar(i - 0.2, fr[i], 0.38, color=fcs[i], edgecolor="black", hatch=hatches[i], lw=0.6)
            axs[j].bar(i + 0.2, vh[i], 0.38, color="white", edgecolor="0.4", lw=0.6, ls="--")
        axs[j].axhline(float(a), color="0.4", lw=0.8, ls=":")
        axs[j].text(4, max(fr[4], vh[4], float(a)) + 0.03, f"{fr[4]:.3f}\n{vh[4]:.3f}", ha="center", va="bottom", fontsize=5.5)
        axs[j].set_xticks(range(5)); axs[j].set_xticklabels(names, rotation=40, ha="right", fontsize=6)
        axs[j].set_title(rf"$\alpha={a}$", pad=2); axs[j].set_ylim(0, 0.62)
    axs[0].set_ylabel("rate on PAWS pairs")
    from matplotlib.patches import Patch
    hd = [Patch(fc="0.7", ec="black", label="false reuse (solid edge)"),
          Patch(fc="white", ec="0.4", ls="--", label="valid hit (dashed edge)")]
    fig.legend(handles=hd, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
    fig.savefig(path); plt.close(fig)


def fig_worstgroup(R, path):
    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    data = [R["E1"]["Static-C"]["worst_group"], R["E2"]["w/o risk-deficit margin"]["worst_group"],
            R["E2"]["CERTA (full)"]["worst_group"], R["E2"]["+ group-conditional thresholds"]["worst_group"]]
    labs = ["Static-C", "CERTA w/o\nmargin", "CERTA", "CERTA +\ngroup thresholds"]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops=dict(color="black", lw=1.1),
                    flierprops=dict(marker="o", ms=2.5, mfc="white", mec="0.3"))
    for p, fc in zip(bp["boxes"], ["0.8", "0.6", "white", "0.9"]):
        p.set_facecolor(fc); p.set_edgecolor("black"); p.set_linewidth(0.7)
    ax.axhline(0.05, color="0.4", lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_ylabel("worst-group false reuse")
    ax.set_xticklabels(labs, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def make_all(R, out="figs"):
    os.makedirs(out, exist_ok=True)
    fig_risk_control(R, f"{out}/fig3_risk_control.png")
    fig_frontier(R, f"{out}/fig4_frontier.png")
    fig_attack(R, f"{out}/fig5_attack.png")
    fig_byz_sybil(R, f"{out}/fig6_byz_sybil.png")
    fig_latency(R, f"{out}/fig7_latency.png")
    fig_timeseries(R, f"{out}/fig8_timeseries.png")
    fig_sensitivity(R, f"{out}/fig9_sensitivity.png")
    fig_paws(R, f"{out}/fig10_paws.png")
    fig_worstgroup(R, f"{out}/fig11_worstgroup.png")
    R["simstats"] = fig_similarity(f"{out}/fig12_similarity.png")




def fig_similarity(path, assets="certa_assets.npz"):
    z = np.load(assets)
    nb, ns, it = z["nb_idx"], z["nb_sim"], z["intent"]
    nbn = int(z["n_benign"])
    s_ok, s_bad = [], []
    for p in range(nbn):
        for j in range(1, nb.shape[1]):
            q = nb[p, j]
            if q < nbn and q != p:
                (s_ok if it[q] == it[p] else s_bad).append(ns[p, j]); break
    adv = z["adv_tgt_sim"]; ps, pl = z["paws_sim"], z["paws_lab"].astype(bool)
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.1), sharey=False)
    bins = np.linspace(0.3, 1.0, 36)
    axs[0].hist(s_bad, bins=bins, density=True, histtype="stepfilled", color="0.82", ec="0.4", lw=0.6,
                label="nearest, other intent")
    axs[0].hist(s_ok, bins=bins, density=True, histtype="step", color="black", lw=1.1, label="nearest, same intent")
    axs[0].hist(adv, bins=bins, density=True, histtype="step", color="0.35", lw=1.0, ls="--",
                label="adversarial key to target")
    axs[0].set_xlabel("cosine similarity\n(a) QQP pool"); axs[0].set_ylabel("density")
    axs[1].hist(ps[~pl], bins=bins, density=True, histtype="stepfilled", color="0.82", ec="0.4", lw=0.6,
                label="non-paraphrase")
    axs[1].hist(ps[pl], bins=bins, density=True, histtype="step", color="black", lw=1.1, label="paraphrase")
    axs[1].set_xlabel("cosine similarity\n(b) PAWS pairs")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=5.8, handlelength=1.6)
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=5.8, handlelength=1.6)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(path); plt.close(fig)
    return dict(ok_med=float(np.median(s_ok)), bad_med=float(np.median(s_bad)), n_ok=len(s_ok), n_bad=len(s_bad),
                bad_above_090=float(np.mean(np.array(s_bad) >= 0.9)), ok_above_090=float(np.mean(np.array(s_ok) >= 0.9)),
                bad_above_080=float(np.mean(np.array(s_bad) >= 0.8)), ok_above_080=float(np.mean(np.array(s_ok) >= 0.8)),
                paws_neg_above_095=float(np.mean(ps[~pl] >= 0.95)), paws_pos_above_095=float(np.mean(ps[pl] >= 0.95)),
                adv_above_090=float(np.mean(adv >= 0.9)))


if __name__ == "__main__":
    R = json.load(open("results/results.json")); make_all(R); json.dump(R, open("results/results.json", "w"))
