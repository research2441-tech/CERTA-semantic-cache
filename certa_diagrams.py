"""
certa_diagrams.py  -  grayscale schematic figures (Fig. 1 architecture, Fig. 2 request pipeline).
Every arrow is drawn between explicit anchor points so that directions can be audited.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "mathtext.fontset": "stix"})


def box(ax, x, y, w, h, text, fc="white", lw=0.9, fs=7.5, ls="-", bold=False, ec="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
                       fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y),
                x=x, y=y, w=w, h=h)


def arrow(ax, p, q, ls="-", lw=0.9, style="-|>", ms=7, both=False, color="black", rad=0.0):
    a = FancyArrowPatch(p, q, arrowstyle=("<|-|>" if both else style), mutation_scale=ms,
                        lw=lw, ls=ls, color=color, shrinkA=0, shrinkB=0,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def poly_arrow(ax, pts, ls="-", lw=0.9, ms=7):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color="black", lw=lw, ls=ls,
                solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], ls=ls, lw=lw, ms=ms)


def label(ax, x, y, t, fs=6.8, ha="center", style="italic", bg=True):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, style=style,
            bbox=dict(fc="white", ec="none", pad=0.6) if bg else None)


# ----------------------------------------------------------------------------- Fig. 1
def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(7.16, 3.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    # top tier
    cloud = box(ax, 4, 47, 38, 7, "Cloud LLM tier\n(large model, WAN round trip, elastic capacity)", fc="#e6e6e6")
    aggr = box(ax, 56, 47, 40, 7, "Federated risk calibration\n(trimmed aggregation  +  risk-deficit margin)",
               fc="#e6e6e6")
    # cells
    cells = []
    xs = [2, 35, 68]
    names = ["Edge cell 1", "Edge cell 2", "Edge cell C"]
    for x, nm in zip(xs, names):
        outer = FancyBboxPatch((x, 15), 30, 23, boxstyle="round,pad=0.3,rounding_size=1.2",
                               fc="#f5f5f5", ec="black", lw=1.0)
        ax.add_patch(outer)
        ax.text(x + 15, 36.3, nm, ha="center", va="center", fontsize=7.5, fontweight="bold")
        c1 = box(ax, x + 2, 26.5, 26, 7.3, "Semantic cache\n+ trust ledger (probation)", fs=7)
        c2 = box(ax, x + 2, 17.2, 26, 6.6, "Edge LLM\n(small model, 12 streams)", fs=7)
        cells.append(dict(x=x, cache=c1, llm=c2))
    ax.text(66.5, 26.5, "...", ha="center", va="center", fontsize=11)
    # users and attacker
    for x in xs:
        for k in range(4):
            ax.add_patch(Circle((x + 7 + 5.5 * k, 6.5), 1.4, fc="white", ec="black", lw=0.8))
        arrow(ax, (x + 15, 9.0), (x + 15, 14.7))
    ax.add_patch(Polygon([[36.0, 1.2], [39.4, 1.2], [37.7, 4.0]], closed=True, fc="black"))
    label(ax, 47.5, 2.6, "sybil accounts (attacker)", fs=6.5, bg=False)
    label(ax, 17, 11.7, "queries", fs=6.5)
    # cooperative lookup between neighbouring cells (bidirectional)
    arrow(ax, (32.3, 30.0), (34.7, 30.0), both=True, ms=6)
    arrow(ax, (65.3, 22.0), (67.7, 22.0), both=True, ms=6)
    # misses offloaded to the cloud (solid, upward)
    arrow(ax, (12, 38.4), (12, 46.6))
    arrow(ax, (40, 38.4), (34, 46.6))
    label(ax, 23, 43.0, "misses offloaded", fs=6.5)
    # risk histograms up to aggregator (dashed), threshold back down (dashed)
    arrow(ax, (58, 38.4), (64, 46.6), ls="--")
    arrow(ax, (76, 38.4), (76, 46.6), ls="--")
    arrow(ax, (90, 46.6), (90, 38.4), ls="--")
    label(ax, 70, 43.0, "risk histograms", fs=6.5)
    label(ax, 92.5, 43.0, r"$\hat{\tau}_t$", fs=8, style="normal")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=400); plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 2
def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(7.16, 4.5))
    ax.set_xlim(0, 101); ax.set_ylim(0, 66); ax.axis("off")
    W, H = 15, 6.4
    q = box(ax, 2, 52, W, H, "Query $q$\nfrom user $j$")
    e = box(ax, 22, 52, W, H, "Embedding\n$\\mathbf{e}_q$ (MiniLM)")
    c = box(ax, 42, 52, W + 2, H, "Nearest eligible entry\n$k^\\star$, similarity $s^\\star$")
    # decision diamond
    dx, dy, dw, dh = 72, 55.2, 9.5, 5.6
    ax.add_patch(Polygon([[dx - dw, dy], [dx, dy + dh], [dx + dw, dy], [dx, dy - dh]], closed=True,
                         fc="#e6e6e6", ec="black", lw=0.9))
    ax.text(dx, dy, "$s^\\star \\geq \\hat{\\tau}_t$ ?", ha="center", va="center", fontsize=7.5)
    arrow(ax, q["r"], e["l"]); arrow(ax, e["r"], c["l"]); arrow(ax, c["r"], (dx - dw, dy))
    # hit branch (right)
    hit = box(ax, 85, 38, 13, H, "Serve cached\nanswer")
    poly_arrow(ax, [(dx + dw, dy), (91.5, dy), (91.5, hit["y"] + H + 0.3)])
    label(ax, 88.0, dy + 1.8, "yes", fs=6.8)
    ver = box(ax, 85, 25, 13, H, "Shadow check\nw.p. $p$", ls="--")
    arrow(ax, hit["b"], (91.5, ver["y"] + H + 0.3), ls="--")
    # miss branch (down)
    tier = box(ax, 64.5, 38, 15, H, "Tier selection\n(edge or cloud)")
    arrow(ax, (dx, dy - dh), (dx, tier["y"] + H + 0.3))
    label(ax, dx + 2.6, 45.9, "no", fs=6.8)
    fresh = box(ax, 64.5, 25, 15, H, "Fresh LLM answer\nreturned to user")
    arrow(ax, tier["b"], (dx, fresh["y"] + H + 0.3))
    comp = box(ax, 40, 25, 18, H, "Compare with $k^\\star$ and\nprobation entry")
    arrow(ax, fresh["l"], comp["r"])
    ins = box(ax, 19, 25, 15, H, "Insert entry\n(probation state)")
    arrow(ax, comp["l"], ins["r"])
    trust = box(ax, 2, 25, 13, H, "Trust ledger\n$T_k$, $u_j$")
    arrow(ax, ins["l"], trust["r"])
    # labels into risk histogram
    hist = box(ax, 40, 11, 18, H, "Decayed IPW risk\nhistogram of cell $c$")
    arrow(ax, comp["b"], (49, hist["y"] + H + 0.3))
    label(ax, 52.6, 21.5, "label, $w=1$", fs=6.5, style="normal")
    poly_arrow(ax, [(91.5, ver["y"]), (91.5, 14.2), (hist["x"] + hist["w"] + 0.3, 14.2)], ls="--")
    label(ax, 78, 15.8, "label, $w=1/p$", fs=6.5, style="normal")
    # probation entry state change driven by the comparison
    arrow(ax, (comp["x"] + 2, comp["y"] + H), (trust["x"] + 9, trust["y"] + H + 0.2), rad=0.35, ls="-")
    label(ax, 22, 35.6, "confirm / dispute", fs=6.5)
    # aggregation and threshold feedback
    agg = box(ax, 13, 2.2, 22, H, "Trimmed aggregation over\ncells + margin $\\eta_t$")
    poly_arrow(ax, [(hist["x"], 14.2), (24, 14.2), (24, agg["y"] + H + 0.3)])
    label(ax, 31.5, 15.8, "to peers", fs=6.5)
    thr = box(ax, 85, 2.2, 13, H, "Threshold $\\hat{\\tau}_t$")
    arrow(ax, agg["r"], thr["l"], ls="--")
    poly_arrow(ax, [(thr["x"] + thr["w"] + 0.3, 5.4), (100.2, 5.4), (100.2, 64.0), (dx, 64.0),
                    (dx, dy + dh + 0.2)], ls="--")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=400); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    fig_architecture("figs/fig1_architecture.png")
    fig_pipeline("figs/fig2_pipeline.png")
