"""Schematic overview of ConDR: cartoon value along one state axis with each loss term annotated."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
FS = 8; plt.rcParams.update({"font.size": FS, "mathtext.fontset": "cm"})
x = np.linspace(-0.3, 1.0, 400); eps = 0.06
g = x - 0.2                                   # signed distance to the failure set (g<=0 is failure)
Vt = np.minimum(g, 0.6 * (x - 0.35))          # true value: = g on the contact set, below g elsewhere
Ve = Vt - eps                                 # target of the margin-shifted VI
Vth = Ve + 0.012 * np.sin(24 * x) * np.clip((x + 0.05) * 3, 0, 1)   # trained network: near V_eps, below V_true
Vdeg = -0.55 + 0.08 * x                       # degenerate subsolution (everything unsafe)
BLUE, RED, GRAY = "#1f77b4", "#d62728", "#8c8c8c"
fig, ax = plt.subplots(figsize=(3.5, 3.1))
ax.fill_between(x, Ve, Vt, color=BLUE, alpha=0.15, lw=0)
ax.plot(x, g, "--", color=GRAY, lw=1.0)
ax.plot(x, Vt, "k-", lw=1.4)
ax.plot(x, Vth, "-", color=BLUE, lw=1.2)
ax.plot(x, Vdeg, "--", color=RED, lw=1.0)
ax.axhline(0, color="k", lw=0.5)
# boundaries
xt = 0.35; xth = x[np.argmax(Vth > 0)]
ax.plot([xt, xt], [-0.02, 0.02], "k-", lw=1); ax.plot([xth, xth], [-0.02, 0.02], "-", color=BLUE, lw=1)
ax.annotate("", xy=(xth, 0.035), xytext=(xt, 0.035), arrowprops=dict(arrowstyle="<->", lw=0.7, color=BLUE))
ax.text((xt + xth) / 2 + 0.02, 0.075, "shell $\\leq \\epsilon\\tau$ (Thm. 1)", ha="center", va="bottom", fontsize=FS - 1, color=BLUE)
ax.text(xt, -0.035, "true BRT\nboundary", ha="center", va="top", fontsize=FS - 1.5)
ax.text(xth + 0.02, -0.035, "learned\nboundary", ha="left", va="top", fontsize=FS - 1.5, color=BLUE)
# labels on curves
ax.text(0.98, 0.78, "$g(x)$", color=GRAY, ha="right", fontsize=FS)
ax.text(0.98, 0.395, "$V_{\\mathrm{true}}$", ha="right", fontsize=FS)
ax.text(0.98, 0.26, "$V_\\theta$", color=BLUE, ha="right", fontsize=FS)
ax.text(-0.28, -0.60, "degenerate $V_\\theta \\ll 0$", color=RED, fontsize=FS - 1, va="top")
# annotations for each mechanism
ax.annotate("obstacle branch by construction (15):\n$V_\\theta = g - \\tau\\,\\mathrm{softplus}(\\mathrm{NN}_\\theta) \\leq g$",
            xy=(-0.2, Vth[np.searchsorted(x, -0.2)]), xytext=(-0.28, 0.42), fontsize=FS - 1,
            arrowprops=dict(arrowstyle="->", lw=0.7), ha="left", va="bottom")
ax.annotate("$\\mathcal{L}_{\\mathrm{diff}}=\\mathbb{E}\\,\\mathrm{relu}(r_\\theta+\\epsilon)$: push $r_\\theta \\leq -\\epsilon$",
            xy=(0.93, Vth[np.searchsorted(x, 0.93)] + 0.012), xytext=(0.12, 0.64), fontsize=FS - 1,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE), ha="left", va="bottom")
ax.annotate("$\\mathcal{L}_{\\mathrm{opt}}=\\mathbb{E}\\,\\mathrm{relu}(-(r_\\theta+\\epsilon))$: hold $r_\\theta$ near $-\\epsilon$",
            xy=(0.86, Vth[np.searchsorted(x, 0.86)] - 0.012), xytext=(0.12, 0.55), fontsize=FS - 1,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=BLUE), ha="left", va="bottom")
ax.annotate("$\\mathcal{L}_{\\mathrm{deg}}=\\mathbb{E}\\min(|r_\\theta+\\epsilon|,\\,g-V_\\theta)$:\none branch tight, rules out", xy=(0.75, Vdeg[np.searchsorted(x, 0.75)]),
            xytext=(0.02, -0.30), fontsize=FS - 1, arrowprops=dict(arrowstyle="->", lw=0.7, color=RED), ha="left", va="top")
ax.text(0.75, Vdeg[np.searchsorted(x, 0.75)], "$\\times$", color=RED, fontsize=14, ha="center", va="center")
ax.set_xlim(-0.3, 1.0); ax.set_ylim(-0.68, 0.85)
ax.set_xticks([]); ax.set_yticks([0]); ax.set_yticklabels(["0"])
ax.set_xlabel("state $x$  (failure set $g \\leq 0$ on the left)", labelpad=2); ax.set_ylabel("value at time-to-go $\\tau$", labelpad=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2)
for ext in ("pdf", "png"): fig.savefig(REPO / "docs/figures" / f"fig_method_schematic.{ext}", dpi=220, bbox_inches="tight")
print("ok")
