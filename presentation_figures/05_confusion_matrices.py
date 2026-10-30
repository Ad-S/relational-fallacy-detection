"""Fig 5 (backup): confusion matrices on the 91 real examples, XGBoost (real-only run) and zero-shot LLM.

Counts are recomputed from the prediction CSVs and asserted equal to mid-report Table 4.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import *

style()
df = load_real_predictions()
rep = reported_tables()
cmap = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
MODELS = [("xgb", "XGBoost", "XGBoost"), ("llm", "Zero-shot LLM", "Zero-shot LLM")]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0))
fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.14, hspace=0.55, wspace=0.28)
vmax = 0
mats = {}
for shape, pos, name in SHAPES:
    sub = df[df.type_candidate == shape]
    for key, _, repname in MODELS:
        c = confusion(sub.gold, sub[key])
        r = rep[(rep["shape"] == shape) & (rep["model"] == repname)].iloc[0]
        assert (c["tn"], c["fp"], c["fn"], c["tp"]) == (r.tn, r.fp, r.fn, r.tp), (shape, key, c)
        mats[(shape, key)] = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        vmax = max(vmax, mats[(shape, key)].max())

for ri, (shape, pos, name) in enumerate(SHAPES):
    n = int((df.type_candidate == shape).sum())
    for ci, (key, label, _) in enumerate(MODELS):
        ax = axes[ri][ci]
        m = mats[(shape, key)]
        ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=26, fontweight="bold",
                        color="white" if m[i, j] > 0.55 * vmax else INK)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["NONE", pos], fontsize=12)
        ax.set_yticklabels(["NONE", pos], fontsize=12)
        ax.set_xlabel("predicted", fontsize=13); ax.set_ylabel("true", fontsize=13)
        ax.set_title(f"{label}\n{name} (n={n})", fontsize=14, loc="left", pad=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([-.5, .5, 1.5], minor=True); ax.set_yticks([-.5, .5, 1.5], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3); ax.tick_params(which="minor", length=0)
fig.suptitle("Confusion matrices, 91 real examples", x=0.02, y=0.985, ha="left", fontsize=19, fontweight="bold")
fig.text(0.02, 0.935, "Counts match mid-report Table 4 exactly (asserted). Rows = true label, columns = predicted.",
         fontsize=12.5, color=INK2)
footer(fig, [
    "XGBoost: real-only run, 5-fold stratified CV, seed 42, pooled held-out predictions. LLM: Claude Sonnet 5 zero-shot, single run, prompt not archived.",
])
save(fig, "05_confusion_matrices.png")
