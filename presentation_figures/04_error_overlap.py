"""Fig 4: which of the 91 real examples each system gets wrong, and where they overlap.

Uses the SAME XGBoost run as the model-comparison figure (real-only, 5-fold CV, seed 42;
data/annotations/dataset_consolidated_xgboost_oof_predictions.csv) and the zero-shot LLM
predictions. The mid-report's section 8.4 counts (48/9/29/10) were computed from a DIFFERENT
XGBoost run (trained on real + synthetic, dataset_final_xgboost_oof_predictions.csv, filtered to
the 91 real rows); both versions are recomputed from the CSVs and saved in
derived/error_overlap_counts.csv. This figure plots the real-only run only.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import *

style()
df = load_real_predictions()
df["xgb_wrong"] = df.xgb != df.gold
df["llm_wrong"] = df.llm != df.gold
df["category"] = "both correct"
df.loc[df.xgb_wrong & df.llm_wrong, "category"] = "both wrong"
df.loc[df.xgb_wrong & ~df.llm_wrong, "category"] = "XGBoost only wrong"
df.loc[~df.xgb_wrong & df.llm_wrong, "category"] = "LLM only wrong"
df[["pair_id", "type_candidate", "label", "gold", "xgb", "llm", "category"]].to_csv(
    DER / "error_overlap_per_example.csv", index=False)

# report 8.4 version (XGBoost from the real+synthetic run), recomputed from file for the record
saved = pd.read_csv(ANNOT / "dataset_final_xgboost_oof_predictions.csv")[["pair_id", "pred_y"]]
rep = df.merge(saved, on="pair_id")
assert len(rep) == 91
rx, rl = rep.pred_y != rep.gold, rep.llm != rep.gold


def counts(xw, lw):
    return dict(any_wrong=int((xw | lw).sum()), both_wrong=int((xw & lw).sum()),
                xgb_only_wrong=int((xw & ~lw).sum()), llm_only_wrong=int((lw & ~xw).sum()))


cnt = pd.DataFrame([dict(xgb_run="real_only (this figure, = Table 3 run)", **counts(df.xgb_wrong, df.llm_wrong)),
                    dict(xgb_run="real+synthetic run (mid-report sec 8.4)", **counts(rx, rl))])
cnt.to_csv(DER / "error_overlap_counts.csv", index=False)

ORDER = [("both wrong", INK2, "white"), ("XGBoost only wrong", C_XGB, "white"),
         ("LLM only wrong", C_LLM, "white"), ("both correct", GRID, INK2)]
GROUPS = [("Motte-and-bailey-shaped", df[df.type_candidate == "motte_bailey"]),
          ("Strawman-shaped", df[df.type_candidate == "strawman"]),
          ("All real examples", df)]

fig, ax = plt.subplots(figsize=(13, 6.0))
fig.subplots_adjust(left=0.22, right=0.985, top=0.70, bottom=0.27)
for gi, (name, sub) in enumerate(GROUPS):
    y = len(GROUPS) - 1 - gi
    left = 0
    for cat, color, tcolor in ORDER:
        k = int((sub.category == cat).sum())
        ax.barh(y, k, 0.56, left=left, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        if k:
            ax.text(left + k / 2, y, str(k), ha="center", va="center", fontsize=16, fontweight="bold", color=tcolor, zorder=5)
        left += k
    ax.text(left + 0.8, y, f"n={len(sub)}", va="center", fontsize=13, color=INK2)
ax.set_yticks(range(len(GROUPS)))
ax.set_yticklabels([g for g, _ in GROUPS][::-1])
ax.set_xlim(0, 100)
ax.set_xlabel("number of examples")
ax.tick_params(length=0)
clean_axis(ax, "x")
fig.legend(handles=[Patch(color=c, label=l) for l, c, _ in ORDER], loc="upper left",
           bbox_to_anchor=(0.22, 0.905), ncol=2, frameon=False, columnspacing=2.0)
fig.suptitle("Where the two systems fail on the 91 real examples", x=0.02, y=0.985, ha="left", fontsize=19, fontweight="bold")

a, b = cnt.iloc[0], cnt.iloc[1]
footer(fig, [
    "XGBoost: real-only run (trained on the 91 real examples, 5-fold stratified CV, seed 42) = the Table 3 run. LLM: Claude Sonnet 5 zero-shot, single run. n = 91 (44 MB-shaped + 47 SM-shaped).",
    f"At least one wrong: {a.any_wrong}/91; both wrong {a.both_wrong}; XGBoost only {a.xgb_only_wrong}; LLM only {a.llm_only_wrong}.",
    f"Differs from mid-report sec 8.4 ({b.any_wrong}/{b.both_wrong}/{b.xgb_only_wrong}/{b.llm_only_wrong}), which used XGBoost predictions from the real + synthetic training run, restricted to the 91 real rows.",
])
save(fig, "04_error_overlap.png")
