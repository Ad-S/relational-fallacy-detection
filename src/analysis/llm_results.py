"""Open-weight LLM results: zero-shot baselines and the two-stage (extract -> NLI) pipeline.

Inputs: llm_bundle/llm_results/llm/<model>/{zeroshot.csv, stage2.csv}; fine-tuning / XGBoost predictions
(via final_results.load()); presentation_figures/llm_rerun/zeroshot_rerun_predictions.csv (Claude Opus 5.5
zero-shot on the 111 original examples of the mid-report dataset).

No training is involved for the LLM rows, so they are evaluated directly on all rows. Unparseable zero-shot
answers count as NONE (counts reported). Two-stage decision rules are fixed in advance (threshold 0.5):
  strawman:          STRAWMAN  iff B attributes a claim X to A's author AND P(A entails X) < 0.5
  motte (denial):    MB        iff B denies a claim X AND P(A entails X) >= 0.5   (the denial is false)
  motte (narrowing): MB        iff B denies/narrows AND P(A entails defended) >= 0.5 AND P(defended entails A) < 0.5
Continuous scores for ROC-AUC: flag * (1 - P(A=>X)), flag * P(A=>X), flag * P(A=>D) * (1 - P(D=>A)).

Usage: python3 src/analysis/llm_results.py   (run after final_results.py)
Writes: results_final/table8_llm_zeroshot.csv, table9_two_stage.csv, table10_paired_llm.csv; appends to RESULTS.md
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import final_results as F

LLM = F.ROOT / "llm_bundle" / "llm_results" / "llm"
MODELS = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Qwen2.5-32B-Instruct": "Qwen2.5-32B"}
POS = {"motte_bailey": "MOTTE_BAILEY", "strawman": "STRAWMAN"}


def binmetrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tp, fp, fn = ((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    out = dict(accuracy=(y == pred).mean(), precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)
    if score is not None:
        out["auc"] = roc_auc_score(y, score)
    return out


def boot_vs_ft(correct_llm, ft, shape, n=2000):
    """Paired cluster bootstrap: fine-tuned DeBERTa-NLI (seeds 0-2 averaged) minus the LLM."""
    g = ft[(ft.config == "nli_pair") & (ft["shape"] == shape) & (ft.seed.isin([0, 1, 2]))]
    c_ft = g.assign(c=(g.y == g.pred).astype(float)).groupby("pair_id").c.mean()
    ids = c_ft.index.intersection(correct_llm.index)
    d = c_ft[ids] - correct_llm[ids]
    grp = pd.Series([F.group[i] for i in ids], index=ids)
    clusters = grp.unique()
    by = {c: d[grp == c].values for c in clusters}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([by[c] for c in rng.choice(clusters, len(clusters))]).mean() for _ in range(n)]
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


def main():
    data = F.data
    ft = F.load()
    lines = ["", "## Zero-shot LLM baselines (no training; evaluated on all rows)", "",
             "| Model | Shape | n | unparsed | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|"]
    zs_rows, paired = [], []
    for folder, name in MODELS.items():
        z = pd.read_csv(LLM / folder / "zeroshot.csv", keep_default_na=False)
        assert len(z) == len(data) and set(z.pair_id) == set(data.pair_id)
        for shape, pos in POS.items():
            g = z[z.type_candidate == shape]
            y = (g.label == pos).astype(int)
            pred = (g.predicted_label == pos).astype(int)
            m = binmetrics(y, pred)
            unparsed = int((g.predicted_label == "").sum())
            zs_rows.append(dict(model=name, shape=shape, n=len(g), unparsed=unparsed, **m))
            lines.append(f"| {name} zero-shot | {F.SHAPES[shape]} | {len(g)} | {unparsed} | {m['accuracy']:.3f} | "
                         f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
            correct = pd.Series((y.values == pred.values).astype(float), index=g.pair_id.values)
            d, lo, hi = boot_vs_ft(correct, ft, shape)
            paired.append(dict(comparison=f"fine-tuned DeBERTa-NLI vs {name} zero-shot", shape=shape, diff=d, ci_low=lo, ci_high=hi))
    pd.DataFrame(zs_rows).to_csv(F.OUT / "table8_llm_zeroshot.csv", index=False)

    lines += ["", "## Fine-tuned DeBERTa-NLI minus zero-shot LLM (accuracy; cluster bootstrap 95% CI)", "",
              "| Comparison | Shape | Δ accuracy | 95% CI |", "|---|---|---|---|"]
    for r in paired:
        lines.append(f"| {r['comparison']} | {F.SHAPES[r['shape']]} | {r['diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |")
    pd.DataFrame(paired).to_csv(F.OUT / "table10_paired_llm.csv", index=False)

    # ---- the 111 originals of the mid-report dataset: add the Claude Opus 5.5 re-run ----
    opus = pd.read_csv(F.ROOT / "presentation_figures" / "llm_rerun" / "zeroshot_rerun_predictions.csv", keep_default_na=False)
    ids111 = set(opus.pair_id)
    lines += ["", "## Same comparison restricted to the 111 originals of the mid-report dataset (91 real + 20 hand-written)", "",
              "| Model | Shape | n | Accuracy | F1 |", "|---|---|---|---|---|"]
    lab = dict(zip(data.pair_id, data.label))
    shp = dict(zip(data.pair_id, data.type_candidate))
    cand = {"Claude Opus 5.5 zero-shot": dict(zip(opus.pair_id, opus.predicted_label))}
    for folder, name in MODELS.items():
        z = pd.read_csv(LLM / folder / "zeroshot.csv", keep_default_na=False)
        cand[f"{name} zero-shot"] = dict(zip(z.pair_id, z.predicted_label))
    for name, predmap in cand.items():
        for shape, pos in POS.items():
            ids = [i for i in ids111 if shp[i] == shape]
            m = binmetrics([int(lab[i] == pos) for i in ids], [int(predmap[i] == pos) for i in ids])
            lines.append(f"| {name} | {F.SHAPES[shape]} | {len(ids)} | {m['accuracy']:.3f} | {m['f1']:.3f} |")
    for cfg, name in [("nli_pair", "Fine-tuned DeBERTa-NLI (mean over seeds)"), ("xgb_NLI + Specificity", "XGBoost NLI+spec (mean over seeds)")]:
        for shape, pos in POS.items():
            g = ft[(ft.config == cfg) & (ft["shape"] == shape) & (ft.pair_id.isin(ids111))]
            s = g.groupby("seed").apply(lambda h: pd.Series(binmetrics(h.y, h.pred)), include_groups=False).mean()
            lines.append(f"| {name} | {F.SHAPES[shape]} | {g.pair_id.nunique()} | {s.accuracy:.3f} | {s.f1:.3f} |")

    # ---- two-stage pipeline ----
    lines += ["", "## Two-stage pipeline: LLM extracts the attributed/denied claim X, frozen NLI checks A => X", "",
              "| Extractor | Shape | rule | n | flagged (pos / neg rows) | Accuracy | F1 | AUC |", "|---|---|---|---|---|---|---|---|"]
    ts_rows = []
    for folder, name in MODELS.items():
        s2 = pd.read_csv(LLM / folder / "stage2.csv", keep_default_na=False)
        s2["flag"] = s2.flag.astype(str).str.lower().eq("true")
        for c in ["entail_A_to_x", "entail_A_to_defended", "entail_defended_to_A"]:
            s2[c] = pd.to_numeric(s2[c], errors="coerce")
        s2 = s2.merge(data[["pair_id", "label"]], on="pair_id")
        for shape, pos in POS.items():
            g = s2[s2.type_candidate == shape].copy()
            y = (g.label == pos).astype(int).values
            fl = g.flag.values & (g.x_claim.str.strip() != "").values
            ex = g.entail_A_to_x.fillna(0).values
            if shape == "strawman":
                rules = {"attributed X not entailed": (fl & (ex < 0.5), fl * (1 - ex))}
            else:
                ad, da = g.entail_A_to_defended.fillna(0).values, g.entail_defended_to_A.fillna(1).values
                rules = {"denied X entailed (false denial)": (fl & (ex >= 0.5), fl * ex),
                         "narrowing (A=>D, not D=>A)": (fl & (ad >= 0.5) & (da < 0.5), fl * ad * (1 - da))}
            flag_pos, flag_neg = fl[y == 1].mean(), fl[y == 0].mean()
            for rule, (pred, score) in rules.items():
                m = binmetrics(y, pred.astype(int), score)
                ts_rows.append(dict(extractor=name, shape=shape, rule=rule, n=len(g), flag_rate_pos=flag_pos,
                                    flag_rate_neg=flag_neg, **m))
                lines.append(f"| {name} | {F.SHAPES[shape]} | {rule} | {len(g)} | {flag_pos:.2f} / {flag_neg:.2f} | "
                             f"{m['accuracy']:.3f} | {m['f1']:.3f} | {m['auc']:.3f} |")
    pd.DataFrame(ts_rows).to_csv(F.OUT / "table9_two_stage.csv", index=False)

    with open(F.OUT / "RESULTS.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
