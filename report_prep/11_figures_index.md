# 11. Figures Index

All figures are 300 DPI PNGs, copied into `report_prep/figures/` for
convenience (originals in `figures/` at the repo root, generated via the
scripts in `src/figures/` — both copies are identical, kept in sync manually).
Each entry below: what it shows, what it's honest about (its limits), and
suggested placement.

## `figures/dataset_funnel.png`
- **Shows**: horizontal bar chart, log-scaled x-axis, six stages from
  293,297 raw utterances down to 111 hand-labeled examples.
- **Honest caveat to put in the caption**: the last two bars ("usable after
  claim extraction" = 240, "hand-labeled" = 111) are not a pure narrowing —
  only 91 of the 240 (from 3 of 6 batches) were actually hand-labeled, and
  the other 20 are separately hand-written synthetic examples, not drawn
  from that 240. Say this explicitly in the caption so it doesn't read as
  strictly sequential.
- **Suggested placement**: Dataset and Data Construction section (file 02),
  early — this is the figure that most efficiently conveys the scale of the
  pipeline work.

## `figures/model_comparison.png`
- **Shows**: grouped bar chart, F1 for XGBoost vs. zero-shot LLM, both
  shapes, on the 91 real examples.
- **Honest caveat**: pair with the zero-shot independence caveat (file 05) in
  the surrounding text, not just the caption.
- **Suggested placement**: Results section (file 06, Table A), immediately
  after or beside the table.

## `figures/label_distribution.png`
- **Shows**: stacked bar chart, final label counts (NONE/MOTTE_BAILEY/
  STRAWMAN), split by source (real vs. synthetic).
- **Suggested placement**: Dataset/Annotation section (file 04), near the
  final dataset composition table — slightly redundant with that table, so
  only include if there's space; otherwise the table alone suffices and this
  figure can move to the appendix.

## `figures/nli_scatter.png`
- **Shows**: scatter plot, entail(A→B) vs. entail(B→A), points colored by
  label (NONE/MOTTE_BAILEY/STRAWMAN), n=111, dashed diagonal reference line.
- **Honest caveat — critical, do not skip in the caption**: this figure shows
  a **null result** — classes do not separate cleanly; most points across all
  three labels cluster near (0,0). State this in the caption itself, not just
  in body text, so the figure isn't misread at a glance.
- **Suggested placement**: Analysis section (file 07.2), or appendix if space
  is tight — it's an honest negative result, valuable for completeness, but
  not load-bearing for the paper's main argument.

## `figures/nli_asymmetry_by_label.png`
- **Shows**: three-panel histogram, entail(A→B) − entail(B→A) distribution,
  one panel per label, n=54/28/29 respectively.
- **Honest caveat**: same null-result caveat as above — distributions
  overlap heavily near zero for all three labels.
- **Suggested placement**: same section as `nli_scatter.png`; only include
  one of the two NLI figures in the main body if space is tight (the scatter
  is slightly more information-dense), move the other to the appendix.

## `figures/real_vs_synthetic_features.png`
- **Shows**: two panels — (left) bar chart, `explicit_denial_B` rate, real
  (37%) vs. synthetic (0%); (right) box plot, claim length ratio B/A, real
  vs. synthetic.
- **Honest caveat**: frame as "a plausible explanation consistent with the
  observed performance difference," not a proven cause (file 07.3).
- **Suggested placement**: Analysis section (file 07.3), directly supporting
  the real-vs-synthetic discussion — this is one of the strongest, most
  concrete figures in the set; prioritize keeping it in the main body.

## Priority ranking if space forces cuts (7-8 pages is tight)

Given the user's own stated priority order, and consistent with it:
1. `dataset_funnel.png` — tells the pipeline-scale story efficiently
2. `model_comparison.png` — the headline result
3. `real_vs_synthetic_features.png` — strongest, most concrete analysis figure
4. One of `nli_scatter.png` / `nli_asymmetry_by_label.png` — the honest null result (pick one, not both, for the main body)
5. `label_distribution.png` — safe to cut to appendix or omit if the dataset-composition table alone covers it

A feature-ablation bar chart (Table from file 07.1) does not currently exist
as a figure — if you want one, it would need to be generated from
`dataset_final_ablation_results.csv`; the table alone may be sufficient
given space constraints.
