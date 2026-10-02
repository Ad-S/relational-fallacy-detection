# 7. Ablation and Deeper Analysis

## 7.1 Feature ablation

Source: `data/annotations/dataset_final_ablation_results.csv` (111
real+synthetic examples, 5-fold stratified CV, same XGBoost hyperparameters
as the main experiments).

| Shape | Feature set | N | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Motte-bailey | NLI only | 58 | 0.724 | 0.731 | 0.679 | 0.704 |
| Motte-bailey | Specificity only | 58 | 0.500 | 0.483 | 0.500 | 0.491 |
| Motte-bailey | NLI + Specificity | 58 | 0.569 | 0.545 | 0.643 | 0.590 |
| Strawman | NLI only | 53 | 0.566 | 0.594 | 0.655 | 0.623 |
| Strawman | Specificity only | 53 | 0.528 | 0.571 | 0.552 | 0.561 |
| Strawman | NLI + Specificity | 53 | 0.547 | 0.586 | 0.586 | 0.586 |

**The headline result**: for motte-bailey, **NLI-only (0.724 accuracy)
clearly outperforms the combined feature set (0.569 accuracy)** — adding the
specificity features actively reduces performance rather than complementing
the NLI signal. The same direction holds for strawman on F1 (0.623 NLI-only
vs. 0.586 combined), with a smaller gap on accuracy (0.566 vs. 0.547).

**Correct interpretation (state precisely)**: on this dataset size, the
simple, hand-built specificity features appear to contribute noise rather
than complementary signal when combined with the NLI features. **Do not
generalize this into "specificity features are useless" or "lexical features
don't matter"** — state it as specific to this dataset size and this
particular, simple feature-engineering choice (5 hand-counted lexical
numbers). A larger dataset, or better-designed lexical features, might behave
differently. This is also in some tension with the error analysis (file 08),
where one specific lexical signal (`explicit_denial_B`, one of the five
specificity features) is implicated in a shared shortcut — worth noting
explicitly that specificity features being net-negative in combination
doesn't mean every individual specificity feature is uninformative; it means
the *combination*, as currently weighted by the classifier on this little
data, underperforms NLI alone.

## 7.2 NLI feature-space visualization (a genuine null result)

Figures: `figures/nli_scatter.png` (entail(A→B) vs. entail(B→A), points
colored by label), `figures/nli_asymmetry_by_label.png` (three-panel
histogram of entail(A→B) − entail(B→A), one panel per label, n=111 for all).

**State this precisely, as a null result — do not oversell it**: neither the
scatter plot nor the asymmetry histograms show a clean visual separation
between `NONE`, `MOTTE_BAILEY`, and `STRAWMAN`. The majority of points across
all three labels cluster near (0, 0) in the scatter (mutual non-entailment in
both directions), and all three asymmetry histograms are concentrated near
zero with long, overlapping tails in both directions.

**Correct interpretation — this does not contradict 7.1**: a multivariate
classifier (XGBoost using all 6 NLI dimensions — entailment, neutral, and
contradiction probabilities in both directions, not just the 2 entailment
values plotted) can exploit combinations of signal that are not visible in a
simple 2D projection or a single derived asymmetry score. The correct
statement is: the *reducible, human-legible* version of the NLI signal (raw
bidirectional entailment, or their difference) is not sufficient for a
simple visual decision rule, even though the full 6-dimensional feature
vector evidently carries useful signal per 7.1's ablation result.

## 7.3 Real vs. synthetic distributional analysis

Figure: `figures/real_vs_synthetic_features.png` (two panels: denial-phrase
rate bar chart; claim-length-ratio box plot).

**Measured differences** (computed directly from
`dataset_final_specificity_features.csv` and `dataset_final_nli_features.csv`,
joined with `dataset_final.csv`'s `source` column):

| Feature | Real (n=91) | Synthetic (n=20) |
|---|---|---|
| `explicit_denial_B` rate | 0.374 (37.4%) | 0.000 (0.0%) |
| entail(A→B), mean | 0.100 | 0.004 |
| entail(B→A), mean | 0.115 | 0.014 |
| NLI asymmetry, mean | −0.015 | −0.010 |
| `hedge_delta`, mean | −0.044 | −0.150 |
| `length_ratio_B_over_A`, mean | 1.218 | 0.818 |
| claim_A word count, mean | 28.7 | 34.0 |
| claim_B word count, mean | 32.9 | 26.4 |

**The single most striking number**: `explicit_denial_B` is present in
**37.4% of real examples but literally 0% of synthetic examples.**

**Correct causal framing (this exact distinction matters, state it
carefully)**: the real examples were *selected* via keyword filtering on
exactly these denial/retreat phrases (file 02) — by construction, over a
third of them contain one. The synthetic examples were hand-written without
that same trigger-based selection constraint, so naturally fewer of them
happen to contain the literal phrase even when conceptually depicting a
retreat. **This is a plausible, well-evidenced explanation consistent with**
the Table B performance drop in file 06 — **it is not proof of a causal
mechanism.** Recommended wording: "the distributions suggest a plausible
distributional mismatch between the two sources, offering one explanation
for the modest performance change in Table B, though a dataset of this size
cannot establish this as causal."

**Supporting detail**: note that the NLI-feature distributions (entailment
means, asymmetry) do *not* differ nearly as starkly between real and
synthetic as the denial-phrase rate does — meaning the mismatch is
concentrated in the lexical/specificity features specifically, not a general
"synthetic text looks different to the NLI model" effect. This is useful
supporting detail, not a separate claim.

## Code and file references

- `src/models/ablation.py` — produces the Table in 7.1, plus out-of-fold
  predictions used in file 08
- `src/figures/make_nli_figures.py` — produces the two figures in 7.2
- `src/figures/make_synthetic_comparison.py` — produces the figure in 7.3
  and prints the exact denial-rate numbers
