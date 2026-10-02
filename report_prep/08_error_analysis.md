# 8. Error Analysis

Full source document: `data/annotations/ERROR_ANALYSIS.md` (use it directly —
it already contains verified categories and verbatim quotes; this file
summarizes and adds framing for the report).

## Method

Built by joining XGBoost's out-of-fold predictions (5-fold CV, full feature
set, from `dataset_final_xgboost_oof_predictions.csv`) with the zero-shot
LLM's predictions (`zeroshot_baseline_predictions.csv`), on the 91 real
examples where both exist (the only set both models were evaluated on — the
20 synthetic examples were never run through the zero-shot baseline, see
file 05).

## Headline numbers

- 48 of 91 real examples (53%) were misclassified by **at least one** model.
- 9 examples were misclassified by **both** models.
- 29 examples were missed by XGBoost only; 10 by the zero-shot LLM only.

## Four error categories (from `ERROR_ANALYSIS.md`, Table reproduced here)

| Error type | XGBoost | LLM | Representative examples |
|---|:---:|:---:|---|
| Lexical cue triggered without a real retreat/distortion | yes | yes | `m026`, `m029`, `m073`, `s064`, `m008`, `m013`, `m017`, `s009` |
| Real retreat/distortion missed when NOT phrased with an obvious trigger | yes | yes | `m005`, `m052`, `m024`, `s013`, `s017`, `s019` |
| Strawman embedded inside an otherwise substantive, fair-sounding rebuttal | yes | yes | `s051`, `s061` |
| Clarifying question or fair paraphrase misread as distortion | yes | yes | `s049`, `s001` |

## The central finding

**8 of the 9 jointly-missed examples**, plus several model-specific errors,
involve a denial/clarifying phrase (e.g., "I'm not saying X, I'm saying Y",
"are you saying...?") being treated as decisive on its own — in both
directions: as a false-positive trigger when the phrase appears without an
actual retreat/distortion, and plausibly contributing to false negatives when
a genuine retreat is present but *not* phrased with the expected trigger.

**This affects both XGBoost and the zero-shot LLM similarly, despite being
very different model classes.** State this precisely: this is consistent
with the shortcut living in the **data's surface statistics** (every
candidate was found via exactly these phrases — file 02) rather than in
either model's specific architecture. This is the project's most important
single finding — more informative than either model's raw F1 number, because
it points directly at *why* the current numbers may be inflated or deflated,
and it directly motivates the highest-priority item in future work (file 10:
the trigger-phrase-masking ablation).

**Hedge correctly, exactly as required**: call this a "suspected lexical
shortcut, supported by qualitative error analysis" — not a proven causal
mechanism. Nine qualitative examples support the hypothesis; they do not
constitute a controlled intervention. The planned trigger-phrase-masking
ablation (file 10) is the controlled test that would confirm or disconfirm
it.

## Representative quotes for the write-up (verbatim, verified against the data)

**`m026`** (false positive, both models predicted MOTTE_BAILEY, true label
NONE): claim_B reads "*I'm fully aware of how the word is used. All I'm
saying is that usage is often incorrect.*" — contains the "all I'm saying is"
pattern, but restates the same position as claim_A ("usage is often
incorrect"), not a narrower one. No actual retreat occurred.

**`s013`** (false negative, LLM predicted NONE, true label STRAWMAN):
claim_B is claim_A's exact argument with every instance of "car(s)" replaced
by "gun(s)" — a parody/mirrored-analogy technique with no "so you're saying"
framing at all, missed entirely by the phrase-sensitive pattern.

**`s051`** (false negative, both models predicted NONE, true label
STRAWMAN): claim_B makes one fair correction ("that's communism, not
socialism") immediately followed, in the same reply, by one genuine strawman
("are you saying anyone with ambition can just snap their fingers and take
over the country?"). Both models/the surface pattern appear to treat the
whole reply as fair pushback because of the legitimate correction alongside
it.

## A notable aside (good qualitative material, not part of the core finding)

One example in the broader (non-error) qualitative review, `s041` (from an
abortion-debate thread), contains a speaker *explicitly admitting* mid-
argument: "you are correct in saying that it is a strawman argument, I was
merely interested to see how you would respond." This is a rare, directly
self-labeled real-world strawman admission — useful as an illustrative
example of the phenomenon being real and recognized by participants
themselves, though note the specific extracted claim_A/claim_B pair for this
row was separately labeled NONE (the self-admission referred to a different
part of the fuller exchange than the specific pair extracted) — so it's a
good illustrative anecdote for the Introduction/Background, not a data point
to cite in the error-analysis table itself.

## Code and file references

- `data/annotations/ERROR_ANALYSIS.md` — full source document
- `data/annotations/dataset_final_xgboost_oof_predictions.csv` — XGBoost's
  per-example predictions (pair_id, true_y, pred_y, shape, positive_label)
- `data/annotations/zeroshot_baseline_predictions.csv` — LLM's per-example
  predictions (pair_id, type_candidate, predicted_label, true_label)
