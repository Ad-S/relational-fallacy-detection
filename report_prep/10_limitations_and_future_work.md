# 10. Limitations and Future Work

## Limitations — state plainly, frame each as motivating a specific next step

- **Only 91 real labeled examples currently.** Small by any standard; all
  results in files 06-07 should be read as indicative, not conclusive.
  → Motivates: labeling batches 4-6 (149 additional usable candidates,
  extraction already done) as the highest-value low-effort next step.

- **Single human annotator; no inter-annotator agreement computed.**
  Directly relevant given Ruggeri et al. (2025)'s finding that fallacy
  judgments carry genuine human disagreement — this project currently cannot
  distinguish "genuine ambiguity" from "annotator error" in its own labels.
  → Motivates: either a second human rater on a subsample, or the
  previously-discussed LLM-as-second-rater proxy (explicitly flagged as a
  proxy, not equivalent to human IAA, if used).

- **20 synthetic examples are not naturally occurring data.** File 07.3
  shows a measurable distributional difference from real data on at least
  the `explicit_denial_B` feature (37.4% vs. 0.0%).
  → Already partially addressed by reporting real-only and real+synthetic
  results separately (file 06, Table B) rather than only the combined number.

- **Keyword-based candidate selection creates a traceable selection bias**
  (file 02) — every candidate in the dataset contains one of a fixed list of
  trigger phrases by construction. Plausibly connected to the shared
  lexical-shortcut finding in file 08.
  → Motivates: the trigger-phrase-masking ablation (below), the single
  highest-priority remaining experiment.

- **The zero-shot baseline is not fully independent of the extraction
  process** (file 05) — same model performed both, within shared context.
  → A threat to how much the 0.750/0.800 F1 numbers can be trusted as a
  clean LLM-capability measurement; state this directly alongside the result
  (file 06), not only here.

- **CMV is one specific discourse register** — structured, good-faith-norm-
  driven online debate with an explicit persuasion mechanic (deltas).
  Findings may not transfer to other discourse types, e.g. the political
  debate transcripts named in the original proposal as a second data source
  but not yet implemented.

- **Only a frozen, off-the-shelf NLI model and simple lexicon-based features
  are used — no fine-tuned transformer yet.** This was a deliberate scope
  decision, not an oversight: with 44-58 examples per shape, fine-tuning a
  transformer risks memorization over generalization, and a single
  evaluation on this little data would be too noisy to interpret reliably.
  → Explicitly conditional future work, see below — only after the dataset
  is meaningfully larger.

- **Per-shape binary evaluation (file 05) is a deliberate, justified
  design choice**, but it means the current work does not yet test joint
  discrimination between all three outcomes under a design that avoids the
  structural-shortcut problem. An open methodological question: are there
  features that vary meaningfully *within* a shape (not just between shapes)
  that could support a principled joint evaluation later?

- **Distinguishing legitimate claim refinement from motte-and-bailey, and a
  fair paraphrase/clarifying question from strawman, remains genuinely hard**
  even for the human annotator in borderline cases — see the guideline's own
  worked counter-examples (file 04) and the three flagged weaker synthetic
  strawman examples (file 04). This is a property of the task's difficulty,
  not a flaw in the protocol — worth stating as such.

## Future work and remaining timeline (Oct 3 – Oct 31)

**1. Trigger-phrase-masking ablation — highest priority (~2-3 days)**
Mask or remove the exact denial/retreat phrases from claim_B, then re-run
both XGBoost and the zero-shot baseline on the masked text. This is the
controlled test of the file 08 finding: if performance drops substantially
when the phrase is removed, that confirms the shortcut hypothesis; if
performance holds, the current results are more trustworthy than the
qualitative analysis alone suggests.

**2. Additional real annotation (~1 week)**
Batches 4, 5, and 6 already have claim extraction completed (149 additional
usable candidates beyond the 91 currently labeled — see file 03). Hand-
labeling these would roughly double the real dataset with no further
pipeline engineering required, directly addressing the small-N limitation
above.

**3. Inter-annotator agreement (~2-3 days)**
Either recruit a second human rater for a subsample of the labeled data, or
run the previously-discussed LLM-as-second-rater proxy — explicitly flagged
in the report as a proxy measurement, not equivalent to genuine human-human
IAA, if used instead of a second human.

**4. Transformer fine-tuning (~1 week, after items 1-2 above)**
Fine-tune DeBERTa/RoBERTa on claim_A + claim_B (plus discourse features where
applicable, i.e. for a future joint evaluation design), once the dataset is
meaningfully larger than 91-111 examples. **Explicitly conditional on item 2
above** — not attempted on the current dataset size, per the overfitting
concern stated above. Do not present this as a mandatory next step
independent of dataset growth.

**5. Full re-run of ablation + error analysis + final write-up (remaining
time to Oct 31)**
Repeat the analyses in files 06-08 on the expanded dataset; finalize the
complete write-up for the final submission.

## A note on scope evolution (useful for framing, not a limitation per se)

The original proposal named political debate transcripts (UC Santa Barbara's
American Presidency Project archive) as a second planned data source,
alongside CMV. This mid-submission implements and evaluates the CMV pipeline
only. **[Decision needed from you]**: state in the report either (a) this
second source remains planned but deprioritized given solo-project time
constraints, or (b) drop the mention entirely and scope the report to CMV
only. Either is defensible; just be consistent between the Introduction,
Related Work, and this Future Work section.
