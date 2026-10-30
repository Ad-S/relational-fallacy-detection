# 5. Features, models and methods

All trained models use one binary classifier per shape (MOTTE_BAILEY vs NONE, STRAWMAN vs NONE), the
decision threshold 0.5, and the grouped cross-validation of `06_evaluation_protocol.md`.

## 5.1 NLI features
- Model: frozen `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (Laurer et al., 2024), never
  fine-tuned for the feature models.
- For each pair: entailment, neutral and contradiction probabilities for **A -> B** (A premise, B
  hypothesis) and **B -> A**: `entail_AB, neutral_AB, contra_AB, entail_BA, neutral_BA, contra_BA`.
- Inputs truncated to 256 tokens. The challenge C is not used by these features.
- Computed for all 683 pairs on GPU in float32; on the 188 rows also computed earlier on CPU, the largest
  difference is 0.008.
- Observation from the mid-term analysis: on whole claims the model mostly answers "neutral" (78% of pairs
  had neutral > 0.5; median entailment about 0.004), because B typically adds content beyond A.

## 5.2 Lexical specificity features (`src/features/specificity_features.py`)
- `hedge_delta`, `hedge_rate_delta` (per word), `absolute_delta`, `length_ratio_B_over_A`,
  `explicit_denial_B` (regex: "I never said", "I didn't/did not say", "I'm/am not saying",
  "I wasn't saying", "that's not what I said").
- Hedge lexicon (27 terms): some, often, sometimes, generally, usually, may, might, could, possibly,
  perhaps, maybe, a few, many, mostly, somewhat, arguably, typically, occasionally, i think, i believe,
  seems, appears, tends to, in some cases, under certain conditions, not necessarily, not always.
- Absolute lexicon (18 terms): all, every, always, never, completely, totally, only, none, no one,
  everyone, everything, nothing, must, certainly, definitely, absolutely, entirely, utterly.
- Known limitation: counts use plain substring matching, so "all" also matches "really" or "especially",
  and overlapping terms are double-counted. On the mid-term data, 55% of absolute matches and 32% of hedge
  matches were such spurious substrings. The features were kept unchanged for comparability with the
  mid-term report; they are the weakest feature family in every experiment.
- `explicit_denial_B` never covers strawman triggers, so it is 0 for all strawman-shaped rows by construction.

## 5.3 XGBoost (`src/models/xgb_grouped.py`)
- 50 trees, maximum depth 3, learning rate 0.1, random_state 42 (same settings as the mid-term report).
- Feature sets: NLI only (6), lexical only (5), NLI + lexical (11).
- Uses exactly the fine-tuning fold assignments (copied from the fine-tuning predictions).

## 5.4 Fine-tuned cross-encoders (`finetune/train_cv.py`)
- **Input:** motte-and-bailey `[A] [SEP] [C] [SEP] [B]` (A as the first segment, "C [SEP] B" as the
  second); strawman `[A] [SEP] [B]`. **B-only ablation:** `[B]` alone.
- **Initialisations:**
  - DeBERTa-NLI: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435M parameters); the 3-way NLI
    head is replaced by a new 2-way head.
  - DeBERTa plain: `microsoft/deberta-v3-large` (same tokenizer).
  - RoBERTa-large-MNLI (`FacebookAI/roberta-large-mnli`) was attempted; its predictions became NaN under
    bf16 mixed-precision training and it is excluded.
- **Training:** AdamW, learning rate 1.5e-5 (sweep: 1e-5, 3e-5), weight decay 0.01, batch size 8, 6 epochs,
  linear schedule with 10% warmup, gradient clipping 1.0, max length 256. Weights kept in float32 with bf16
  autocast on GPU. A fresh output layer (normal, std 0.02) and a fresh copy of the pretrained model for
  every fold and seed. Fixed epoch budget, no early stopping (inner validation splits would be ~10 examples).
- **Seeds:** main DeBERTa-NLI model 5 seeds (0-4); all other configurations 3 seeds (0-2).
- **Learning curve:** a seeded random 25%, 50% or 75% of the training conversations is kept in each fold;
  the test folds are unchanged.
- About 280 (motte-and-bailey) / 265 (strawman) training pairs per fold with the full data.
- Peak GPU memory about 10.6 GB; about 0.09 s per training step on an H200.

## 5.5 Trigger-phrase masking (test time)
- Each fine-tuned model is also scored with the selection trigger phrases (section 3.3 lists) deleted
  from claim B (`finetune/triggers.py`, longest phrases first, whitespace collapsed).
- Affects 218 of 353 motte-and-bailey rows and 141 of 330 strawman rows.
- The model is not retrained; only the test input changes.

## 5.6 Layer-wise probing (`finetune/probe_layers.py`)
- Frozen DeBERTa-NLI and frozen plain DeBERTa (no fine-tuning).
- Input built exactly as for fine-tuning (A, C, B or A, B).
- Hidden states of all 25 outputs (embedding layer = layer 0, plus 24 transformer layers), mean-pooled over
  non-padding tokens.
- Per layer: standardised logistic regression (C = 0.1), 5-fold grouped CV, 3 seeds; accuracy, F1, ROC-AUC.

## 5.7 Zero-shot LLM baselines
- **Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct** with vLLM 0.31.0: bf16, temperature 0, seed 0, max 300
  output tokens, on all 683 pairs. Unparseable answers count as NONE (7B: 0; 32B: 2).
- **Prompt (definitions only, no examples):**
  - Motte-and-bailey: "Label MOTTE_BAILEY if Claim B retreats to a narrower, weaker or more easily defensible
    position than Claim A, while presenting that retreat as if it still defends Claim A ... Label NONE if
    Claim B is an honest clarification, adds detail, restates the same position, or openly concedes the
    point." Inputs: A, C, B. Output: JSON with label and a one-sentence reason.
  - Strawman: "Label STRAWMAN if Claim B misrepresents Claim A as a different, stronger or more extreme
    position than the one actually stated, and argues against that distorted version. Label NONE if Claim B
    fairly paraphrases Claim A, asks a genuine clarifying question (even a pointed one), or disagrees with
    what Claim A actually says."
- **Claude Opus 5.5** with the same prompts, on the 111 original pairs of the mid-term dataset (91 real + 20
  hand-written), one isolated session per example.
- Full prompts: `llm_bundle/zeroshot_prompts.py`, also saved with each run in `prompts.json`.

## 5.8 Two-stage pipeline: extract, then verify (`llm_bundle/`)
Operationalises "strawman as misuse of rephrase": a strawman reports a commitment A never made; a
motte-and-bailey denies a commitment A did make.
- **Stage 1 (LLM extraction, Qwen 7B or 32B):**
  - Strawman: does B attribute a position to A's author (for example "so you're saying ...")? If so,
    write it as "The author of Claim A claims that ...".
  - Motte-and-bailey: does B deny, narrow or reinterpret what the speaker said? If so, write the denied
    claim and the position now defended.
- **Stage 2 (frozen NLI):** P(A entails X) for the extracted claim X; for motte-and-bailey also
  P(A entails defended claim D) and P(D entails A).
- **Decision rules, fixed in advance (threshold 0.5):**
  - Strawman: STRAWMAN iff an attributed claim X exists and P(A => X) < 0.5.
  - Motte-and-bailey, false denial: MB iff a denied claim X exists and P(A => X) >= 0.5.
  - Motte-and-bailey, narrowing: MB iff P(A => D) >= 0.5 and P(D => A) < 0.5.
- Continuous scores for ROC-AUC: flag x (1 - P(A=>X)); flag x P(A=>X); flag x P(A=>D) x (1 - P(D=>A)).
