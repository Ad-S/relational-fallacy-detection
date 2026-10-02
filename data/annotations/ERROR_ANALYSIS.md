# Error Analysis

Based on joining XGBoost's out-of-fold predictions (5-fold CV, full feature set)
with the zero-shot LLM baseline's predictions, on the 91 real CMV examples where
both exist. 48 of 91 examples were misclassified by at least one model; 9 were
missed by both.

## Error categories

| Error type | XGBoost | LLM | Representative examples |
|---|:---:|:---:|---|
| **Lexical cue triggered without a real retreat/distortion** -- phrase like "I'm not saying X" or "are you saying Y?" is present, but claim_B actually restates/matches claim_A, not narrows or distorts it | X | X | `m026`, `m029`, `m073`, `s064`, `m008`, `m013`, `m017`, `s009` |
| **Real retreat/distortion missed when NOT phrased with an obvious trigger** -- the fallacy is genuinely there, but worded subtly (quantitative hedge, mirrored-analogy strawman, etc.) | X | X | `m005`, `m052`, `m024`, `s013`, `s017`, `s019` |
| **Strawman embedded inside an otherwise substantive, fair-sounding rebuttal** -- claim_B makes some fair points AND a distortion in the same breath; models/annotator anchor on the fair part | X | X | `s051`, `s061` |
| **Clarifying question or fair paraphrase misread as distortion** -- claim_B asks a genuine, even if pointed, clarifying question, not a strawman | X | X | `s049`, `s001` |

## What this supports

This is direct evidence for the "trigger-phrase shortcut" concern raised earlier
in the project: 8 of the 9 jointly-missed examples, plus several model-specific
errors, involve a denial/clarifying phrase being treated as decisive on its own,
in either direction (both false positives when the phrase appears without a real
retreat, and arguably a contributing factor when a retreat without the phrase
gets missed). This affects XGBoost and the LLM baseline similarly, despite being
very different model classes -- consistent with the shortcut living in the
**data's surface statistics** (since all candidates were found via that exact
phrase), not in either model's specific architecture.

**Representative quotes (for the write-up):**

- `m026` (false positive, both models): "*I'm fully aware of how the word is
  used. All I'm saying is that usage is often incorrect.*" -- contains the
  "all I'm saying is" pattern, but restates the same position as claim_A, not a
  narrower one.
- `s013` (false negative, LLM): claim_B is claim_A's exact argument with every
  instance of "car(s)" replaced by "gun(s)" -- a parody/mirrored-analogy
  strawman technique with no "so you're saying" framing, missed entirely.
- `s051` (false negative, both models): claim_B makes one fair correction
  ("that's communism, not socialism") immediately followed by one genuine
  strawman ("are you saying anyone with ambition can just snap their fingers
  and take over the country?") -- both models/the surface pattern treat the
  whole reply as fair pushback.
