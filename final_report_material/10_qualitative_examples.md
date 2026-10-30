# 10. Qualitative examples (real r/ChangeMyView pairs)

Claims are the extracted, condensed versions used for annotation. "FT-NLI" and "XGBoost" give the fraction
of seeds 0-2 predicting the fallacy (1.00 = all three seeds). "Masked" is FT-NLI with the trigger phrase
removed from B. Stage-1 X is the claim extracted by Qwen2.5-32B; P(A=>X) is the NLI entailment probability.

## 10.1 Strawman detected by the fine-tuned model but missed by XGBoost (9 such real cases)

**s011** - gold STRAWMAN; FT-NLI 1.00, XGBoost 0.33, masked 1.00, Qwen-32B STRAWMAN.
- A: an extended allegory about a group's cultural creation being taken and profited from by another group
  without credit (framed as cultural appropriation).
- B: "So what you're saying is either white people make money off of 'thing' or nobody does? Why should
  white people refrain - it's not costing you anything, you're just annoyed."
- Stage-1 X: "The author of Claim A claims that white people should refrain from using 'thing' ..."; P(A=>X) = 0.005
  -> the pipeline also calls it a strawman (attributed claim not supported by A).

**s030** - gold STRAWMAN; FT-NLI 1.00, XGBoost 0.00, masked 1.00, Qwen-32B STRAWMAN.
- A: "By 'we know how [the universe behaves]' I meant its behavior once already in motion, not what
  originally set it in motion."
- B: "So are you one of the ones saying 'I don't know how this happened'? And are you saying that's the same
  as saying 'God did it'?"
- Stage-1 X: A's author equates ignorance with "God did it"; P(A=>X) = 0.001 -> strawman.

Both keep their prediction when the trigger phrase is removed: the decision does not rest on "so you're saying".

## 10.2 Trigger-phrase false positive (motte-and-bailey shape)

**m026** - gold NONE; FT-NLI 0.67, XGBoost 0.67, masked **0.00**, Qwen-32B MOTTE_BAILEY.
- A: "'Retarded' isn't the same as 'stupid' - calling a non-developmentally-delayed person retarded likely
  just means you meant 'stupid'."
- C: "Most people think of 'retarded' as meaning exceptionally/permanently stupid, so using them
  interchangeably as hyperbole is common, if technically incorrect."
- B: "I'm fully aware of how the word is used. **All I'm saying is** that usage is often incorrect."
- B restates A's position; it does not retreat. The models (and Qwen-32B) flag it, and the fine-tuned model
  changes its answer to NONE as soon as "All I'm saying is" is removed - a direct illustration of the
  motte-and-bailey trigger shortcut. (This pair was also a shared error in the mid-term analysis.)

**m128** - gold NONE; FT-NLI 0.67, masked 0.33. B: "Maybe. All I'm saying is if several different people
call me selfish, I'd take time to evaluate why." Same pattern.

## 10.3 Motte-and-bailey missed by every system

**m005** - gold MOTTE_BAILEY; FT-NLI 0.00, XGBoost 0.33, Qwen-32B NONE.
- A: replace the fixed minimum wage with a bidding system where you propose your own salary above a floor.
- C: this removes flexibility and adds serious risk of losing income.
- B: "Removing flexibility is a real downside. What I meant by 'tiered' was different minimum floors
  depending on profession/age/experience, not one single all-or-nothing threshold."
- The retreat is real but phrased as a concession and clarification; nothing on the surface signals it.

**m061** - gold MOTTE_BAILEY; FT-NLI 0.00, XGBoost 0.00, Qwen-32B NONE.
- A: being transgender is a choice to alter a body that is not biologically wrong, so comparing it to birth
  defects is inaccurate.
- C (a trans commenter): transitioning may be a choice, but being trans is not.
- B: "I apologize if I was unclear - what I meant by 'choice' is that undergoing surgery is a choice, not that
  the underlying feelings/identity are a choice."
- Stage-1 X (denied claim): "Being transgender is a choice ..."; P(A=>X) = 0.999 - the denial is false, which
  the "false denial" rule would catch for this pair, but the rule fails on the data overall.

## 10.4 Fair reply flagged as strawman by both trained models (9 such real cases)

**s001** - gold NONE; FT-NLI 0.67, XGBoost 1.00, masked 1.00, Qwen-32B STRAWMAN.
- A: "Everyone implicitly agrees to a social contract by being a citizen (even by birth); if you don't like it,
  you're free to leave and forgo the benefits of citizenship."
- B: "So you're saying that just by being randomly born in a country, I owe it something? ..."
- B is a pointed but fair paraphrase. Stage-1 X: "...by being born in a country, one implicitly agrees to a
  social contract and owes something to that country"; P(A=>X) = 0.67 -> the two-stage rule correctly says
  NONE here, while every learned model and Qwen-32B say STRAWMAN.

**s023** - gold NONE; FT-NLI 0.67, XGBoost 1.00, Qwen-32B NONE. B points out an apparent inconsistency
between two of A's statements rather than distorting A.

## 10.5 Generated-variant examples (for the data section)
Hand-written seed syn001 (MOTTE_BAILEY), A: "I don't want to study anything right now because I'm pretty
sure AI is going to take all the jobs..." - B (seed): retreats to "finding a job will be harder".
- Topic-transfer variant: A "I'm not going to bother recycling anymore, because it's all a scam and literally
  none of it ever gets recycled..."; C points out that cans, cardboard and glass are recycled; B "I'm fine with
  sorting my trash, I just think recycling programs are less efficient than people make them out to be."
- Flip variant (label NONE): same A and C; B "Yeah, fair, I overstated it. AI won't take all the jobs, and I was
  probably using it as an excuse..."
