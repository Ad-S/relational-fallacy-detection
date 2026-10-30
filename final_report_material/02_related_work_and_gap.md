# 2. Related work and research gap

Verify exact bibliographic details (venue, year, page numbers) against the source before citing;
details that could not be verified are marked.

## 2.1 Fallacy detection on single texts or spans
- **MAFALDA** (Helwe et al., NAACL 2024): unified taxonomy of 30+ fallacy types; span-level annotations;
  evaluates LLMs and classifiers.
- **LOGIC** (Jin et al., Findings of EMNLP 2022): logical fallacy dataset built from quiz-style texts;
  strawman is one of its classes.
- **Alhindi et al. (EMNLP 2022):** multitask instruction-based prompting over 28 fallacy types from five
  datasets; strawman is among the types (reported in their summary; verify).
- **CoCoLoFa** (Jin et al., EMNLP 2024): news comments with common fallacies written by LLM-assisted crowds;
  seed-then-augment data construction.
- **FAINA** (Ruggeri et al., NAACL 2025): Italian social-media posts, 20 fallacy types, span-level, keeps
  individual annotator labels (human label variation).

Takeaway: strawman already exists as a label, but the label is attached to one text; the original claim
that is being misrepresented is not part of the unit of annotation.

## 2.2 Fallacies with argument structure or conversational context
- **Goffredo et al. (IJCAI 2022; EMNLP 2023):** fallacies in U.S. presidential debates (ElecDeb60to20);
  the 2023 work adds argument components (claims, premises) and support/attack relations.
- **Dore et al. (2026, arXiv 2608.27471):** retrieval-augmented fallacy detection in political debates,
  using support/attack relations to guide retrieval.
- **Sahai et al. (ACL 2021):** Reddit fallacy dataset built from comments that name a fallacy (noisy labels,
  cleaned by crowdsourcing); adding the parent comment improves models. Strawman is not among its 8 final types.
- **Habernal et al. (NAACL 2018):** ad hominem in r/ChangeMyView.

Takeaway: context helps, but the label still belongs to one component or comment.

## 2.3 Cross-turn semantic relations without fallacy labels
- **QT30 (LREC 2022) and the DialAM-2024 shared task:** BBC Question Time debates annotated with
  Inference, Conflict and Rephrase relations between propositions across turns.
- **Koszowy et al. (Informal Logic, 2022):** pragmatic account of rephrase types in argumentation.
- **Uberna et al. (ICAART 2026, arXiv 2602.13713):** classify rephrase functions (deintensification,
  intensification, specification, generalisation, other) in political debates with LLM agents
  (only the abstract was checked).

Takeaway: the closest representation to this project, but without fallacy labels and not on online discussion.

## 2.4 The theory this project operationalises
- **Visser et al. (European Conference on Argumentation 2017; proceedings 2018), "Straw man as misuse of
  rephrase":** models a strawman as a relation between two propositions - a rephrase used to misrepresent
  the opponent's claim. It also describes retreat to "safe ground" by weakening a standpoint through
  rephrase (close to motte-and-bailey), argues that paraphrase or entailment between spans is neither
  necessary nor sufficient, and lists linguistic cues for calling out such fallacies as future work. It is
  theoretical, built around one debate example, with no dataset or classifier. (Title and year differ
  slightly between listings - check before citing.)
- **Shackel (Metaphilosophy, 2005):** origin of the term motte-and-bailey. No computational work on
  motte-and-bailey was found.

## 2.5 Evaluation validity
- **Singh et al. (2026, arXiv 2609.18644), "Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy
  Detection":** detectors label valid arguments that share a fallacy's scheme as fallacious; benchmarks lack
  scheme-matched negatives. This parallels the trigger-phrase shortcut found here.

## 2.6 NLI as a tool
- **Bowman et al. (EMNLP 2015):** SNLI and the NLI task.
- **Laurer et al. (2024):** NLI-based transfer for classification with little labeled data; source of the
  DeBERTa-v3-large NLI checkpoint used throughout (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli).
- **Toulmin (1958), Dung (1995):** claim as the unit; arguments and attacks as relations.

## 2.7 Tools and data
- **ConvoKit** (Chang et al., SIGDIAL 2020); **Winning Arguments / CMV corpus** (Tan et al., WWW 2016).
- **XGBoost** (Chen and Guestrin, KDD 2016). **vLLM** for LLM inference. **Qwen2.5** open-weight models.

## 2.8 The research gap (how to state it)
Single-text fallacy datasets label texts; cross-turn relation corpora label relations without fallacies;
Visser et al. connect the two only in theory. Missing is **an annotated, evaluated set of cross-turn claim
pairs for strawman and motte-and-bailey in online discussion, and models evaluated on it** - and there is
no computational motte-and-bailey work at all in the reviewed literature.

Do not claim: "first to detect strawman" (it is a class in single-text datasets); "first relational
formulation" (Visser et al. proposed it). Do claim: the first empirical operationalisation and evaluation
of that relational view for these two fallacies, to the best of the review.
