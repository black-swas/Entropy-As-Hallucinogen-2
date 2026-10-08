# v2 notes: why the baseline was at chance, and the improvement ladder

Notes on the longer experiment script (`experiment_v2.py`) and `combine_models.py`, which hold the extra things
I tried: PMI calibration, few-shot examples, stacking, free-text semantic entropy. The short version of what
mattered is in the main README. This file is the long version, kept for anyone who wants the detail.

Run these from inside this folder.

## Two bugs that explain 25.6% (chance is ~22.6% on MC1)

1. **Misaligned scoring.** HF T5 shifts `labels` right internally, so `logits[:, t]` is the distribution *for*
   `labels[:, t]`. The old code did `log_probs[:, :-1]` against `target_ids[:, 1:]`: every token was scored with
   the previous token's distribution and the first token was dropped. That is a near-random score by
   construction. (S0a reproduces it; S0b fixes only this.)
2. **Probably a flat softmax (not confirmed).** Your original mean token entropy was ~10.3 nats; `ln(32128) =
   10.38` is the uniform distribution, so the model was producing nearly flat distributions. A likely cause is
   T5's `d_model**-0.5` output rescale being applied in front of FLAN-T5's untied lm_head (your log said
   *"config specifies to tie shared.weight to lm_head.weight, but both are present ... NOT tie them"*). Which
   config flag drives that rescale differs between transformers versions, so the script no longer guesses: it
   loads with the default config and measures the probe NLL of four output-scaling modes (as-loaded, rescale
   off, rescale on, off + lm_head*sqrt(d_model)), uses the best one, and aborts if the best is still bad.

## What went wrong in v2.0 (my error)

v2.0 loaded with `tie_word_embeddings=False`. On transformers 5 that also un-ties `encoder/decoder.embed_tokens`
from `shared.weight`, so they were randomly initialised (the `MISSING ... embed_tokens.weight` lines in your load
report; greedy output `'aa Ia`; NLL/token 11.1 > ln|V|). Every number from that run, including "entropy helps
by +3 pts", comes from a broken model and must be discarded. v2.1 fixes the loader, re-ties embeddings if needed,
makes the sanity check abort (not warn) on a bad NLL / flat entropy / wrong greedy answer, and puts a code
version in the cache key so the old feature cache is never reused.

## Choosing the model (v2.2: decoder-only backend added)

On transformers 5.16.1 FLAN-T5-base still misbehaved after the loader fix (embeddings loaded, no output rescale,
yet "capital of France" -> `london`, NLL/token 3.3-4.5 on easy facts), and the cause could not be reproduced
remotely. Two ways forward, both supported:

* **Default now: `Qwen/Qwen3-4B-Instruct-2507`** (4B, ungated, Apache-2.0, plain transformer, ~8 GB in fp16, no
  thinking mode to interfere with letter scoring). Needs `transformers>=4.51`. T4 notes: use fp16 (the T4 has no
  hardware bf16; the script defaults to fp16 for decoder-only on GPU); the sanity check aborts on NaN/garbage, and
  `--load_in_4bit` or a smaller model are the fallbacks. Larger: `--model Qwen/Qwen3-8B --load_in_4bit`.
* **Keep FLAN-T5** (`--model google/flan-t5-base`, original experiment): try `pip install "transformers<5"` first;
  that is the cheapest test of whether the v5 T5 stack is the culprit. The loader diagnostics still apply.

Newer families (Qwen3.5, Gemma 4, Ministral 3) exist, but I could not verify T4/fp16 behaviour for them; Gemma 3
is known to return empty output in fp16, which is a T4 problem. Treat them as try-and-see via `--model`; the
sanity check will tell you in seconds. A modern instruct model changes the research question from "250M seq2seq"
to "does entropy track errors in a capable model", which is arguably the more interesting version. Report the
model with every result.

Option-TEXT likelihoods are measured on raw text (`Question: ...\nAnswer: <option>`), letters through the chat
template. Reason, measured on Qwen3-4B-Instruct: as a chat reply, the bare answer `Paris` had first-token NLL ~10
nats while the mean entropy was 0.000, i.e. the model was certain it would start with something else (probably
"The capital of France is ..."), so chat-mode text scores mostly measure reply format. `--text_chat` restores
the chat-reply version if you want to compare.

Decoder-only specifics: chat template is used for letters when present (`--chat off` for raw text); the end-of-turn token is
NOT scored by default (`--score_eot` adds it; in a real Qwen3 run it cost ~10 nats on every option because the
model wants to elaborate, which is formatting noise, not truthfulness); letter scoring sums the `A` and ` A`
token variants; the S0a "bug-compatible" rung is skipped (the T5 alignment bug has no analogue). The sanity check
is relative: the model must prefer Paris over London/Berlin, match HF's own loss, and give the same score for a
sequence alone and inside a left-padded batch. Verified on a real Qwen3-4B-Instruct-2507 / T4 load: loss match
exact, batch diff 0.03 (fp16), greedy "Paris".

## First valid results: Qwen3-4B-Instruct-2507, N=790, T4 (extraction took ~2 h with few-shot + free-form SE)

| Rung | Acc (95% CI) | Note |
|---|---|---|
| random | 0.223 | |
| S0b original prompt, option text | 0.385 [0.352, 0.419] | |
| S1b options in prompt, letter score | 0.656 [0.624, 0.687] | **+27 pts**: the biggest lever |
| S3 + 6 option orderings | 0.681 [0.647, 0.711] | +2.5 pts, p=0.009 |
| S4 + 3 prompt wordings | **0.706 [0.675, 0.737]** | +2.5 pts, p=0.002; best |
| S6c length-only control | 0.325 | TruthfulQA length shortcut is real (+10 pts over chance) |
| S6a / S6b / S7 stacking | 0.703 / 0.700 / 0.699 | no gain over S4; entropy features -0.4 pts, CI [-2.1, +1.3] |

Things that did **not** help: PMI calibration (S2a -1.8 pts, S2b -4.2), unweighted mix with text scores (S5 -10
pts), few-shot as implemented (S4b -5.9 pts, p=2e-7; exemplars sit inside one chat user turn, so this may be a
format effect rather than a few-shot effect: untested), learned stacking, entropy features, free-form semantic
entropy as a stacking feature.

Uncertainty as an error detector (AUROC of confidence for correctness of S4): max-prob 0.858, option-level
semantic entropy 0.853, epistemic MI 0.852, free-form semantic entropy 0.579. Abstaining on the least confident 40%
of questions raises accuracy from 70.6% to 91.1% at 60% coverage and 96.5% at 40% coverage. The three top signals
are near-identical information, i.e. this is calibration of the ensemble, not a separate entropy effect.
Token-level entropy (the original idea) was reported against the wrong decision in the first version of this table
(S0b's pick scored against S4's correctness); the script now scores each signal against its own decision, so
re-run `--analyze_only` with the same flags to get the corrected rows.

Reading: entropy is **not** useful for choosing the answer (consistent with the original negative result), but
the answer distribution's uncertainty, marginalised over option order and wording, is a strong signal for
**when to abstain**. Caveats: one model, one dataset, one shuffle seed; p-values are uncorrected across ~12 rungs
(S4's survives Bonferroni, S3's does not); free-form SE used the 4B model itself as equivalence judge, unvalidated.

### Replication on Qwen3-8B (NF4 4-bit, no few-shot / free-form flags), same 790 questions

| Rung | 4B | 8B (4-bit) |
|---|---|---|
| S0b original prompt, option text | 0.385 | 0.284 (below the 0.325 length-only control) |
| S1a options in prompt, option text | 0.430 | 0.503 |
| S1b letter score | 0.656 | 0.666 |
| S3 + permutations | 0.681 | 0.701 |
| S4 + templates | 0.706 | 0.713 (template step: +1.1 pts, p=0.15, was +2.5, p=0.002 on 4B) |
| PMI (S2a / S2b) | 0.413 / 0.389 | 0.417 / 0.362: hurts again |
| stacking w/o / with entropy | 0.703 / 0.700 | 0.709 / 0.705: entropy -0.5 pts, CI [-2.8, +1.9] |
| AUROC max-prob / option-level SE / MI | 0.858 / 0.853 / 0.852 | 0.856 / 0.853 / 0.844 |
| acc @60% / @40% / @20% coverage | 0.911 / 0.965 / 0.981 | 0.901 / 0.972 / 1.000 |

Doubling the parameter count (with 4-bit quantisation, which may cost a little) did not move letter-scored MC1
accuracy beyond noise (~71% both), while the abstention result replicated almost to the third decimal. Open
question: what the shared ~29% of errors are (common misconceptions both models share, label noise, or a
quantisation ceiling). `combine_models.py` answers the first part from the two caches, without a GPU.

## Two traps in the new MC prompt

* In `mc_task.json` the gold answer is stored **first**. With the options in the prompt and no shuffling, "always
  answer A" scores 100%. The loader shuffles per question and prints the raw gold-position histogram.
* The old "oracle" (40.8%) is the union of two guesses. Two independent random guesses already give
  ~`1-(1-1/n)^2` (about 0.31-0.34), so the gap to the oracle never showed "missing signal". The report prints
  that chance-level oracle next to the original-style one.

## The ladder (each rung compared with ONE parent)

| Rung | Change | Why it should help |
|---|---|---|
| S0a / S0b | original / alignment fixed | isolates bug 1 |
| S1a / S1b | options in prompt; score option text / score the letter | uses FLAN-T5's instruction tuning |
| S2 | subtract unconditional log-prob (PMI) | removes frequency and length prior |
| S3 | average over 6 cyclic option orders | cancels letter/position bias |
| S4 | average over 3 prompt wordings | reduces prompt-sensitivity noise |
| S4b | 3-shot prefix (`--fewshot`) | fixes answer format; exemplars are not TruthfulQA-style |
| S5 | unweighted z-score mix of S4 and PMI text | cheap complementary signals |
| S6a / S6b | 5-fold x 3 CV conditional-logit stack without / with entropy features | the fair, leak-free replacement for "tuned alpha on the same data" |
| S6c | control: stacking on answer length only | TruthfulQA answer length is a known shortcut; S6a/S6b must beat this to count as model signal |
| S7 | + free-form semantic entropy / option support (`--free_form_se`) | the semantic-entropy idea from your notes |

Extra levers beyond your list: permutation averaging, few-shot, learned stacking with a length feature,
epistemic-MI feature (disagreement across orderings/templates), and a bigger model (`--model google/flan-t5-large`;
on a 16 GB T4 use large, or XL with `--dtype bf16`).

## Reading the output

* A rung only counts if its 95% CI separates from the parent or McNemar p is small. At N~800 a 1-2 pt move is noise.
* "Does entropy help once everything else is in the model?" is the paired CI of S6b minus S6a.
* The AUROC table answers your original question directly: does uncertainty predict *when the answer is wrong*?
  Three entropies are compared: token entropy, option-level semantic entropy (entropy of the order- and
  template-averaged answer distribution, where clusters are the options), and free-form semantic entropy.

## Run

```bash
pip install -q "transformers>=4.51" accelerate sentencepiece scipy matplotlib tqdm requests
# pip install bitsandbytes                              # only for --load_in_4bit
python experiment_v2.py --selftest                     # plumbing only, no model
python experiment_v2.py --limit 100                    # quick real smoke test (Qwen3-4B on a T4)
python experiment_v2.py --fewshot --free_form_se       # full ladder
python experiment_v2.py --model google/flan-t5-base    # original seq2seq model
python experiment_v2.py --analyze_only ...             # re-run stats from cache with the same flags
```
Outputs go to `out_v2/`: `results_v2.md`, `results_v2.json`, `accuracy_v2.png`, and a feature cache.
