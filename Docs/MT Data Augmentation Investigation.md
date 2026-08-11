# MT Data Augmentation Investigation (Future Scope)

## Context

The English-Balti MT stage was fine-tuned on `facebook/bouquet` (bft_Arab config — 504 dev + 854 test pairs), achieving **4.883 BLEU** on `facebook/nllb-200-distilled-600M`. Validation loss began climbing from step ~300 onward, indicating the model had exhausted what it could learn from 504 training examples — the constraint is data volume, not architecture or hyperparameters.

Before accepting this as the documented baseline, three additional data sources were investigated to see if BLEU could be meaningfully improved.

## Paths Investigated

### 1. Tibetic-family transfer learning
Balti is a Tibetic language, closely related to Standard Tibetan (bo) and Ladakhi, both of which have more NLLB/FLORES-200 coverage than Balti itself. The idea: fine-tune on a mix of Tibetan-English pairs plus the 504 Balti pairs, letting the model transfer grammatical/script knowledge from the larger related-language set.

**Status:** Viable in principle, but blocked on the same script problem as path 2 below — Tibetan-language parallel data is in Tibetan Unicode script, not the Perso-Arabic Nastaliq script the Balti pipeline uses throughout (ASR output, BOUQuET data). Mixing scripts without transliteration would corrupt training, not help it.

### 2. Distillation from `LiaqatAlisha/english-balti-translation`
This repository was initially assumed to be a parallel dataset. On inspection, it is a Hugging Face **model** repo — a fine-tuned M2M-100 checkpoint that translates English → Balti in **Tibetan script**, not a raw dataset.

Ran a manual sanity check (5 English sentences → model output):

| English | Model output (Tibetan script) |
|---|---|
| Hello, how are you? | ཅི་ན་ཡོ་། |
| What is your name? | ཡ་རི་མེན་ཏཁ༹་པོ་ཅིན་། |
| The weather is good today. | ནམ་ཟོ་དི་རིང་ལྱཁ༹་མོ་ཡོད་། |
| I am going home. | ང་ནང་ནུ་གྭེན་ཡོད་། |
| Thank you very much. | ཡ་རི་ཤཟ་དེ་། |

Output is structurally plausible — varied length correlating with input, correct tsheg spacing, grammatically consistent sentence-final particles (་། , ཡོད་/ཡིན་ endings typical of Tibetic syntax). No repetition or collapse across inputs. Semantic accuracy could not be verified (no Balti speaker or reference corpus available).

**Status:** Would require (a) treating this model's output as synthetic/distilled labels of unknown reliability, and (b) the same script conversion problem as path 1.

### 3. Community-sourced Balti translation data (balti.pk)
A crowdsourced English/Urdu → Balti platform supporting both Perso-Arabic and Tibetan scripts. Community-contributed, likely small and inconsistent in quality/coverage. Not pursued further given the higher-priority blockers above; worth a revisit if it grows.

## The Common Blocker: Tibetan → Balti Nastaliq Transliteration

Paths 1 and 2 both require converting Tibetan-script Balti/Tibetan text into the Perso-Arabic Nastaliq script used throughout this project. This is **not a solved problem** — no existing tool or library was found for it.

The realistic pipeline for this transliteration would be two hops:

1. **Tibetan Unicode → Wylie romanization** — solved. Wylie is a standard, deterministic transliteration scheme for Tibetan script (libraries such as `pyewts` exist). Balti's Tibetan-script form is documented under Wylie as `sbal ti`.
2. **Wylie/Latin → Balti Nastaliq** — **unsolved**. No reference implementation exists. Balti is largely phonetic, which helps, but building this mapping would mean constructing a novel rule-based (and likely hybrid rule + probabilistic) transliteration system from scratch, similar in spirit to published Hindi-Urdu or Sindhi Devanagari→Perso-Arabic transliteration work — except without an existing Balti-specific reference to build from.

Critically, there is currently no way to validate the accuracy of such a system's output — no Balti speaker or reference corpus available to check against. Building it now would mean stacking two sources of unverified uncertainty (unverified transliteration accuracy + unverified distilled-model quality) on top of an already tiny dataset.

## Decision

**Not pursued for the current release.** Documenting 4.883 BLEU on the 504-pair BOUQuET baseline as the honest, current result, and proceeding to pipeline assembly and streaming implementation.

## Future Scope

If a way to verify Balti output becomes available (e.g. access to a Balti speaker, a reference corpus, or contact with the BOUQuET/LiaqatAlisha authors for raw data), revisit in this order:

1. Reach out to `LiaqatAlisha` (HF) for the original raw English-Balti parallel pairs used to train their M2M-100 model — this sidesteps the transliteration problem entirely if raw pairs exist in Nastaliq or can be paired with a verified transliteration.
2. Build and validate the Wylie → Balti Nastaliq transliteration mapping (rule-based + probabilistic hybrid), using any available Balti dictionary/phrasebook resources as ground truth for validation.
3. Once transliteration is validated, revisit both the Tibetic-family transfer learning approach and M2M-100 distillation as data augmentation sources for the MT fine-tune.
4. Re-evaluate balti.pk's community dataset for growth/quality.
