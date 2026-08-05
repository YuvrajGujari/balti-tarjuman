## MT Stage — Fine-tune Translation (Balti → English) ✅ (5 August 2026)
- Investigated two pretrained MT options, both ruled out:
  - LiaqatAlisha/english-balti-translation: wrong direction (English→Balti), Tibetan-script output vs. dataset's Nastaliq script, no formal `bft` language code
  - facebook/nllb-200-distilled-600M: confirmed no native Balti (`bft`) coverage among its 200 languages
- Found facebook/bouquet's `bft_Arab` config: correct script, English-paired, curated — 504 dev + 854 test rows
- Fine-tuned facebook/nllb-200-distilled-600M on this data: added `bft_Arab` as a new special token, embedding warm-started from `bod_Tibt` (Tibetan, closest available known code)
- Trained 800 steps, best checkpoint step 650, **BLEU 4.883** — val loss climbing steadily from step 300 on, clear overfitting past that point (same pattern as Whisper Round 2)
- Verified live on Hub via SHA256, same method as Whisper checkpoints. Pushed to `YuvrajGujari/nllb-balti-mt`
- Low absolute BLEU is expected/honest given only 504 training examples and a cold-started language token — documented as a known limitation, not hidden
