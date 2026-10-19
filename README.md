# Speculative decoding from scratch

`main.py` runs two decoders on the same prompt and seed. A small draft model proposes tokens, and a larger target model accepts or rejects them in one forward pass. The script then generates again with the target model alone, one token at a time.

The default pair is local base models `Qwen2.5-0.5B` (draft) and `Qwen2.5-3B` (target), since the task is plain next-token continuation without a chat template. Both must share a tokenizer and vocabulary. Weights load in bfloat16; sampling math is done in float32. The device is MPS when it is available, otherwise CPU.

Both models keep a `DynamicCache`, so each forward pass only processes tokens the cache has not seen, and `logits_to_keep` limits the LM head to the positions that are sampled or verified. After a rejection, both caches are cropped back to the last accepted token.

Place the Hugging Face snapshots here (this directory is gitignored):

```
hf_models/Qwen2.5-0.5B/
hf_models/Qwen2.5-3B/
```

## Usage

```bash
uv sync
uv run python main.py --input "A dog has four "
```

`--approx_model_name` and `--target_model_name` override the paths above. `--num_tokens` (default 20) and `--gamma` (default 4) set the length and the draft window. See `python main.py --help` for temperature (`0` is greedy), top-k, top-p, and seed.

After one warm-up pass, both decoders are timed with `contexttimer`, with an MPS synchronize at each end. The script prints both texts and then a summary like:

```
speculative (gamma=4): 64 tokens in 3.68s (17.4 tok/s)
  20 rounds, accepted 44/78 drafts (56.4%), 2.20 accepted and 3.20 tokens per target forward
autoregressive: 64 tokens in 3.43s (18.7 tok/s)
speedup: 0.93x
```

The acceptance rate is accepted drafts over proposed drafts. Each round also appends one corrected or bonus token, so tokens per target forward is accepted per round plus one. In bfloat16 on MPS, greedy outputs of the two decoders can still split at a near-tie: one target forward over k+1 tokens rounds differently from k+1 single-token forwards. The float32 tests require an exact match.

## Tests

```bash
uv run pytest
```

The tests check that speculative sampling is lossless: on toy models, sampled tokens follow the target model's processed distribution (chi-square and total variation), and at temperature 0 the output matches the target model's greedy decode token for token. They also cover γ=1, rejection at the first draft, full acceptance, and `num_tokens < γ`. KV-cache tests compare cached logits after rollback with a full recompute, and check that each position reaches the target model once and that the LM head returns only the rows that are used. The greedy test downloads the 19 MB [`onnx-internal-testing/tiny-random-Qwen2ForCausalLM`](https://huggingface.co/onnx-internal-testing/tiny-random-Qwen2ForCausalLM) on first run and is skipped if it cannot be loaded.

## References

- [feifeibear/LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling/tree/1571d8998f0009bb6b1275f61c10096bd33f9018)
- Leviathan, Kalman, and Matias. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv:2302.01318.
