# Speculative decoding from scratch

`main.py` runs two decoders on the same prompt and seed. A small draft model proposes tokens, and a larger target model accepts or rejects them in one forward pass. The script then generates again with the target model alone, one token at a time.

The default pair is local `Qwen2.5-0.5B-Instruct` (draft) and `Qwen2.5-3B-Instruct` (target). Both must share a tokenizer and vocabulary. Weights load in bfloat16; sampling math is done in float32. The device is MPS when it is available, otherwise CPU.

Place the Hugging Face snapshots here (this directory is gitignored):

```
hf_models/Qwen2.5-0.5B-Instruct/
hf_models/Qwen2.5-3B-Instruct/
```

## Usage

```bash
uv sync
uv run python main.py --input "A dog has four "
```

`--approx_model_name` and `--target_model_name` override the paths above. `--num_tokens` (default 20) and `--gamma` (default 4) set the length and the draft window. See `python main.py --help` for temperature (`0` is greedy), top-k, top-p, and seed.

## Tests

```bash
uv run pytest
```

The tests check that speculative sampling is lossless: on toy models, sampled tokens follow the target model's processed distribution (chi-square and total variation), and at temperature 0 the output matches the target model's greedy decode token for token. They also cover γ=1, rejection at the first draft, full acceptance, and `num_tokens < γ`. The greedy test downloads the 19 MB [`onnx-internal-testing/tiny-random-Qwen2ForCausalLM`](https://huggingface.co/onnx-internal-testing/tiny-random-Qwen2ForCausalLM) on first run and is skipped if it cannot be loaded.

## References

- [feifeibear/LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling/tree/1571d8998f0009bb6b1275f61c10096bd33f9018)
- Leviathan, Kalman, and Matias. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling*. arXiv:2302.01318.
