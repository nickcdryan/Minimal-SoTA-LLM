# Minimal-SoTA-LLM
Compact, legible, "state of the art" decoder language model


This project provides an extremely minimal, legible, and easy-to-modify "state of the art" (March, 2024) decoder language model.

I created this in order to have a quick baseline against which to test architecture modifications and research ideas. This model approximates the "best practices" according to a survey of LLMs over the last few years. Special thanks to [Stella Biderman](https://twitter.com/BlancheMinerva/status/1740365334467756267) for aggregating and sharing all of these details.

I grabbed the [Mistral 7B](https://github.com/mistralai/mistral-src) model, cleaned it up a bit, and added the following:

ARCHITECTURE:
- RMS Norm
- ROPE with interpolation / extrapolation (via [rotary_embedding_torch](https://github.com/lucidrains/rotary-embedding-torch))
- Flash Attention (via F.scaled_dot_product_attention)
- SwiGLU
- Pre-layer norm ff
- Parallel pre-layer norm ff
- Group query attention

MODEL CONFIG:
- no bias
- no dropout
- use (4 \* dim) for regular ff, (8/3 \* dim) for GLU variants

OPTIMIZATION:
- Adam, AdamW, h params: .9, .95
- gradient clipping 1.0
- LR decay cosine schedule to 10%
- LR warmup ~3% of total training steps (used for Llama 1 -  no "standard" warmup schedule that I've seen)
- weight decay .1

AN EXAMPLE TRAINING LOOP:
- C4 dataset, streamed :) from huggingface datasets (async, pin_memory)
- gpt-neox-20b tokenizer
- torch.compile()
- automatic mixed precision (AMP) autocast
- disable torch debuggers for performance
- gradient accumulation
- many tips from the [pytorch guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)


TODO:
- T5 packing? (pack multiple training examples to fit in one context window)
- user requests






This project is not built for optimal speed, inference, or parallelism. It is best to use it as a reference benchmark while training, or as a base upon which to experiment with new architecture modifications. The "state of the art" (or whatever that means to you) changes all the time, so quickly implementing and testing changes should be easy and fast.

If you want something faster and more performant with a bunch of off-the-shelf transformer building blocks, [X-Transformers](https://github.com/lucidrains/x-transformers), and [xformers](https://github.com/facebookresearch/xformers) are good options. You'll have to invest some time in order to modify / understand them when compared to this project, which is intended to be understood and hackable with minimal effort.



