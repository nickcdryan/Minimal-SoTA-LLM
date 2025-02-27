# Minimal-SoTA-LLM
Compact, legible, "state of the art" baseline decoder language model 

[Link to Colab Notebook](https://colab.research.google.com/drive/1l6ydblW754hXqhjcHiL8I17G_84OSaoE?usp=sharing)

## Description
This notebook provides an extremely minimal, legible, and easy-to-modify "state of the art" (February, 2025) decoder language model.

(I will periodically update the model to keep it fresh.)

I created this in order to have a quick baseline against which to test architecture and research ideas. This model approximates the "best practices" according to a survey of LLMs over the last few years. Special thanks to [Stella Biderman](https://twitter.com/BlancheMinerva/status/1740365334467756267) for aggregating and sharing all of these details.

## Features
I grabbed the [Mistral 7B](https://github.com/mistralai/mistral-src) model, cleaned it up a bit, and added components where necessary. 

**ARCHITECTURE:**
- RMS Norm
- ROPE with interpolation / extrapolation (via [rotary_embedding_torch](https://github.com/lucidrains/rotary-embedding-torch))
- Flash Attention (via F.scaled_dot_product_attention)
- SwiGLU
- ~~Parallel layers (PALM)~~ removed: latest architectures favor normal "two hop residual"
- Pre-layer norms
- GQA / MQA
- KV caching
- MLA with compressed KV caching (DeepSeek V3)

**MODEL CONFIG:**
- no bias
- no dropout
- use (4 \* dim) for regular ff, (8/3 \* dim) for GLU variants

**OPTIMIZATION:**
- Adam, AdamW, h params: .9, .95
- gradient clipping 1.0
- LR decay cosine schedule to 10%
- LR warmup ~3% of total training steps (used for Llama 1 -  no "standard" warmup schedule that I've seen)
- weight decay .1

**AN EXAMPLE TRAINING LOOP:**
- C4 dataset, streamed :) from huggingface datasets (async, pin_memory)
- gpt-neox-20b tokenizer
- torch.compile()
- automatic mixed precision (AMP) autocast
- disable torch debuggers for performance
- ~~gradient accumulation~~
- many tips from the [pytorch guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)


**TODO/QUESTIONS:**
- T5 packing? (pack multiple training examples to fit in one context window)
- pre-norm and/or post-norm?
- warmup regimen?
- user requests
- optimizer.step warning when using gradscaler [seems harmless](https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/7)


## What this doesn't do:
This project is not built for speed, inference, or parallelism. It is best to use it as a reference benchmark while training, or as a base upon which to experiment with new architecture modifications. The "state of the art" (whatever that means to you) changes all the time, so quickly implementing and testing changes should be easy and fast.

NB: obviously this "state of the art" configuration is not guaranteed to work best for your dataset, your setup, your scale, etc. and there's valid disagreement about the value of each little change. However this should approximate a very good baseline for you. 

If you want something more performant and with a bunch of off-the-shelf transformer building blocks, [X-Transformers](https://github.com/lucidrains/x-transformers), and [xformers](https://github.com/facebookresearch/xformers) are good options. You'll have to invest some time in order to modify / understand them when compared to this project, which is intended to be understood and hackable with minimal effort.

## Copy wandb benchmark runs:
Once you've got a good benchmark run, you want to copy it out to multiple projects. For some reason you can't do this in wandb easily, so here's a script.
Not totally relevant to this project, but I use it all the time.

```python
!pip install wandb
import wandb
wandb.login()
api = wandb.Api()

src_entity = "johnsmith"
src_project = "project-1"
src_name = "benchmark-run-1"

dst_entity = "johnsmith"
dst_project = "project-2"

runs = api.runs(f"{src_entity}/{src_project}")

for run in runs:
    if run.name == src_name:
        # Get the run history and files
        history = run.history()
        files = run.files()

        # Create a new run in the destination project
        new_run = wandb.init(project=dst_project, entity=dst_entity, config=run.config, name=run.name,resume="allow")

        # Log the history to the new run
        for index, row in history.iterrows():

            # Include project step_size. Can also enter this manually
            step_size = history['_step'].values[1]

            new_run.log(row.to_dict(), step= index * step_size)

        # Upload the files to the new run
        for file in files:
            file.download(replace=True)
            new_run.save(file.name,policy = "now")

        # Finish the new run
        new_run.finish()
```



