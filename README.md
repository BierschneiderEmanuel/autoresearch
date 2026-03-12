# autoresearch

<img width="871" height="441" alt="autoresearch" src="https://github.com/user-attachments/assets/63d28977-34e0-4cc1-9d9d-10b05425f2cd" />

## Results after half a day of training
```
Baseline val_bpb:  1.978096
Best val_bpb:      1.089244
Total improvement: 0.888852 (44.93%)
Best experiment:   RoPE base 50000->60000

Cumulative effort per improvement:
  Experiment #  0: bpb=1.978096  baseline
  Experiment #  1: bpb=1.114434  reduce batch size 2^19->2^16, device 128->32 for more optimizer steps
  Experiment #  9: bpb=1.114312  reduce MATRIX_LR 0.04->0.02
  Experiment # 15: bpb=1.110488  RoPE base 10000->50000
  Experiment # 26: bpb=1.109812  S-layer window 1024->512 (long_window//4) for faster attention, more steps
  Experiment # 27: bpb=1.105889  S-layer window 512->256 (long_window//8) even more steps (1140)
  Experiment # 28: bpb=1.104112  S-layer window 256->128 (long_window//16) 1159 steps
  Experiment # 29: bpb=1.103181  S-layer window 128->64 (long_window//32) 1170 steps
  Experiment # 31: bpb=1.100676  WINDOW_PATTERN SSSL->SSSSL (L at pos 4,7 vs 3,7) 1176 steps
  Experiment # 33: bpb=1.099409  S-window 64->32 with SSSSL pattern (long_window//64) 1173 steps
  Experiment # 35: bpb=1.098821  DEPTH 8->7 (same dim=512, 47.2M params, 1304 steps)
  Experiment # 38: bpb=1.098446  S-window 48 (between 32 and 64) with DEPTH=7 SSSSL
  Experiment # 40: bpb=1.094826  WINDOW_PATTERN SSSSL->SSSL with DEPTH=7 S=48 (L at pos 3,6 vs 4,6) 1301 steps
  Experiment # 44: bpb=1.094283  logit softcap 15->12
  Experiment # 48: bpb=1.092723  MLP hidden 4x->2x (1024 hidden, 39.8M params, 1634 steps)
  Experiment # 52: bpb=1.091977  S-window 32 with 2x MLP DEPTH=7 SSSL softcap=12 (1639 steps)
  Experiment # 55: bpb=1.091906  SCALAR_LR 0.5->0.1 (slower residual scalar learning)
  Experiment # 58: bpb=1.091643  SSML pattern: 2xS=32 + M=256 + L=2048 (1629 steps)
  Experiment # 63: bpb=1.091288  EMBEDDING_LR 0.6->0.8
  Experiment # 66: bpb=1.090030  WINDOW_PATTERN SSML->SSMM (1L instead of 2L, 4S+2M+1L, 1674 steps)
  Experiment # 72: bpb=1.089408  Adam beta1 0.8->0.75 (more aggressive momentum)
  Experiment # 73: bpb=1.089332  Adam beta1 0.75->0.7 (even more aggressive momentum)
  Experiment # 80: bpb=1.089259  WARMDOWN_RATIO 0.5->0.55 (slightly longer warmdown)
  Experiment # 84: bpb=1.089244  RoPE base 50000->60000
```
## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.


## License

MIT
