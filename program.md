# autoresearch

This is an experiment to have the LLM do its own research, in order to compete in a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

Experiments will be conducted using MLX (`train_gpt_mlx.py`) on a MacBook Pro with an M1 Max and 64 GB of RAM. Promising directions will later be run by the user on NVIDIA GPUs using PyTorch (`train_gpt.py`).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `./README.md` — repository context.
   - `./data/README.md` — data prep, tokenizer, dataloader, evaluation.
   - `train_gpt.py` — PyTorch-based. Model architecture, optimizer, training loop. Do not modify.
   - `train_gpt_mlx.py` — MLX-based. The only file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `./data` contains data shards and a tokenizer. If not, tell the human to run `uv run data/cached_challenge_fineweb.py --variant sp1024`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train_gpt_mlx.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `train_gpt.py`. It is read-only. It contains the PyTorch baseline implementation and is only to be modified by the user.
- Modify any file in `./data`. It is read-only. It contains the data loading, tokenizer, and some other metadata.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness in `train_gpt_mlx.py`.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. MLX uses unified memory shared between CPU and GPU. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          2.534000
training_seconds: 312.4
total_seconds:    405.7
peak_vram_mb:     27528.9
mfu_percent:      0.00
total_tokens_M:   39.8
num_steps:        46
num_params_M:     50.3
depth:            8
```

Note that the script runs for a fixed 5-minute training budget. On Apple Silicon the throughput, step count, and absolute val_bpb will differ from NVIDIA results — that's expected. Compare only against your own baseline on the same hardware.

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
383abb4	2.667000	26.9	keep	baseline
909dd59	2.588904	26.9	keep	halve total batch size to 2^16
4161af3	2.533728	26.9	keep	increase matrix LR to 0.04
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_gpt_mlx.py` with an experimental idea by directly hacking the code.
3. `git add autoresearch-mlx/train_gpt_mlx.py && git commit -m "experiment: <description>"` (never `git add -A` — this may be inside a larger repo)
4. Run the experiment: `uv run train_gpt_mlx.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), `git add autoresearch-mlx/results.tsv && git commit --amend --no-edit` to include the log, advancing the branch
9. If val_bpb is equal or worse, record the discard commit hash, then `git reset --hard <previous kept commit>` to discard it cleanly

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~7 minutes total (5 min training + ~1 min compile/eval overhead on Apple Silicon). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~7 minutes then you can run approx 8-9/hour, for a total of about 70 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!