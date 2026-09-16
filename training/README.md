# E3 training: ACT on the relay dataset

## What the policy is (honest)

The installed `lerobot==0.4.4` ACT policy consumes images + measured
joint state and predicts absolute joint-action chunks. It has **no text
input**: no language encoder, no instruction slot, no task-conditioned
weights exist anywhere in its config or forward pass (verified by
inspecting the installed package: the training batch carries only
images, state, action, and padding mask).

So instruction handling lives **outside the network** in an explicit
router, and that is declared here rather than implied:

- The deployment grammar accepts exactly one in-grammar instruction
  (the relay sentence). The router maps it to this checkpoint and runs
  the rollout. Anything else is out-of-grammar: the router refuses, the
  arms never move, the episode scores as a failure.
- The swap-instruction control therefore runs at the harness level: the
  wrong sentence must produce refusal (success collapses to zero),
  which is what the architecture doc demands of the language path. The
  policy net itself is instruction-blind; the claim is about the system,
  and the eval harness measures exactly that.

A text-conditioned policy variant is E4+ scope, not claimed here.

## E3 action contract (from the dataset writer)

Per frame: `observation.state` (12 measured joints+jaws),
`action` (next frame's measured state), `observation.commanded`
(ctrl intent), `observation.jaw_forcerange` (authority caps).
Training fits measured-next from images + measured state.
`observation.commanded` / jaw authority stay available for analysis.

## How to run

- Smoke (~300 steps, in-lane):
  `.venv/bin/python training/act_mambo.py smoke`
- Full ladder (background, resume-safe):
  `nohup ./.venv/bin/python training/act_mambo.py full > out/training/full.log 2>&1 &`
  `echo $! > out/training/full.pid`
- Resume after interruption (reuses the checkpoint config):
  `.venv/bin/python training/act_mambo.py full --resume`
- Monitor: `tail -f out/training/full.log`, `nvidia-smi -l 2`
- Evaluate: `.venv/bin/python scripts/eval_policy.py --checkpoint out/checkpoints/act_full --seeds 0-9`

## Budgets (RTX 2060 SUPER, 8 GB)

Batch 4, chunk 50, AMP on. Smoke-train peak allocation is sampled with
`nvidia-smi` and must stay under 7.5 GB; on overflow, halve the batch
before touching anything else.
