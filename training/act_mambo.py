"""ACT smoke/full trainer for the relay dataset (lerobot 0.4.4 API).

Why a code launcher instead of CLI flags: the policy feature maps
(input images + state, output action) need structured values that are
painful through draccus CLI overrides, and the smoke/full ladders share
everything except budgets. No third-party code is copied here; only the
installed package API is called.

Usage:
  .venv/bin/python training/act_mambo.py smoke [--steps N]
  .venv/bin/python training/act_mambo.py full [--resume]

Outputs: out/checkpoints/act_smoke/ and out/checkpoints/act_full/.
"""

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

REPO_ID = "mambo/rc-dinner"
DATASET_ROOT = ROOT / "out" / "lerobot"

INPUT_FEATURES = {
    f"observation.images.{cam}": PolicyFeature(FeatureType.VISUAL, (256, 256, 3))
    for cam in ("overhead", "table_left", "table_right", "wrist_cam")
} | {
    "observation.state": PolicyFeature(FeatureType.STATE, (12,)),
}
OUTPUT_FEATURES = {
    "action": PolicyFeature(FeatureType.ACTION, (12,)),
}

# Extra dataset streams (tool sites, jaw authority, commanded intent) are
# deliberately NOT policy inputs: the training policy sees what the
# deployment policy will see (images + measured joint state). The extra
# streams stay in the dataset for analysis and for E3 contract checks.


def build_config(mode: str, steps_override: int | None, resume: bool) -> TrainPipelineConfig:
    smoke = mode == "smoke"
    policy = ACTConfig(
        input_features=dict(INPUT_FEATURES),
        output_features=dict(OUTPUT_FEATURES),
        device="cuda",
        push_to_hub=False,
        chunk_size=50,
        n_action_steps=50,
        use_amp=True,
    )
    dataset = DatasetConfig(repo_id=REPO_ID, root=str(DATASET_ROOT))
    return TrainPipelineConfig(
        dataset=dataset,
        env=None,
        policy=policy,
        output_dir=ROOT / "out" / "checkpoints" / ("act_smoke" if smoke else "act_full"),
        resume=resume,
        seed=1000,
        num_workers=4,
        batch_size=4,
        steps=steps_override if steps_override is not None else (300 if smoke else 200_000),
        eval_freq=20_000,
        log_freq=25 if smoke else 200,
        save_checkpoint=True,
        save_freq=150 if smoke else 20_000,
    )


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    assert mode in ("smoke", "full"), "usage: act_mambo.py smoke|full [--resume] [--steps N]"
    resume = "--resume" in sys.argv
    steps_override = None
    for i, arg in enumerate(sys.argv):
        if arg == "--steps" and i + 1 < len(sys.argv):
            steps_override = int(sys.argv[i + 1])
    cfg = build_config(mode, steps_override, resume)
    print(f"mode={mode} steps={cfg.steps} batch={cfg.batch_size} "
          f"chunk={cfg.policy.chunk_size} out={cfg.output_dir} resume={resume}", flush=True)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
