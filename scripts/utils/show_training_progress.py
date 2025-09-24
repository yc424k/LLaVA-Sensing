#!/usr/bin/env python
"""Show Transformers Trainer progress as a simple progress bar."""

import argparse
import json
import math
from pathlib import Path


def load_trainer_state(output_dir: Path):
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"{state_path} not found. Run training first or specify correct --output_dir.")
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_bar(ratio: float, width: int = 40) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = math.floor(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Show remaining training progress as a progress bar")
    parser.add_argument("--output_dir", required=True, help="Trainer output directory (contains trainer_state.json)")
    args = parser.parse_args()

    state = load_trainer_state(Path(args.output_dir))
    global_step = state.get("global_step", 0)
    max_steps = state.get("max_steps", None)

    if max_steps is None or max_steps == 0:
        raise ValueError("max_steps missing in trainer_state.json; unable to compute progress")

    ratio = global_step / max_steps
    bar = render_bar(ratio)
    remaining = max_steps - global_step

    print(bar)
    print(f"Step {global_step}/{max_steps} (remaining {remaining})")


if __name__ == "__main__":
    main()
