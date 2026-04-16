#!/usr/bin/env python3
"""Play the local Sokoban environment as a human."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable


# Matplotlib tries to write under ~/.config by default, which may be read-only
# in shared or remote environments. Use a writable cache dir unless the user
# already chose one explicitly.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from agent_system.environments.env_package.sokoban.sokoban.env import SokobanEnv


ACTION_BY_NAME = {
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
}

TEXT_COMMANDS = {
    "w": ACTION_BY_NAME["up"],
    "s": ACTION_BY_NAME["down"],
    "a": ACTION_BY_NAME["left"],
    "d": ACTION_BY_NAME["right"],
}


@dataclass
class StepResult:
    reward: float
    done: bool
    effective: bool
    won: bool


class HumanSokoban:
    def __init__(
        self,
        dim_room: tuple[int, int],
        num_boxes: int,
        max_steps: int,
        search_depth: int,
        seed: int | None,
    ) -> None:
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.max_steps = max_steps
        self.search_depth = search_depth
        self.seed = seed
        self.steps_taken = 0
        self.env = self._build_env(mode="rgb_array")
        self.last_rgb, _ = self.env.reset(seed=self.seed)

    def _build_env(self, mode: str) -> SokobanEnv:
        return SokobanEnv(
            mode,
            dim_room=self.dim_room,
            num_boxes=self.num_boxes,
            max_steps=self.max_steps,
            search_depth=self.search_depth,
        )

    def reset(self) -> None:
        self.steps_taken = 0
        self.last_rgb, _ = self.env.reset(seed=self.seed)

    def rgb(self):
        return self.env.render("rgb_array")

    def text(self) -> str:
        return self.env.render("tiny_rgb_array")

    def step(self, action: int) -> StepResult:
        obs, reward, done, info = self.env.step(action)
        self.last_rgb = obs
        self.steps_taken += 1
        return StepResult(
            reward=reward,
            done=done,
            effective=info.get("action_is_effective", True),
            won=info.get("won", False),
        )

    def status_line(self, result: StepResult | None = None) -> str:
        base = (
            f"steps={self.steps_taken}/{self.max_steps} "
            f"boxes={self.env.boxes_on_target}/{self.env.num_boxes}"
        )
        if result is None:
            return base
        return (
            f"{base} reward={result.reward:.2f} "
            f"effective={result.effective} won={result.won}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play the local Sokoban environment as a human."
    )
    parser.add_argument(
        "--train-defaults",
        action="store_true",
        help="Match the repository's Sokoban RL training difficulty.",
    )
    parser.add_argument(
        "--ui",
        choices=("gui", "text"),
        default="gui",
        help="Interaction mode. Use text mode over SSH or without a display.",
    )
    parser.add_argument("--rows", type=int, default=6, help="Room height.")
    parser.add_argument("--cols", type=int, default=6, help="Room width.")
    parser.add_argument("--num-boxes", type=int, default=1, help="Number of boxes.")
    parser.add_argument("--max-steps", type=int, default=100, help="Episode limit.")
    parser.add_argument(
        "--search-depth",
        type=int,
        default=100,
        help="Puzzle generation search depth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed seed for repeatable levels. Omit for random levels.",
    )
    return parser.parse_args()


def run_text_ui(game: HumanSokoban) -> int:
    print("Text Sokoban")
    print("controls: w/a/s/d move, r reset, q quit")
    print()
    print(game.text())
    print(game.status_line())

    while True:
        raw = input("> ").strip().lower()
        if not raw:
            continue
        if raw == "q":
            return 0
        if raw == "r":
            game.reset()
            print()
            print(game.text())
            print(game.status_line())
            continue
        action = TEXT_COMMANDS.get(raw)
        if action is None:
            print("unknown command; use w/a/s/d, r, q")
            continue

        result = game.step(action)
        print()
        print(game.text())
        print(game.status_line(result))
        if result.won:
            print("Solved. Press r to restart or q to quit.")
        elif result.done:
            print("Episode ended. Press r to restart or q to quit.")


def run_gui(game: HumanSokoban) -> int:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        print(f"Failed to import matplotlib: {exc}", file=sys.stderr)
        print("Use --ui text instead.", file=sys.stderr)
        return 1

    if not os.environ.get("DISPLAY") and sys.platform != "darwin":
        print("No DISPLAY detected. Use --ui text over SSH or in headless shells.", file=sys.stderr)
        return 1

    key_to_action: dict[str, int] = {
        "up": ACTION_BY_NAME["up"],
        "down": ACTION_BY_NAME["down"],
        "left": ACTION_BY_NAME["left"],
        "right": ACTION_BY_NAME["right"],
    }

    fig, ax = plt.subplots()
    image = ax.imshow(game.rgb())
    ax.set_axis_off()

    def update_title(result: StepResult | None = None) -> None:
        fig.suptitle(
            "Sokoban | arrows move | r reset | q/esc quit\n" + game.status_line(result),
            fontsize=11,
        )

    def redraw(result: StepResult | None = None) -> None:
        image.set_data(game.rgb())
        update_title(result)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        key = (event.key or "").lower()
        if key in {"q", "escape"}:
            plt.close(fig)
            return
        if key == "r":
            game.reset()
            redraw()
            return
        action = key_to_action.get(key)
        if action is None:
            return
        result = game.step(action)
        redraw(result)
        if result.won:
            print("Solved. Press r to restart or q to quit.")
        elif result.done:
            print("Episode ended. Press r to restart or q to quit.")

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()
    return 0


def main() -> int:
    args = parse_args()
    if args.train_defaults:
        args.rows = 6
        args.cols = 6
        args.num_boxes = 1
        args.max_steps = 15
        args.search_depth = 30

    game = HumanSokoban(
        dim_room=(args.rows, args.cols),
        num_boxes=args.num_boxes,
        max_steps=args.max_steps,
        search_depth=args.search_depth,
        seed=args.seed,
    )
    runner: Callable[[HumanSokoban], int] = run_gui if args.ui == "gui" else run_text_ui
    return runner(game)


if __name__ == "__main__":
    raise SystemExit(main())
