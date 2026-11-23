"""Reproduce HW6 Part B without training or changing FrozenLake's rules."""

import argparse
import csv
import json
import platform
import re
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import gymnasium as gym
import numpy as np


# Written in SUBMISSION.md before experiments. Gymnasium: L=0, D=1, R=2, U=3.
FIXED_POLICY = {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 6: 1,
                8: 2, 9: 1, 10: 1, 13: 2, 14: 2}
ACTION_NAMES = ("left", "down", "right", "up")
BATCH_SEEDS = (11, 22, 33)
EPISODES_PER_BATCH = 50
MAX_STEPS = 100
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def run_episode(env, policy, episode_seed, max_steps=MAX_STEPS):
    """Return a result and text-rendered trace; terminal outcome takes priority."""
    state, _ = env.reset(seed=episode_seed)
    # Separate generator from environment dynamics; paired across slippery modes.
    action_rng = np.random.default_rng(episode_seed)
    trace = [f"policy={policy}, episode_seed={episode_seed}", env.render()]
    total_reward = 0.0
    outcome = "time limit"
    for steps in range(1, max_steps + 1):
        action = (int(action_rng.integers(4)) if policy == "random"
                  else FIXED_POLICY[int(state)])
        previous_state = int(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        trace.extend([
            f"step={steps}, state={previous_state}, action={ACTION_NAMES[action]}, "
            f"next_state={state}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}",
            env.render(),
        ])
        if terminated:
            tile = env.unwrapped.desc.flat[int(state)].decode("ascii")
            if tile not in ("G", "H"):
                raise AssertionError(f"Unexpected terminal tile: {tile}")
            outcome = "goal" if tile == "G" else "hole"
            break
        if truncated:
            break
    result = {
        "episode_seed": episode_seed,
        "policy": policy,
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
        "final_state": int(state),
    }
    trace.append(f"outcome={outcome}, steps={steps}, total_reward={total_reward}")
    return result, ANSI_ESCAPE.sub("", "\n".join(trace)) + "\n"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(exist_ok=True)
    episodes, summaries, trace_index = [], [], []
    for slippery in (False, True):
        for policy in ("random", "fixed"):
            condition = f"{'slippery' if slippery else 'deterministic'}_{policy}"
            saved_outcomes = set()
            env = gym.make("FrozenLake-v1", map_name="4x4",
                           is_slippery=slippery, render_mode="ansi",
                           max_episode_steps=MAX_STEPS)
            try:
                nonterminal = {i for i, tile in enumerate(env.unwrapped.desc.flat)
                               if tile not in (b"H", b"G")}
                assert set(FIXED_POLICY) == nonterminal
                for batch_seed in BATCH_SEEDS:
                    batch = []
                    for i in range(EPISODES_PER_BATCH):
                        episode_seed = 1000 * batch_seed + i
                        result, trace = run_episode(env, policy, episode_seed)
                        row = {"condition": condition, "slippery": slippery,
                               "batch_seed": batch_seed, **result}
                        episodes.append(row)
                        batch.append(row)
                        if result["outcome"] not in saved_outcomes:
                            saved_outcomes.add(result["outcome"])
                            name = f"{condition}_{result['outcome'].replace(' ', '_')}.txt"
                            (trace_dir / name).write_text(
                                f"condition={condition}, batch_seed={batch_seed}\n" + trace,
                                encoding="utf-8")
                            trace_index.append({**row, "file": f"traces/{name}"})
                    goals = sum(r["outcome"] == "goal" for r in batch)
                    holes = sum(r["outcome"] == "hole" for r in batch)
                    limits = sum(r["outcome"] == "time limit" for r in batch)
                    assert goals + holes + limits == EPISODES_PER_BATCH
                    summaries.append({
                        "slippery": slippery, "policy": policy,
                        "batch_seed": batch_seed, "goals": goals, "holes": holes,
                        "time_limits": limits, "success_percent": goals * 2,
                        "mean_steps": mean(r["steps"] for r in batch),
                    })
            finally:
                env.close()

    assert len(episodes) == 600 and len(summaries) == 12
    for row in episodes:
        assert 1 <= row["steps"] <= MAX_STEPS
        assert row["total_reward"] == (1.0 if row["outcome"] == "goal" else 0.0)
        assert row["outcome"] != "time limit" or row["steps"] == MAX_STEPS
        if row["condition"] == "deterministic_fixed":
            assert row["outcome"] == "goal" and row["steps"] == 6
    seed_lists = [tuple(r["episode_seed"] for r in episodes if r["condition"] == c)
                  for c in dict.fromkeys(r["condition"] for r in episodes)]
    assert all(seeds == seed_lists[0] for seeds in seed_lists)

    write_csv(output_dir / "episodes.csv", episodes)
    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "trace_index.json").write_text(
        json.dumps(trace_index, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in (
            "gymnasium", "numpy", "cloudpickle", "typing_extensions", "farama-notifications")},
        "environment": "FrozenLake-v1", "map_name": "4x4",
        "map": ["SFFF", "FHFH", "FFFH", "HFFG"],
        "reward_and_slip_parameters": "defaults, not overridden",
        "batch_seeds": BATCH_SEEDS, "episodes_per_batch": EPISODES_PER_BATCH,
        "max_steps": MAX_STEPS, "episode_seed_rule": "1000 * batch_seed + i, i=0..49",
        "random_actions": "numpy.random.default_rng(episode_seed).integers(4)",
        "fixed_policy": {str(s): ACTION_NAMES[a] for s, a in FIXED_POLICY.items()},
        "checks": "600 rows; 12 batches; paired seeds; steps/rewards/outcomes; "
                  "all 150 deterministic fixed episodes reach goal in 6 steps: passed",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Slippery | Policy | Seed | Goals | Holes | Limits | Success % | Mean steps")
    for row in summaries:
        print(" | ".join(str(value) for value in row.values()))
    print(f"Saved {len(episodes)} episodes and {len(trace_index)} traces to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "part_b")
    experiment(parser.parse_args().output_dir)
