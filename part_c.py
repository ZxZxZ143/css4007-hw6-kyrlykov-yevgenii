"""Solve both HW6 maps, log every iteration, check rules, and simulate policies."""

import argparse
import csv
import json
import math
import platform
import random
from pathlib import Path
from statistics import mean

import numpy as np

from grid_world import ARROWS, GridWorld, coordinate, load_config


TIE_TOLERANCE = 1e-12


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def describe_rules(config):
    return """# Part C maps and rules

Coordinates start at 1 (row, column).

Original map:
```text
%s
```
Extended map:
```text
%s
```

S is the start; . is floor; M is mud; I is ice; G is goal;
H is a hole; # is an impassable wall; D is sand; W is shallow water.
The wall is not a state. Actions: up, right, down, left.
Rewards are paid on arrival: S/. /I = -1, M = -3, D = -2, W = -5,
G = +10, H = -10. Entering G/H ends the episode, without extra floor cost.
G/H have zero value and no rewards after termination.

Moves from S, ., M, D, W are deterministic. Only moves STARTING on I slip:
0.8 intended direction, 0.1 each perpendicular direction. For each branch,
a wall or boundary leaves the agent in place and pays that destination's reward.
Entering ice from ordinary floor does not itself cause slipping.

Discount gamma = 0.9; initialize all states at V0 = 0. Each complete update
uses only the previous round. Stop when max_s |Vnew(s)-Vold(s)| < 0.000001;
stop after at most 10000 updates and explicitly report nonconvergence if needed.
Extract actions from FINAL values, maximizing expected discounted reward.
Numerical ties within 1e-12 use up, right, down, left, in that order.

This objective is not the same as maximizing goal probability. All actions
and their stochastic outcomes are included in every Bellman update. Full
trajectory enumeration is infinite because cycles and repeated slips are possible.
Goal/hole probabilities are evaluated separately for the resulting policy.
Simulations: 3 batches of 50 episodes, seeds 11/22/33, episode seed
1000*batch_seed+i for i=0..49, 100-step limit, terminal outcome priority.
""" % ("\n".join(config["paper_map"]), "\n".join(config["extended_map"]))


def solve(world, output_dir):
    tolerance = world.config["solver"]["tolerance"]
    maximum = world.config["solver"]["max_iterations"]
    values = {state: 0.0 for state in world.states}
    initial_rounds = {0: values.copy()}
    log_path = output_dir / "iterations.txt"
    with log_path.open("w", encoding="utf-8") as log, \
            (output_dir / "iterations.csv").open("w", encoding="utf-8", newline="") as csv_stream:
        writer = csv.writer(csv_stream)
        writer.writerow(["iteration", "row", "column", "tile", "value", "delta"])

        def record(iteration, current, delta):
            block = (f"{world.name}: V{iteration}; delta={delta:.12g}\n"
                     + world.value_map(current) + "\n")
            print(block, flush=True)
            if iteration:
                log.write("\n")
            log.write(block)
            for state in world.states:
                writer.writerow([iteration, state[0] + 1, state[1] + 1,
                                 world.tile(state), current[state], delta])
            log.flush()
            csv_stream.flush()

        record(0, values, 0.0)
        for iteration in range(1, maximum + 1):
            updated = {state: (0.0 if world.terminal(state) else
                               max(world.q(state, action, values) for action in world.actions))
                       for state in world.states}
            delta = max(abs(updated[state] - values[state]) for state in world.states)
            values = updated
            if iteration <= 2:
                initial_rounds[iteration] = values.copy()
            record(iteration, values, delta)
            if delta < tolerance:
                break

    policy, ties, q_rows = {}, {}, []
    for state in world.nonterminal:
        scores = {action: world.q(state, action, values) for action in world.actions}
        best = max(scores.values())
        tied = [action for action in world.actions if abs(scores[action] - best) <= TIE_TOLERANCE]
        policy[state] = tied[0]
        ties[coordinate(state)] = tied
        q_rows.append({"row": state[0] + 1, "column": state[1] + 1,
                       "tile": world.tile(state), **scores, "chosen_action": tied[0],
                       "tied_actions": ",".join(tied)})
    residual = max(abs(values[state] - max(world.q(state, a, values) for a in world.actions))
                   for state in world.nonterminal)
    info = {"iterations": iteration, "final_delta": delta,
            "converged": delta < tolerance, "tolerance": tolerance,
            "max_iterations": maximum, "bellman_residual": residual,
            "value_error_bound": residual / (1 - world.discount),
            "start_value": values[world.start],
            "tie_rule": "up, right, down, left; absolute tolerance 1e-12",
            "ties": ties}
    write_json(output_dir / "convergence.json", info)
    write_csv(output_dir / "final_q.csv", q_rows)
    (output_dir / "values.txt").write_text(world.value_map(values) + "\n", encoding="utf-8")
    (output_dir / "policy.txt").write_text(world.policy_map(policy) + "\n", encoding="utf-8")
    print(f"{world.name}: converged={info['converged']}, updates={iteration}, delta={delta:.12g}")
    print("Final policy:\n" + world.policy_map(policy), flush=True)
    return values, policy, initial_rounds, info


def evaluate_policy(world, policy):
    """Linear-system evaluation independent of the Bellman iteration.

    For these proper policies I-P is invertible; no claim that arbitrary
    nonterminating policies have finite expected termination time is made.
    """
    states = world.nonterminal
    indices = {state: i for i, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)))
    rewards = np.zeros(len(states))
    goals = np.zeros(len(states))
    holes = np.zeros(len(states))
    for state, index in indices.items():
        for probability, destination, reward in world.transitions(state, policy[state]):
            rewards[index] += probability * reward
            if destination in indices:
                transition[index, indices[destination]] += probability
            elif world.tile(destination) == "G":
                goals[index] += probability
            else:
                holes[index] += probability
    identity = np.eye(len(states))
    exact_values = np.linalg.solve(identity - world.discount * transition, rewards)
    goal_probability = np.linalg.solve(identity - transition, goals)
    hole_probability = np.linalg.solve(identity - transition, holes)
    expected_steps = np.linalg.solve(identity - transition, np.ones(len(states)))
    assert np.allclose(goal_probability + hole_probability, 1.0, atol=1e-12)
    return [{"row": state[0] + 1, "column": state[1] + 1,
             "value": float(exact_values[i]), "goal_probability": float(goal_probability[i]),
             "hole_probability": float(hole_probability[i]),
             "expected_steps": float(expected_steps[i])} for i, state in enumerate(states)]


def nominal_route(world, policy):
    """Illustrate intended directions only; not a promise of a stochastic path."""
    state = world.start
    route = [coordinate(state)]
    visited = set()
    while not world.terminal(state):
        if state in visited:
            route.append("cycle in nominal route")
            break
        visited.add(state)
        action = policy[state]
        # transitions() lists the intended movement first.
        destination = world.transitions(state, action)[0][1]
        route.append(f"--{action}--> {coordinate(destination)}")
        state = destination
    return " ".join(route)


def check_rules(world, values, policy, rounds):
    checks = []

    def check(label, condition, detail):
        checks.append({"check": label, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{world.name}: {label}: {detail}")

    check("wall is not a state", (1, 1) not in world.state_set, "(2,2) excluded")
    check("boundary", world.transitions((0, 0), "up") == [(1.0, (0, 0), -1.0)],
          "(1,1) up -> (1,1), reward -1")
    check("wall from mud", world.transitions((0, 1), "down") == [(1.0, (0, 1), -3.0)],
          "(1,2) down -> (1,2), reward -3")
    ice = world.transitions((1, 2), "up")
    check("ice outcomes", ice == [(0.8, (0, 2), -1.0), (0.1, (1, 2), -1.0),
                                  (0.1, (1, 3), -10.0)],
          "up: 0.8 floor; 0.1 blocked by wall and remain on ice; 0.1 hole")
    check("probabilities sum to one", all(math.isclose(sum(p for p, _, _ in world.transitions(s, a)), 1.0)
          for s in world.states for a in world.actions), "checked every state/action")
    check("ice entry deterministic", world.transitions((0, 2), "down") == [(1.0, (1, 2), -1.0)],
          "starting on ordinary floor: one outcome even when destination is ice")
    check("mud entry reward", world.transitions((0, 0), "right") == [(1.0, (0, 1), -3.0)],
          "enter M: -3")
    check("goal entry reward", world.transitions((0, 2), "right") == [(1.0, (0, 3), 10.0)],
          "enter G: +10 without extra floor cost")
    check("hole entry reward", world.transitions((2, 3), "up") == [(1.0, (1, 3), -10.0)],
          "enter H: -10")
    check("no post-terminal rewards", all(world.transitions(s, a) == [(1.0, s, 0.0)]
          for s in world.states if world.terminal(s) for a in world.actions),
          "all actions from G/H self-loop with zero reward")
    check("terminal values zero", all(values[s] == 0 for s in world.states if world.terminal(s)),
          "V(G)=V(H)=0")
    if world.name == "paper_map":
        check("V1 matches paper", all(math.isclose(rounds[1][s], 10 if s == (0, 2) else -1)
              for s in world.nonterminal), "all nine nonterminal cells")
        for state, expected in [((0, 1), 8), ((0, 2), 10), ((1, 2), 5.21)]:
            check("V2 " + coordinate(state), math.isclose(rounds[2][state], expected, abs_tol=1e-12),
                  f"actual={rounds[2][state]}, paper={expected}")
    else:
        check("sand reward and motion", world.transitions((2, 0), "right") == [(1.0, (2, 1), -2.0)]
              and world.transitions((2, 1), "up") == [(1.0, (2, 1), -2.0)]
              and world.transitions((2, 1), "right") == [(1.0, (2, 2), -1.0)],
              "enter D=-2; blocked stay on D=-2; leaving D is deterministic")
        check("water reward and motion", world.transitions((2, 2), "right") == [(1.0, (2, 3), -5.0)]
              and world.transitions((2, 3), "down") == [(1.0, (2, 3), -5.0)]
              and world.transitions((2, 3), "left") == [(1.0, (2, 2), -1.0)],
              "enter W=-5; blocked stay on W=-5; leaving W is deterministic")
    evaluation = evaluate_policy(world, policy)
    difference = max(abs(values[(row["row"] - 1, row["column"] - 1)] - row["value"])
                     for row in evaluation)
    check("independent fixed-policy value", difference < 1e-5,
          f"max difference from direct linear-system solution={difference:.12g}")
    return checks, evaluation


def simulate(world, policy, output_dir):
    episodes, summaries, steps_rows = [], [], []
    settings = world.config["simulation"]
    for batch_seed in settings["batch_seeds"]:
        batch = []
        for i in range(settings["episodes_per_batch"]):
            seed = 1000 * batch_seed + i
            rng = random.Random(seed)
            state = world.start
            total_reward = discounted_return = 0.0
            outcome = "time limit"
            for step in range(1, settings["max_steps"] + 1):
                action = policy[state]
                outcomes = world.transitions(state, action)
                draw, cumulative = rng.random(), 0.0
                destination, reward = outcomes[-1][1:]
                for probability, candidate, candidate_reward in outcomes:
                    cumulative += probability
                    if draw < cumulative:
                        destination, reward = candidate, candidate_reward
                        break
                total_reward += reward
                discounted_return += world.discount ** (step - 1) * reward
                steps_rows.append({"map": world.name, "batch_seed": batch_seed,
                    "episode_seed": seed, "step": step, "state": coordinate(state),
                    "action": action, "next_state": coordinate(destination), "reward": reward})
                state = destination
                if world.terminal(state):
                    outcome = "goal" if world.tile(state) == "G" else "hole"
                    break
            row = {"map": world.name, "batch_seed": batch_seed, "episode_seed": seed,
                   "steps": step, "total_reward": total_reward,
                   "discounted_return": discounted_return, "outcome": outcome,
                   "final_state": coordinate(state)}
            batch.append(row)
            episodes.append(row)
        counts = {outcome: sum(row["outcome"] == outcome for row in batch)
                  for outcome in ("goal", "hole", "time limit")}
        summaries.append({"map": world.name, "batch_seed": batch_seed,
            "goals": counts["goal"], "holes": counts["hole"],
            "time_limits": counts["time limit"], "mean_steps": mean(r["steps"] for r in batch),
            "mean_discounted_return": mean(r["discounted_return"] for r in batch)})
    write_csv(output_dir / "episodes.csv", episodes)
    write_csv(output_dir / "episode_steps.csv", steps_rows)
    return episodes, summaries


def main(output_dir):
    config = load_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = describe_rules(config)
    (output_dir / "rules.md").write_text(rules, encoding="utf-8")
    print(rules, flush=True)
    worlds = {name: GridWorld(config, name) for name in ("paper_map", "extended_map")}
    solved = {}
    # First solve the original map, then extend and solve, before final checks/runs.
    for name, world in worlds.items():
        folder = output_dir / name
        folder.mkdir(exist_ok=True)
        solved[name] = solve(world, folder)

    summaries, checks_by_map, comparison_rows = [], {}, []
    for name, world in worlds.items():
        values, policy, rounds, info = solved[name]
        folder = output_dir / name
        checks, evaluation = check_rules(world, values, policy, rounds)
        checks_by_map[name] = checks
        write_json(folder / "rule_checks.json", checks)
        write_csv(folder / "policy_evaluation.csv", evaluation)
        transition_rows = [{"state": coordinate(state), "action": action,
                            "probability": p, "next_state": coordinate(destination), "reward": reward}
                           for state in world.states for action in world.actions
                           for p, destination, reward in world.transitions(state, action)]
        write_csv(folder / "transitions.csv", transition_rows)
        if name == "paper_map":
            paper_rows = [{"state": coordinate(state), "action": action,
                           "Q_using_V1": world.q(state, action, rounds[1]),
                           "V2": rounds[2][state]}
                          for r, c in config["paper_v2_positions"] for state in [(r - 1, c - 1)]
                          for action in world.actions]
            write_csv(folder / "paper_comparison.csv", paper_rows)

        # A guaranteed safe route helps distinguish reward optimality from reachability.
        safe_policy = policy.copy()
        safe_policy.update({(2, 0): "up", (1, 0): "up", (0, 0): "right",
                            (0, 1): "right", (0, 2): "right"})
        alternatives = []
        for label, candidate in [("reward_optimal", policy), ("safe_upper_route", safe_policy)]:
            evaluated = evaluate_policy(world, candidate)
            start = next(row for row in evaluated if (row["row"], row["column"]) == (3, 1))
            alternatives.append({"strategy": label, "nominal_route": nominal_route(world, candidate), **start})
        write_json(folder / "route_comparison.json", alternatives)
        (folder / "nominal_route.txt").write_text(nominal_route(world, policy) + "\n", encoding="utf-8")
        if not info["converged"]:
            print(f"WARNING: {name} did not converge; simulations use the last greedy policy.")
        _, batch_summaries = simulate(world, policy, folder)
        summaries.extend(batch_summaries)
        print(f"{name}: {len(checks)} rule/value checks passed; routes="
              + json.dumps(alternatives, ensure_ascii=False), flush=True)

    original_values, original_policy, _, _ = solved["paper_map"]
    extended_values, extended_policy, _, _ = solved["extended_map"]
    for state in worlds["paper_map"].states:
        comparison_rows.append({"state": coordinate(state),
            "original_tile": worlds["paper_map"].tile(state),
            "extended_tile": worlds["extended_map"].tile(state),
            "original_value": original_values[state], "extended_value": extended_values[state],
            "value_change": extended_values[state] - original_values[state],
            "original_action": original_policy.get(state, "terminal"),
            "extended_action": extended_policy.get(state, "terminal")})
    write_csv(output_dir / "map_comparison.csv", comparison_rows)
    write_csv(output_dir / "summary.csv", summaries)
    write_json(output_dir / "metadata.json", {"python": platform.python_version(),
        "numpy": np.__version__, "config_source": "data/maps.json", "config": config,
        "simulation_rng": "random.Random(episode_seed); one draw per step",
        "solver": "synchronous value iteration implemented in part_c.py",
        "objective": "expected discounted reward; goal probability evaluated separately",
        "rule_checks": {name: len(checks) for name, checks in checks_by_map.items()}})
    print("Batch summaries:\n" + json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs" / "part_c")
    args = parser.parse_args()
    main(args.output_dir)
