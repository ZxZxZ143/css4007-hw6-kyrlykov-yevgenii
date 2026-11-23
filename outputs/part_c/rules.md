# Part C maps and rules

Coordinates start at 1 (row, column).

Original map:
```text
.M.G
.#IH
S...
```
Extended map:
```text
.M.G
.#IH
SD.W
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
