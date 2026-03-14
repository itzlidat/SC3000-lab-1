import json
import heapq
import math
import random


# =========================================================
# Part 1: Graph Search
# =========================================================

def load_graph():
    with open("G.json", "r") as f:
        G_raw = json.load(f)
    with open("Dist.json", "r") as f:
        Dist_raw = json.load(f)
    with open("Cost.json", "r") as f:
        Cost_raw = json.load(f)
    with open("Coord.json", "r") as f:
        Coord_raw = json.load(f)

    G = {int(k): [int(x) for x in v] for k, v in G_raw.items()}

    Dist = {}
    for k, val in Dist_raw.items():
        u, v = k.split(",")
        Dist[(int(u), int(v))] = float(val)

    Cost = {}
    for k, val in Cost_raw.items():
        u, v = k.split(",")
        Cost[(int(u), int(v))] = float(val)

    Coord = {}
    for k, val in Coord_raw.items():
        Coord[int(k)] = (float(val[0]), float(val[1]))

    return G, Dist, Cost, Coord


def reconstruct_path(parent, start, goal):
    if start == goal:
        return [start]

    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def reconstruct_label_path(labels, label_id):
    path = []
    cur_id = label_id

    while cur_id is not None:
        label = labels[cur_id]
        path.append(label["node"])
        cur_id = label["parent"]

    path.reverse()
    return path


def path_sum(path, weight):
    total = 0.0
    for i in range(len(path) - 1):
        total += weight[(path[i], path[i + 1])]
    return total


def print_result(task_name, path, total_dist, total_energy):
    print(task_name)
    if not path:
        print("No path found.")
    else:
        print("Shortest path:", "->".join(map(str, path)) + ".")
        print(f"Shortest distance: {total_dist}.")
        print(f"Total energy cost: {total_energy}.")
    print()


# ---------------- Task 1 ----------------
# Relaxed shortest path: no energy constraint
def dijkstra_distance(G, Dist, start, goal):
    dist_to = {start: 0.0}
    parent = {}
    pq = [(0.0, start)]  # (distance, node)

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist_to.get(u, float("inf")):
            continue

        if u == goal:
            path = reconstruct_path(parent, start, goal)
            return path, d

        for v in G.get(u, []):
            w = Dist.get((u, v))
            if w is None:
                continue

            nd = d + w
            if nd < dist_to.get(v, float("inf")):
                dist_to[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return [], float("inf")


# ---------------- Shared helper for Tasks 2 and 3 ----------------
# Keeps non-dominated labels at each node
def is_dominated(active_labels, labels, node, new_dist, new_energy):
    to_remove = []

    for old_id in active_labels.get(node, []):
        old_label = labels[old_id]
        old_dist = old_label["dist"]
        old_energy = old_label["energy"]

        # Old label dominates new label
        if old_dist <= new_dist and old_energy <= new_energy:
            return True, []

        # New label dominates old label
        if new_dist <= old_dist and new_energy <= old_energy:
            to_remove.append(old_id)

    return False, to_remove


# ---------------- Task 2 ----------------
# Uninformed search with energy constraint: UCS
def constrained_ucs(G, Dist, Cost, start, goal, budget):
    labels = {
        0: {
            "node": start,
            "dist": 0.0,
            "energy": 0.0,
            "parent": None
        }
    }

    active = {start: [0]}
    next_label_id = 1

    pq = [(0.0, 0.0, start, 0)]  # (distance, energy, node, label_id)

    while pq:
        cur_dist, cur_energy, u, label_id = heapq.heappop(pq)

        if label_id not in active.get(u, []):
            continue

        if u == goal:
            path = reconstruct_label_path(labels, label_id)
            return path, cur_dist, cur_energy

        for v in G.get(u, []):
            edge_dist = Dist.get((u, v))
            edge_cost = Cost.get((u, v))

            if edge_dist is None or edge_cost is None:
                continue

            new_dist = cur_dist + edge_dist
            new_energy = cur_energy + edge_cost

            if new_energy > budget:
                continue

            dominated, to_remove = is_dominated(active, labels, v, new_dist, new_energy)
            if dominated:
                continue

            if v not in active:
                active[v] = []

            for old_id in to_remove:
                active[v].remove(old_id)

            labels[next_label_id] = {
                "node": v,
                "dist": new_dist,
                "energy": new_energy,
                "parent": label_id
            }

            active[v].append(next_label_id)
            heapq.heappush(pq, (new_dist, new_energy, v, next_label_id))
            next_label_id += 1

    return [], float("inf"), float("inf")


# ---------------- Task 3 ----------------
# A* with Euclidean heuristic and energy constraint
def astar_with_energy_budget(G, Dist, Cost, Coord, start, goal, budget):
    def heuristic(node):
        x1, y1 = Coord[node]
        x2, y2 = Coord[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    labels = {
        0: {
            "node": start,
            "dist": 0.0,
            "energy": 0.0,
            "parent": None
        }
    }

    active = {start: [0]}
    next_label_id = 1

    pq = [(heuristic(start), 0.0, 0.0, start, 0)]  # (f, g, energy, node, label_id)

    while pq:
        f, cur_dist, cur_energy, u, label_id = heapq.heappop(pq)

        if label_id not in active.get(u, []):
            continue

        if u == goal:
            path = reconstruct_label_path(labels, label_id)
            return path, cur_dist, cur_energy

        for v in G.get(u, []):
            edge_dist = Dist.get((u, v))
            edge_cost = Cost.get((u, v))

            if edge_dist is None or edge_cost is None:
                continue

            new_dist = cur_dist + edge_dist
            new_energy = cur_energy + edge_cost

            if new_energy > budget:
                continue

            dominated, to_remove = is_dominated(active, labels, v, new_dist, new_energy)
            if dominated:
                continue

            if v not in active:
                active[v] = []

            for old_id in to_remove:
                active[v].remove(old_id)

            labels[next_label_id] = {
                "node": v,
                "dist": new_dist,
                "energy": new_energy,
                "parent": label_id
            }

            active[v].append(next_label_id)
            heapq.heappush(
                pq,
                (new_dist + heuristic(v), new_dist, new_energy, v, next_label_id)
            )
            next_label_id += 1

    return [], float("inf"), float("inf")


# =========================================================
# Part 2: Grid World MDP / RL
# =========================================================

ACTIONS = ["U", "R", "D", "L"]
ARROWS = {"U": "↑", "R": "→", "D": "↓", "L": "←"}

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
ROADBLOCKS = {(2, 1), (2, 3)}

GAMMA = 0.9
THETA = 1e-10
EPSILON = 0.1
ALPHA = 0.1


def get_states():
    return [
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if (x, y) not in ROADBLOCKS
    ]


def in_bounds(state):
    x, y = state
    return (
        0 <= x < GRID_SIZE
        and 0 <= y < GRID_SIZE
        and state not in ROADBLOCKS
    )


def move(state, action):
    if state == GOAL:
        return GOAL

    x, y = state

    if action == "U":
        nxt = (x, y + 1)
    elif action == "D":
        nxt = (x, y - 1)
    elif action == "L":
        nxt = (x - 1, y)
    else:  # "R"
        nxt = (x + 1, y)

    if not in_bounds(nxt):
        return state

    return nxt


def perpendicular_actions(action):
    if action in ("U", "D"):
        return ["L", "R"]
    return ["U", "D"]


def get_transitions(state, action):
    if state == GOAL:
        return [(1.0, GOAL, 0.0)]

    outcomes = {}

    candidates = [
        (0.8, action),
        (0.1, perpendicular_actions(action)[0]),
        (0.1, perpendicular_actions(action)[1]),
    ]

    for prob, actual_action in candidates:
        next_state = move(state, actual_action)
        reward = 10.0 if next_state == GOAL else -1.0
        key = (next_state, reward)
        outcomes[key] = outcomes.get(key, 0.0) + prob

    return [(p, s2, r) for (s2, r), p in outcomes.items()]


def env_step(state, action, rng):
    if state == GOAL:
        return GOAL, 0.0, True

    r = rng.random()

    if r < 0.8:
        actual_action = action
    elif r < 0.9:
        actual_action = perpendicular_actions(action)[0]
    else:
        actual_action = perpendicular_actions(action)[1]

    next_state = move(state, actual_action)
    reward = 10.0 if next_state == GOAL else -1.0
    done = (next_state == GOAL)

    return next_state, reward, done


def q_value(state, action, V):
    return sum(
        prob * (reward + GAMMA * V[next_state])
        for prob, next_state, reward in get_transitions(state, action)
    )


def greedy_action_from_V(state, V):
    q_vals = {a: q_value(state, a, V) for a in ACTIONS}
    return max(q_vals, key=q_vals.get)


def epsilon_greedy_action(state, Q, rng, epsilon=EPSILON):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)

    q_vals = [Q[(state, a)] for a in ACTIONS]
    max_q = max(q_vals)
    best_actions = [a for a in ACTIONS if Q[(state, a)] == max_q]
    return rng.choice(best_actions)


def print_value_table(V, title):
    print(title)
    for y in range(GRID_SIZE - 1, -1, -1):
        row = []
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row.append("#####".rjust(8))
            elif s == GOAL:
                row.append(" GOAL ".rjust(8))
            else:
                row.append(f"{V[s]:8.2f}")
        print(" ".join(row))
    print()


def print_policy_table(policy, title):
    print(title)
    for y in range(GRID_SIZE - 1, -1, -1):
        row = []
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row.append("■".center(5))
            elif s == GOAL:
                row.append("G".center(5))
            else:
                row.append(ARROWS[policy[s]].center(5))
        print(" ".join(row))
    print()


# ---------------- Part 2 Task 1 ----------------

def value_iteration():
    states = get_states()
    V = {s: 0.0 for s in states}

    iterations = 0
    while True:
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if s == GOAL:
                new_V[s] = 0.0
                continue

            new_V[s] = max(q_value(s, a, V) for a in ACTIONS)
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        iterations += 1

        if delta < THETA:
            break

    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = "G"
        else:
            policy[s] = greedy_action_from_V(s, V)

    return V, policy, iterations


def policy_iteration():
    states = get_states()
    policy = {s: ("G" if s == GOAL else "U") for s in states}
    V = {s: 0.0 for s in states}

    outer_iterations = 0

    while True:
        # policy evaluation
        while True:
            delta = 0.0
            for s in states:
                old_v = V[s]

                if s == GOAL:
                    V[s] = 0.0
                else:
                    a = policy[s]
                    V[s] = sum(
                        prob * (reward + GAMMA * V[next_state])
                        for prob, next_state, reward in get_transitions(s, a)
                    )

                delta = max(delta, abs(V[s] - old_v))

            if delta < THETA:
                break

        # policy improvement
        stable = True
        outer_iterations += 1

        for s in states:
            if s == GOAL:
                continue

            old_action = policy[s]
            best_action = greedy_action_from_V(s, V)
            policy[s] = best_action

            if best_action != old_action:
                stable = False

        if stable:
            break

    return V, policy, outer_iterations


def compare_policies(policy_a, policy_b):
    diffs = []
    for s in get_states():
        if s == GOAL:
            continue
        if policy_a[s] != policy_b[s]:
            diffs.append(s)
    return diffs


def run_part2_task1():
    V_vi, policy_vi, vi_iters = value_iteration()
    V_pi, policy_pi, pi_iters = policy_iteration()

    print("Part 2 - Task 1")
    print(f"Value Iteration converged in {vi_iters} iterations.")
    print(f"Policy Iteration converged in {pi_iters} improvement rounds.\n")

    print_value_table(V_vi, "Value Function from Value Iteration:")
    print_policy_table(policy_vi, "Policy from Value Iteration:")

    print_value_table(V_pi, "Value Function from Policy Iteration:")
    print_policy_table(policy_pi, "Policy from Policy Iteration:")

    diffs = compare_policies(policy_vi, policy_pi)
    if not diffs:
        print("Comparison: VI and PI produced the same policy.\n")
    else:
        print("Comparison: VI and PI differ at states:", diffs, "\n")

    return V_vi, policy_vi


# ---------------- Part 2 Task 2 ----------------

def monte_carlo_control(num_episodes=50000, seed=42):
    rng = random.Random(seed)
    states = get_states()

    Q = {(s, a): 0.0 for s in states for a in ACTIONS}
    returns_sum = {(s, a): 0.0 for s in states for a in ACTIONS}
    returns_count = {(s, a): 0 for s in states for a in ACTIONS}

    episode_lengths = []

    for _ in range(num_episodes):
        episode = []
        state = START
        done = False

        while not done:
            action = epsilon_greedy_action(state, Q, rng, EPSILON)
            next_state, reward, done = env_step(state, action, rng)
            episode.append((state, action, reward))
            state = next_state

        episode_lengths.append(len(episode))

        G_return = 0.0
        visited = set()

        for state, action, reward in reversed(episode):
            G_return = reward + GAMMA * G_return

            if (state, action) not in visited:
                visited.add((state, action))
                returns_sum[(state, action)] += G_return
                returns_count[(state, action)] += 1
                Q[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]

    V = {}
    policy = {}

    for s in states:
        if s == GOAL:
            V[s] = 0.0
            policy[s] = "G"
        else:
            best_action = max(ACTIONS, key=lambda a: Q[(s, a)])
            V[s] = Q[(s, best_action)]
            policy[s] = best_action

    return Q, V, policy, episode_lengths


def run_part2_task2(optimal_policy):
    _, V_mc, policy_mc, episode_lengths = monte_carlo_control(
        num_episodes=50000,
        seed=42
    )

    print("Part 2 - Task 2")
    print("Monte Carlo training episodes: 50000")
    print(f"Average episode length over last 1000 episodes: {sum(episode_lengths[-1000:]) / 1000:.2f}\n")

    print_value_table(V_mc, "Monte Carlo Learned State Values:")
    print_policy_table(policy_mc, "Monte Carlo Learned Policy:")

    diffs = compare_policies(policy_mc, optimal_policy)
    if not diffs:
        print("Comparison with Task 1 optimal policy: same policy.\n")
    else:
        print("Comparison with Task 1 optimal policy: different at states:")
        for s in diffs:
            print(f"{s}: MC={policy_mc[s]}, Optimal={optimal_policy[s]}")
        print()

    return V_mc, policy_mc


# ---------------- Part 2 Task 3 ----------------

def q_learning(num_episodes=50000, alpha=ALPHA, epsilon=EPSILON, seed=42, max_steps=200):
    rng = random.Random(seed)
    states = get_states()

    Q = {(s, a): 0.0 for s in states for a in ACTIONS}
    episode_lengths = []

    for _ in range(num_episodes):
        state = START
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = epsilon_greedy_action(state, Q, rng, epsilon)
            next_state, reward, done = env_step(state, action, rng)

            if done:
                target = reward
            else:
                target = reward + GAMMA * max(Q[(next_state, a)] for a in ACTIONS)

            Q[(state, action)] += alpha * (target - Q[(state, action)])

            state = next_state
            steps += 1

        episode_lengths.append(steps)

    V = {}
    policy = {}

    for s in states:
        if s == GOAL:
            V[s] = 0.0
            policy[s] = "G"
        else:
            best_action = max(ACTIONS, key=lambda a: Q[(s, a)])
            V[s] = Q[(s, best_action)]
            policy[s] = best_action

    return Q, V, policy, episode_lengths


def run_part2_task3(optimal_policy, mc_policy):
    _, V_q, policy_q, episode_lengths = q_learning(
        num_episodes=50000,
        alpha=0.1,
        epsilon=0.1,
        seed=42,
        max_steps=200
    )

    print("Part 2 - Task 3")
    print("Q-learning training episodes: 50000")
    print(f"Average episode length over last 1000 episodes: {sum(episode_lengths[-1000:]) / 1000:.2f}\n")

    print_value_table(V_q, "Q-learning Learned State Values:")
    print_policy_table(policy_q, "Q-learning Learned Policy:")

    diff_mc = compare_policies(policy_q, mc_policy)
    diff_opt = compare_policies(policy_q, optimal_policy)

    if not diff_mc:
        print("Comparison with Monte Carlo policy: same policy.")
    else:
        print("Comparison with Monte Carlo policy: different at states:")
        for s in diff_mc:
            print(f"{s}: Q-learning={policy_q[s]}, MC={mc_policy[s]}")

    print()

    if not diff_opt:
        print("Comparison with Task 1 optimal policy: same policy.\n")
    else:
        print("Comparison with Task 1 optimal policy: different at states:")
        for s in diff_opt:
            print(f"{s}: Q-learning={policy_q[s]}, Optimal={optimal_policy[s]}")
        print()

    return V_q, policy_q


def run_part2():
    _, optimal_policy = run_part2_task1()
    _, mc_policy = run_part2_task2(optimal_policy)
    run_part2_task3(optimal_policy, mc_policy)


# =========================================================
# Main
# =========================================================

def main():
    # ---------------- Part 1 ----------------
    G, Dist, Cost, Coord = load_graph()

    start = 1
    goal = 50
    budget = 287932

    path1, dist1 = dijkstra_distance(G, Dist, start, goal)
    energy1 = path_sum(path1, Cost) if path1 else float("inf")
    print_result("Task 1", path1, dist1, energy1)

    path2, dist2, energy2 = constrained_ucs(G, Dist, Cost, start, goal, budget)
    print_result("Task 2", path2, dist2, energy2)

    path3, dist3, energy3 = astar_with_energy_budget(G, Dist, Cost, Coord, start, goal, budget)
    print_result("Task 3", path3, dist3, energy3)

    print("=" * 60)

    # ---------------- Part 2 ----------------
    run_part2()


if __name__ == "__main__":
    main()