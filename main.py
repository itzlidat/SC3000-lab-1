import json
import heapq
import math


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

        if old_dist <= new_dist and old_energy <= new_energy:
            return True, []

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


def main():
    G, Dist, Cost, Coord = load_graph()

    start = 1
    goal = 50
    budget = 287932

    # Task 1
    path1, dist1 = dijkstra_distance(G, Dist, start, goal)
    energy1 = path_sum(path1, Cost) if path1 else float("inf")
    print_result("Task 1", path1, dist1, energy1)

    # Task 2
    path2, dist2, energy2 = constrained_ucs(G, Dist, Cost, start, goal, budget)
    print_result("Task 2", path2, dist2, energy2)

    # Task 3
    path3, dist3, energy3 = astar_with_energy_budget(G, Dist, Cost, Coord, start, goal, budget)
    print_result("Task 3", path3, dist3, energy3)


if __name__ == "__main__":
    main()