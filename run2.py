import sys
from collections import deque, defaultdict


def solve(edges: list[tuple[str, str]]) -> list[str]:
    graph, gateways = build_graph_and_gateways(edges)
    virus_node = "a"
    blocked_edges = []

    while get_reachable_gateways(graph, gateways, virus_node):
        gateway_dists = compute_gateway_distances(graph, gateways)
        target_dist, target_gateway = choose_closest_gateway(gateway_dists, virus_node)

        if target_gateway is None:
            break

        preportal_nodes = find_preportal_nodes(graph, gateway_dists[target_gateway], target_gateway, virus_node)
        blocked_edge = cut_connection(graph, target_gateway, preportal_nodes)
        blocked_edges.append(blocked_edge)

        virus_node, moved = move_virus(graph, gateways, virus_node)

        if not moved:
            break

    return blocked_edges

def build_graph_and_gateways(edges: list[tuple[str, str]]):
    graph = defaultdict(set)
    nodes = set()

    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        nodes.add(u); nodes.add(v)

    gateways = sorted([n for n in nodes if is_gateway(n)])
    return graph, gateways


def get_reachable_gateways(graph, gateways, virus_node):
    distances = bfs(graph, virus_node)
    return [g for g in gateways if g in distances]


def compute_gateway_distances(graph, gateways):
    return {g: bfs(graph, g) for g in gateways if g in graph}


def bfs(graph: dict[str, set[str]], src: str) -> dict[str, int]:
    dist = {src: 0}
    q = deque([src])

    while q:
        v = q.popleft()
        for w in graph[v]:
            if w not in dist:
                dist[w] = dist[v] + 1
                q.append(w)

    return dist


def is_gateway(node: str) -> bool:
    return node.isupper()


def lex_min_edge(edges_iter):
    return min(edges_iter, key=lambda x: (x[0], x[1]))


def choose_closest_gateway(gateway_distances, virus_node):
    candidates = [
        (distances[virus_node], gateway)
        for gateway, distances in gateway_distances.items()
        if virus_node in distances
    ]
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0]


def find_preportal_nodes(graph, dist_to_gateway, gateway, virus_node):
    distance_to_gateway = dist_to_gateway[virus_node]
    frontier = {virus_node}
    for d in range(distance_to_gateway, 1, -1):
        next_frontier = set()
        for node in frontier:
            for neighbor in graph[node]:
                if dist_to_gateway.get(neighbor) == d - 1:
                    next_frontier.add(neighbor)
        frontier = next_frontier

    preportal_nodes = {
        node for node in frontier
        if dist_to_gateway.get(node) == 1 and gateway in graph and node in graph[gateway]
    }
    return preportal_nodes


def cut_connection(graph, gateway, preportal_nodes):
    if preportal_nodes:
        cut_gateway, cut_node = gateway, min(preportal_nodes)
    else:
        cut_gateway, cut_node = lex_min_edge((gateway, n) for n in graph[gateway])

    graph[cut_gateway].discard(cut_node)
    graph[cut_node].discard(cut_gateway)
    return f"{cut_gateway}-{cut_node}"


def move_virus(graph, gateways, virus_node):
    new_gateway_dists = compute_gateway_distances(graph, gateways)
    distance, nearest_gateway = choose_closest_gateway(new_gateway_dists, virus_node)

    if nearest_gateway is None or distance == 0:
        return virus_node, False

    path_options = [
        neighbor for neighbor in graph[virus_node]
        if new_gateway_dists[nearest_gateway].get(neighbor) == distance - 1
    ]

    if path_options:
        return min(path_options), True
    return virus_node, False


def main():
    edges = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            node1, sep, node2 = line.partition('-')
            if sep:
                edges.append((node1, node2))

    result = solve(edges)
    for edge in result:
        print(edge)


if __name__ == "__main__":
    main()