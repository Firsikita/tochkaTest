import sys
from collections import deque, defaultdict


def solve(edges: list[tuple[str, str]]) -> list[str]:
    graph, gateways = build_graph_and_gateways(edges)
    virus = "a"
    answer = []

    while True:
        reachable = bfs(graph, virus)
        if not any(g in reachable for g in gateways):
            break

        g, x = choose_edge_to_cut(graph, gateways, virus)
        graph[g].discard(x)
        graph[x].discard(g)
        answer.append(f"{g}-{x}")

        new_gateway_dists = {g: bfs(graph, g) for g in gateways if g in graph}
        choice = [
            (dists[virus], g) for g, dists in new_gateway_dists.items() if virus in dists
        ]

        if not choice:
            break

        choice.sort(key=lambda t: (t[0], t[1]))
        dist_to_target, target_g = choice[0]

        if dist_to_target == 0:
            raise RuntimeError()

        next_options = [
            nei for nei in graph[virus]
            if new_gateway_dists[target_g].get(nei) == dist_to_target - 1
        ]
        virus = min(next_options)
    return answer

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


def choose_edge_to_cut(graph, gateways, virus_node) -> tuple[str, str]:
    dist_from_virus = bfs(graph, virus_node)

    candidates = []
    for g in gateways:
        for x in graph[g]:
            if x in dist_from_virus:
                candidates.append((dist_from_virus[x], g, x))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    _, g, x = candidates[0]
    return g, x


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