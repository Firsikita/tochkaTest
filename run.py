import heapq
import sys
from typing import Tuple, Dict, List

DOORS = [2, 4, 6, 8]
HALL_STOPS = [0, 1, 3, 5, 7, 9, 10]
COST = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
ROOM_IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
IDX_ROOM = {v: k for k, v in ROOM_IDX.items()}

State = Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]


def solve(lines: list[str]) -> int:
    start = parse(lines)

    if is_goal(start):
        return 0

    g_best: Dict[State, int] = {start: 0}
    pq = []
    counter = 0
    h0 = heuristic(start)
    heapq.heappush(pq, (h0, h0, counter, start))

    while pq:
        f, h_cur, _, s = heapq.heappop(pq)
        g = g_best.get(s, None)
        if g is None:
            continue

        if is_goal(s):
            return g

        for ns, move_cost in moves_from(s):
            ng = g + move_cost
            if ng < g_best.get(ns, float('inf')):
                g_best[ns] = ng
                h_ns = heuristic(ns)
                counter += 1
                heapq.heappush(pq, (ng + h_ns, h_ns, counter, ns))

    raise RuntimeError("Решение не найдено")


def parse(lines: list) -> State:
    room_rows = []
    for i, s in enumerate(lines):
        if i >= 2:
            cols = []
            for col in (3, 5, 7, 9):
                ch = s[col]
                cols.append(ch)

            if len(cols) == 4 and all(c in "ABCD" for c in cols):
                if any(c in "ABCD" for c in cols):
                    room_rows.append(cols)

    D = len(room_rows)
    if D not in (2, 4):
        raise ValueError("Неподдерживаемая глубина комнат")

    rooms = []
    for r in range(4):
        col = []
        for d in range(D):
            ch = room_rows[d][r]
            col.append(ch)
        rooms.append(tuple(col))

    hall = tuple(['.'] * 11)
    return (hall, tuple(rooms))


def is_goal(state: State) -> bool:
    hall, rooms = state
    if any(c != '.' for c in hall):
        return False

    deep = len(rooms[0])
    for r in range(4):
        want = IDX_ROOM[r]
        if any(rooms[r][d] != want for d in range(deep)):
            return False

    return True


def heuristic(state: State) -> int:
    hall, rooms = state
    coast = 0

    for i, chr in enumerate(hall):
        if chr == '.':
            continue
        target = DOORS[ROOM_IDX[chr]]
        coast += abs(i - target) * COST[chr]

    for row in range(4):
        col = rooms[row]
        deep = len(col)

        for d in range(deep):
            chr = col[d]
            if chr == '.':
                continue

            target_idx = ROOM_IDX[chr]
            coast += abs(DOORS[row] - DOORS[target_idx]) * COST[chr]

    return coast


def moves_from(state: State) -> List[Tuple[State, int]]:
    hall, rooms = state
    res = []
    res.extend(moves_from_hall_to_rooms(hall, rooms))
    res.extend(moves_from_rooms_to_hall(hall, rooms))
    return res


def moves_from_hall_to_rooms(hall: Tuple[str, ...], rooms: Tuple[Tuple[str, ...], ...]) -> List[Tuple[State, int]]:
    res = []
    for pos in range(11):
        chr = hall[pos]
        if chr == '.':
            continue

        r_idx = ROOM_IDX[chr]
        ok, depth = room_accepts(rooms, r_idx, chr)
        if not ok:
            continue

        door = DOORS[r_idx]
        if path_clear(hall, pos, door):
            steps = abs(pos - door) + (depth + 1)
            cost = steps * COST[chr]

            new_hall = list(hall)
            new_hall[pos] = '.'
            new_rooms = [list(col) for col in rooms]
            new_rooms[r_idx][depth] = chr
            res.append(((tuple(new_hall), tuple(tuple(c) for c in new_rooms)), cost))

    return res


def moves_from_rooms_to_hall(hall: Tuple[str, ...], rooms: Tuple[Tuple[str, ...], ...]) -> List[Tuple[State, int]]:
    res = []

    for r_idx in range(4):
        ok, d, chr = room_top_movable(rooms, r_idx)
        if not ok:
            continue

        door = DOORS[r_idx]
        for pos in range(door - 1, -1, -1):
            if hall[pos] != '.':
                break

            if pos in DOORS:
                continue

            steps = (d + 1) + abs(pos - door)
            cost = steps * COST[chr]

            new_hall = list(hall)
            new_hall[pos] = chr
            new_rooms = [list(col) for col in rooms]
            new_rooms[r_idx][d] = '.'
            res.append(((tuple(new_hall), tuple(tuple(c) for c in new_rooms)), cost))

        for pos in range(door + 1, 11):
            if hall[pos] != '.':
                break

            if pos in DOORS:
                continue

            steps = (d + 1) + abs(pos - door)
            cost = steps * COST[chr]

            new_hall = list(hall)
            new_hall[pos] = chr
            new_rooms = [list(col) for col in rooms]
            new_rooms[r_idx][d] = '.'
            res.append(((tuple(new_hall), tuple(tuple(c) for c in new_rooms)), cost))

    return res


def room_accepts(rooms: Tuple[Tuple[str, ...], ...], room_idx: int, chr: str) -> Tuple[bool, int]:
    col = rooms[room_idx]
    deep = len(col)
    if any(c not in ('.', chr) for c in col):
        return (False, -1)

    for d in range(deep - 1, -1, -1):
        if col[d] == '.':
            return (True, d)

    return (False, -1)


def path_clear(hall: Tuple[str, ...], i: int, j: int) -> bool:
    if i < j:
        rng = range(i + 1, j + 1)
    else:
        rng = range(j, i)
    return all(hall[k] == '.' for k in rng)


def room_top_movable(rooms: Tuple[Tuple[str, ...], ...], room_idx: int) -> Tuple[bool, int, str]:
    col = rooms[room_idx]
    deep = len(col)
    top = -1
    for d in range(deep):
        if col[d] != '.':
            top = d
            break

    if top == -1:
        return (False, -1, '')

    chr = col[top]
    target_room = ROOM_IDX[chr]
    below = col[top + 1:] if top + 1 < deep else ()
    if room_idx == target_room and all(c == chr for c in below):
        return (False, -1, '')

    return (True, top, chr)


def main():
    # Чтение входных данных
    lines = []
    for line in sys.stdin:
        lines.append(line.rstrip('\n'))

    result = solve(lines)
    print(result)


if __name__ == "__main__":
    main()
