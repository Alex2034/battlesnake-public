"""Stronger heuristic move-selection logic for Battlesnake.

Drop this file in the project as ``logic.py``.

The bot is intentionally dependency-free and deterministic.  It does not rely on
an embedded ML checkpoint; instead it scores every currently legal move using
survival-first heuristics:

1. avoid walls, bodies, lethal hazards, and losing head-to-head cells;
2. prefer moves with large reachable space and multiple exits;
3. keep a path to our tail when possible, so we do not trap ourselves;
4. use Voronoi-style territory against enemy snakes;
5. go for food when hungry, but avoid cramped/contested food traps.

Board coordinates: (0, 0) is bottom-left.
"""

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

NEIGHBORS: Tuple[Point, ...] = tuple(DIRECTIONS.values())
BIG = 10_000

# Tuning constants.  They are deliberately simple so you can adjust them after
# watching a few replays.
HUNGRY_HEALTH = 45
VERY_HUNGRY_HEALTH = 25
LOSING_HEAD_TO_HEAD_PENALTY = 8_000
HAZARD_PENALTY = 450
ONE_EXIT_PENALTY = 220
NO_EXIT_PENALTY = 2_500
FOOD_TRAP_PENALTY = 900
TAIL_REACH_REWARD = 140


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from GET /."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#34c759",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.2.0-heuristic",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next Battlesnake move: up, down, left, or right."""
    try:
        scored = _score_moves(game_state)
        if scored:
            attack_move = _find_safe_forced_kill_move(game_state, scored, horizon=3)
            if attack_move is not None:
                return attack_move

            scored.sort(key=lambda row: row["score"], reverse=True)
            return str(scored[0]["move"])
    except Exception as exc:  # noqa: BLE001 - gameplay must never 500 on /move
        print(f"choose_move error: {exc}")

    return _last_chance_move(game_state)


def _score_moves(state: Dict) -> List[Dict[str, float]]:
    board = state["board"]
    you = state["you"]
    width, height = int(board["width"]), int(board["height"])

    my_id = you["id"]
    my_body = _body_points(you)
    head = my_body[0]
    my_tail = my_body[-1]
    my_length = int(you["length"])
    health = int(you["health"])

    snakes = board.get("snakes", [])
    foods = _points(board.get("food", []))
    food_set = set(foods)
    hazards = set(_points(board.get("hazards", [])))
    hazard_damage = _hazard_damage(state)

    # For immediate legality, allow stepping onto tails that are likely to move.
    # This is an important improvement over treating every tail as permanently
    # blocked.  If a snake can plausibly eat next turn, keep its tail blocked.
    blocked_now = _blocked_for_current_turn(snakes, food_set)

    enemy_snakes = [s for s in snakes if s.get("id") != my_id]
    enemy_heads = [_head_point(s) for s in enemy_snakes]
    losing_h2h = _head_to_head_cells(enemy_snakes, my_length, include_equal=True)
    winning_h2h = _head_to_head_cells(
        [s for s in enemy_snakes if int(s.get("length", 0)) < my_length],
        my_length,
        include_equal=False,
    )

    scored: List[Dict[str, float]] = []
    for move, delta in DIRECTIONS.items():
        nxt = _add(head, delta)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in blocked_now:
            continue
        if _lethal_hazard(nxt, food_set, hazards, health, hazard_damage):
            continue

        eating = nxt in food_set
        blocked_after = _blocked_after_our_move(snakes, my_id, nxt, eating, food_set)
        blocked_after.discard(nxt)  # BFS starts from our new head cell.

        safe_blocked = set(blocked_after)
        if health <= HUNGRY_HEALTH:
            # When low on health, hazard cells are not really usable space.
            safe_blocked |= hazards

        open_space = _flood_fill(nxt, blocked_after, width, height, limit=width * height)
        safe_space = _flood_fill(nxt, safe_blocked, width, height, limit=width * height)
        space_capped = min(safe_space, my_length * 3)
        exits = _exit_count(nxt, blocked_after, width, height, hazards, health, hazard_damage)
        wall_dist = _wall_distance(nxt, width, height)
        center_dist = _center_distance(nxt, width, height)

        # Can we still reach our tail after moving?  This prevents many common
        # self-traps in corridors and loops.
        projected_tail = _projected_own_tail(my_body, nxt, eating)
        tail_reachable = _reachable(nxt, projected_tail, blocked_after - {projected_tail}, width, height)

        territory = _voronoi_score(nxt, enemy_heads, blocked_after, width, height)
        nearest_food_next = min((_manhattan(nxt, f) for f in foods), default=BIG)
        nearest_food_now = min((_manhattan(head, f) for f in foods), default=BIG)
        food_delta = nearest_food_now - nearest_food_next if foods else 0
        contested_food = _is_contested_food(nxt, enemy_snakes, my_length)

        score = 0.0

        # Survival / space dominates everything else.
        score += 5.0 * space_capped
        score += 1.2 * safe_space
        score += 0.25 * open_space
        score += 2.2 * territory
        score += 32.0 * exits
        score += 9.0 * wall_dist
        score -= 0.8 * center_dist

        if tail_reachable:
            score += TAIL_REACH_REWARD
        elif safe_space < my_length * 2:
            score -= 70.0 * (my_length * 2 - safe_space)

        if safe_space < my_length:
            score -= 130.0 * (my_length - safe_space)

        if exits == 0:
            score -= NO_EXIT_PENALTY
        elif exits == 1:
            score -= ONE_EXIT_PENALTY

        # Head-to-head handling.  Equal length is dangerous: both die, so avoid.
        if nxt in losing_h2h:
            score -= LOSING_HEAD_TO_HEAD_PENALTY
        elif nxt in winning_h2h and safe_space >= my_length:
            score += 120.0

        # Hazards are survivable but expensive.  The lower health is, the worse.
        if nxt in hazards:
            score -= HAZARD_PENALTY + max(0, HUNGRY_HEALTH - health) * 12

        # Food: go hard when hungry, otherwise do not let food lure us into traps.
        if foods:
            if health <= VERY_HUNGRY_HEALTH:
                score += 38.0 * food_delta + max(0, width + height - nearest_food_next) * 8.0
            elif health <= HUNGRY_HEALTH:
                score += 24.0 * food_delta + max(0, width + height - nearest_food_next) * 4.0
            else:
                score += 5.0 * food_delta

        if eating:
            score += 60.0 if health <= HUNGRY_HEALTH else 10.0
            if contested_food:
                score -= 220.0
            if safe_space < my_length + 3:
                score -= FOOD_TRAP_PENALTY

        # Prefer not to move next to a larger/equal enemy head even if the exact
        # h2h cell calculation missed something because of obstacles.
        bigger_or_equal = [s for s in enemy_snakes if int(s.get("length", 0)) >= my_length]
        if bigger_or_equal:
            nearest_big = min((_manhattan(nxt, _head_point(s)) for s in bigger_or_equal), default=BIG)
            score += min(nearest_big, 5) * 18.0

        scored.append(
            {
                "move": move,
                "score": score,
                "safe_space": float(safe_space),
                "open_space": float(open_space),
                "exits": float(exits),
                "tail_reachable": 1.0 if tail_reachable else 0.0,
            }
        )

    return scored


def _last_chance_move(state: Dict) -> str:
    """Return something reasonable even when every move looks bad."""
    board = state.get("board", {})
    you = state.get("you", {})
    width, height = int(board.get("width", 11)), int(board.get("height", 11))
    body = _body_points(you) if you.get("body") else [(0, 0)]
    head = body[0]
    occupied = _all_body_cells(board.get("snakes", []))

    # First: any in-bounds move not immediately into a body.
    for move, delta in DIRECTIONS.items():
        nxt = _add(head, delta)
        if _in_bounds(nxt, width, height) and nxt not in occupied:
            return move

    # Second: any in-bounds move.  We may be dead anyway, but avoid walls.
    for move, delta in DIRECTIONS.items():
        if _in_bounds(_add(head, delta), width, height):
            return move

    return "up"



# ---------------------------------------------------------------------------
# Short-horizon attack search


def _find_safe_forced_kill_move(state: Dict, scored: Sequence[Dict[str, float]], horizon: int = 3) -> Optional[str]:
    """Return an aggressive move only when it safely forces a shorter snake to die.

    This is intentionally conservative.  The search only attacks snakes that are
    strictly shorter than us, because those are the only snakes we can safely
    beat in a head-to-head collision.  It then checks a small minimax tree:

    * we choose a move;
    * the target may choose any currently survivable reply;
    * after observing that reply on the next turn, we may choose again;
    * if every target line dies within ``horizon`` plies while we stay alive,
      the first move is considered a forced kill.

    Other enemy snakes are treated as static blockers.  That keeps the search
    fast and avoids reckless attacks in crowded boards.
    """
    board = state.get("board", {})
    you = state.get("you", {})
    width, height = int(board.get("width", 11)), int(board.get("height", 11))
    foods = set(_points(board.get("food", [])))
    hazards = set(_points(board.get("hazards", [])))
    hazard_damage = _hazard_damage(state)

    my_id = you.get("id")
    my_body = _body_points(you)
    my_length = int(you.get("length", len(my_body)))
    my_health = int(you.get("health", 100))
    snakes = board.get("snakes", [])

    if not my_body or my_length < 3:
        return None

    scored_by_move = {str(row["move"]): row for row in scored}
    candidate_moves = sorted(scored_by_move, key=lambda m: scored_by_move[m]["score"], reverse=True)

    for move in candidate_moves:
        row = scored_by_move[move]
        # Do not sacrifice the survival policy for a speculative tactic.
        if row.get("safe_space", 0.0) < max(float(my_length + 2), float(my_length) * 1.4):
            continue
        if row.get("exits", 0.0) <= 0:
            continue

        for target in snakes:
            if target.get("id") == my_id:
                continue
            target_body = _body_points(target)
            target_length = int(target.get("length", len(target_body)))
            if not target_body or target_length >= my_length:
                continue

            other_blocked = _static_other_snakes(snakes, {my_id, target.get("id")})
            if _move_forces_target_death(
                my_body=my_body,
                my_health=my_health,
                target_body=target_body,
                target_health=int(target.get("health", 100)),
                first_move=move,
                depth=horizon,
                width=width,
                height=height,
                foods=foods,
                hazards=hazards,
                hazard_damage=hazard_damage,
                other_blocked=other_blocked,
            ):
                print(f"ATTACK: forced kill on {target.get('name') or target.get('id')} via {move}")
                return move

    return None


def _move_forces_target_death(
    my_body: List[Point],
    my_health: int,
    target_body: List[Point],
    target_health: int,
    first_move: str,
    depth: int,
    width: int,
    height: int,
    foods: Set[Point],
    hazards: Set[Point],
    hazard_damage: int,
    other_blocked: Set[Point],
) -> bool:
    """Check whether one chosen move wins against every target reply."""
    my_moves = [first_move]
    return _exists_killing_strategy(
        my_body=my_body,
        my_health=my_health,
        target_body=target_body,
        target_health=target_health,
        depth=depth,
        width=width,
        height=height,
        foods=foods,
        hazards=hazards,
        hazard_damage=hazard_damage,
        other_blocked=other_blocked,
        forced_my_moves=my_moves,
    )


def _exists_killing_strategy(
    my_body: List[Point],
    my_health: int,
    target_body: List[Point],
    target_health: int,
    depth: int,
    width: int,
    height: int,
    foods: Set[Point],
    hazards: Set[Point],
    hazard_damage: int,
    other_blocked: Set[Point],
    forced_my_moves: Optional[List[str]] = None,
) -> bool:
    """Minimax search: exists our move such that all target replies lose."""
    if depth <= 0:
        return False

    if forced_my_moves:
        my_moves = forced_my_moves
    else:
        my_moves = _tactical_safe_moves(my_body, target_body, my_health, width, height, foods, hazards, hazard_damage, other_blocked)

    if not my_moves:
        return False

    for my_move in my_moves:
        target_replies = _tactical_safe_moves(target_body, my_body, target_health, width, height, foods, hazards, hazard_damage, other_blocked)

        # If the target has no legal reply before this simultaneous turn, the
        # trap is already complete.
        if not target_replies:
            return True

        all_replies_die = True
        for target_move in target_replies:
            result = _resolve_two_snake_turn(
                my_body=my_body,
                my_health=my_health,
                my_move=my_move,
                target_body=target_body,
                target_health=target_health,
                target_move=target_move,
                width=width,
                height=height,
                foods=foods,
                hazards=hazards,
                hazard_damage=hazard_damage,
                other_blocked=other_blocked,
            )

            if result["my_dead"]:
                all_replies_die = False
                break
            if result["target_dead"]:
                continue

            next_foods = set(foods)
            if result["my_ate"]:
                next_foods.discard(result["my_head"])
            if result["target_ate"]:
                next_foods.discard(result["target_head"])

            if not _exists_killing_strategy(
                my_body=result["my_body"],
                my_health=result["my_health"],
                target_body=result["target_body"],
                target_health=result["target_health"],
                depth=depth - 1,
                width=width,
                height=height,
                foods=next_foods,
                hazards=hazards,
                hazard_damage=hazard_damage,
                other_blocked=other_blocked,
                forced_my_moves=None,
            ):
                all_replies_die = False
                break

        if all_replies_die:
            return True

    return False


def _tactical_safe_moves(
    body: List[Point],
    opponent_body: List[Point],
    health: int,
    width: int,
    height: int,
    foods: Set[Point],
    hazards: Set[Point],
    hazard_damage: int,
    other_blocked: Set[Point],
) -> List[str]:
    """Legal moves for a simplified two-snake tactical search."""
    if not body:
        return []
    head = body[0]
    own_tail = body[-1]
    opp_tail = opponent_body[-1] if opponent_body else None

    blocked = set(body) | set(opponent_body) | set(other_blocked)
    blocked.discard(own_tail)
    if opp_tail is not None:
        blocked.discard(opp_tail)

    moves: List[str] = []
    for move, delta in DIRECTIONS.items():
        nxt = _add(head, delta)
        if not _in_bounds(nxt, width, height):
            continue
        if nxt in blocked:
            continue
        if _lethal_hazard(nxt, foods, hazards, health, hazard_damage):
            continue
        moves.append(move)
    return moves


def _resolve_two_snake_turn(
    my_body: List[Point],
    my_health: int,
    my_move: str,
    target_body: List[Point],
    target_health: int,
    target_move: str,
    width: int,
    height: int,
    foods: Set[Point],
    hazards: Set[Point],
    hazard_damage: int,
    other_blocked: Set[Point],
) -> Dict:
    """Project one simultaneous turn for our snake and one target snake."""
    my_head = _add(my_body[0], DIRECTIONS[my_move])
    target_head = _add(target_body[0], DIRECTIONS[target_move])

    my_ate = my_head in foods
    target_ate = target_head in foods

    new_my_body = _advance_body(my_body, my_head, my_ate)
    new_target_body = _advance_body(target_body, target_head, target_ate)

    my_next_health = _next_health(my_health, my_head, my_ate, hazards, hazard_damage)
    target_next_health = _next_health(target_health, target_head, target_ate, hazards, hazard_damage)

    my_dead = not _in_bounds(my_head, width, height) or my_next_health <= 0
    target_dead = not _in_bounds(target_head, width, height) or target_next_health <= 0

    if my_head in other_blocked:
        my_dead = True
    if target_head in other_blocked:
        target_dead = True

    if my_head == target_head:
        if len(new_my_body) > len(new_target_body):
            target_dead = True
        elif len(new_my_body) < len(new_target_body):
            my_dead = True
        else:
            my_dead = True
            target_dead = True
    else:
        # Body collisions after tails have moved.
        if my_head in set(new_my_body[1:]) or my_head in set(new_target_body[1:]):
            my_dead = True
        if target_head in set(new_target_body[1:]) or target_head in set(new_my_body[1:]):
            target_dead = True

    return {
        "my_dead": my_dead,
        "target_dead": target_dead,
        "my_body": new_my_body,
        "target_body": new_target_body,
        "my_health": my_next_health,
        "target_health": target_next_health,
        "my_ate": my_ate,
        "target_ate": target_ate,
        "my_head": my_head,
        "target_head": target_head,
    }


def _advance_body(body: List[Point], new_head: Point, eating: bool) -> List[Point]:
    new_body = [new_head] + list(body)
    if not eating:
        new_body = new_body[:-1]
    return new_body


def _next_health(health: int, head: Point, eating: bool, hazards: Set[Point], hazard_damage: int) -> int:
    if eating:
        return 100
    next_health = health - 1
    if head in hazards:
        next_health -= hazard_damage
    return next_health


def _static_other_snakes(snakes: Sequence[Dict], excluded_ids: Set[str]) -> Set[Point]:
    blocked: Set[Point] = set()
    for snake in snakes:
        if snake.get("id") in excluded_ids:
            continue
        blocked.update(_body_points(snake))
    return blocked

# ---------------------------------------------------------------------------
# Geometry and board helpers


def _points(items: Iterable[Dict]) -> List[Point]:
    return [(int(p["x"]), int(p["y"])) for p in items]


def _body_points(snake: Dict) -> List[Point]:
    return _points(snake.get("body", []))


def _head_point(snake: Dict) -> Point:
    head = snake.get("head") or snake.get("body", [{}])[0]
    return int(head["x"]), int(head["y"])


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _wall_distance(p: Point, width: int, height: int) -> int:
    return min(p[0], width - 1 - p[0], p[1], height - 1 - p[1])


def _center_distance(p: Point, width: int, height: int) -> float:
    return abs(p[0] - (width - 1) / 2) + abs(p[1] - (height - 1) / 2)


def _hazard_damage(state: Dict) -> int:
    settings = (
        state.get("game", {})
        .get("ruleset", {})
        .get("settings", {})
    )
    return int(settings.get("hazardDamagePerTurn", 15))


def _lethal_hazard(
    p: Point,
    foods: Set[Point],
    hazards: Set[Point],
    health: int,
    hazard_damage: int,
) -> bool:
    if p not in hazards:
        return False
    if p in foods:
        # Food usually saves us; still gets penalized later.
        return False
    return health <= hazard_damage + 1


# ---------------------------------------------------------------------------
# Collision and projection helpers


def _all_body_cells(snakes: Sequence[Dict]) -> Set[Point]:
    cells: Set[Point] = set()
    for snake in snakes:
        cells.update(_body_points(snake))
    return cells


def _could_eat_next_turn(snake: Dict, foods: Set[Point]) -> bool:
    """Conservative guess: if food is adjacent, the snake may keep its tail."""
    if not foods:
        return False
    head = _head_point(snake)
    return any(_add(head, d) in foods for d in NEIGHBORS)


def _blocked_for_current_turn(snakes: Sequence[Dict], foods: Set[Point]) -> Set[Point]:
    blocked = _all_body_cells(snakes)
    for snake in snakes:
        body = _body_points(snake)
        if body and not _could_eat_next_turn(snake, foods):
            blocked.discard(body[-1])
    return blocked


def _blocked_after_our_move(
    snakes: Sequence[Dict],
    my_id: str,
    new_head: Point,
    eating: bool,
    foods: Set[Point],
) -> Set[Point]:
    """Approximate board occupancy immediately after our chosen move."""
    blocked: Set[Point] = set()

    for snake in snakes:
        body = _body_points(snake)
        if not body:
            continue

        if snake.get("id") == my_id:
            projected = [new_head] + body
            if not eating:
                projected = projected[:-1]
            blocked.update(projected)
            continue

        # We do not know enemy moves.  Keep their bodies, but allow tails to
        # vacate unless that enemy can plausibly eat.
        enemy_body = list(body)
        if not _could_eat_next_turn(snake, foods):
            enemy_body = enemy_body[:-1]
        blocked.update(enemy_body)

    return blocked


def _projected_own_tail(my_body: List[Point], new_head: Point, eating: bool) -> Point:
    projected = [new_head] + list(my_body)
    if not eating:
        projected = projected[:-1]
    return projected[-1]


def _head_to_head_cells(snakes: Sequence[Dict], my_length: int, include_equal: bool) -> Set[Point]:
    cells: Set[Point] = set()
    for snake in snakes:
        length = int(snake.get("length", len(snake.get("body", []))))
        dangerous = length >= my_length if include_equal else length < my_length
        if not dangerous:
            continue
        head = _head_point(snake)
        for delta in NEIGHBORS:
            cells.add(_add(head, delta))
    return cells


def _is_contested_food(p: Point, enemies: Sequence[Dict], my_length: int) -> bool:
    """Food is contested if an equal/bigger enemy head can also reach it now."""
    for enemy in enemies:
        if int(enemy.get("length", 0)) < my_length:
            continue
        if _manhattan(p, _head_point(enemy)) <= 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Search / scoring helpers


def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: int) -> int:
    if not _in_bounds(start, width, height):
        return 0
    if start in blocked:
        return 0

    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0

    while stack and count < limit:
        cell = stack.pop()
        count += 1
        for delta in NEIGHBORS:
            nxt = _add(cell, delta)
            if nxt in seen or nxt in blocked or not _in_bounds(nxt, width, height):
                continue
            seen.add(nxt)
            stack.append(nxt)

    return count


def _reachable(start: Point, target: Point, blocked: Set[Point], width: int, height: int) -> bool:
    if start == target:
        return True
    return target in _bfs_dist([start], blocked, width, height)


def _bfs_dist(sources: Iterable[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    dist: Dict[Point, int] = {}
    queue: deque = deque()

    for source in sources:
        if not _in_bounds(source, width, height):
            continue
        if source in dist:
            continue
        dist[source] = 0
        queue.append(source)

    while queue:
        cell = queue.popleft()
        for delta in NEIGHBORS:
            nxt = _add(cell, delta)
            if nxt in dist or nxt in blocked or not _in_bounds(nxt, width, height):
                continue
            dist[nxt] = dist[cell] + 1
            queue.append(nxt)

    return dist


def _exit_count(
    p: Point,
    blocked: Set[Point],
    width: int,
    height: int,
    hazards: Set[Point],
    health: int,
    hazard_damage: int,
) -> int:
    count = 0
    for delta in NEIGHBORS:
        nxt = _add(p, delta)
        if not _in_bounds(nxt, width, height):
            continue
        if nxt in blocked:
            continue
        if _lethal_hazard(nxt, set(), hazards, health, hazard_damage):
            continue
        count += 1
    return count


def _voronoi_score(
    my_start: Point,
    enemy_heads: Sequence[Point],
    blocked: Set[Point],
    width: int,
    height: int,
) -> int:
    if not enemy_heads:
        return _flood_fill(my_start, blocked, width, height, width * height)

    bfs_blocked = set(blocked)
    bfs_blocked.discard(my_start)
    for head in enemy_heads:
        bfs_blocked.discard(head)

    my_dist = _bfs_dist([my_start], bfs_blocked, width, height)
    enemy_dist = _bfs_dist(enemy_heads, bfs_blocked, width, height)

    score = 0
    for cell, my_d in my_dist.items():
        enemy_d = enemy_dist.get(cell, BIG)
        if my_d < enemy_d:
            score += 1
        elif my_d == enemy_d:
            # Tied cells are less valuable than cells we clearly control.
            score += 0
    return score
