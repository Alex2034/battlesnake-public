"""Hand-tuned board evaluation for the algorithmic Battlesnake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set

from .danger import build_threat_map
from .pathfinding import bfs_distances, count_exits, flood_fill, shortest_distance
from .types import BoardState, Point, Snake

DEATH_SCORE = -1_000_000.0


@dataclass(frozen=True)
class Evaluation:
    score: float
    features: Dict[str, float]


def evaluate_state(state: BoardState, previous_head: Optional[Point] = None) -> Evaluation:
    you = state.you
    if you is None or not you.alive:
        return Evaluation(DEATH_SCORE, {"dead": 1.0})

    blocked = set(state.occupied(exclude={you.head}))
    space = flood_fill(you.head, blocked, state.width, state.height)
    exits = count_exits(you.head, blocked, state.width, state.height)
    tail_distance = _tail_distance(state, you, blocked)
    reaches_tail = 1.0 if tail_distance is not None else 0.0
    territory = _voronoi_territory(state, you)
    food_score = _food_score(state, you, blocked)
    wall_pressure = _wall_pressure(state, you.head)
    threats = build_threat_map(state)
    head_danger = 1.0 if you.head in threats.lethal_head_to_head else 0.0
    kill_pressure = _kill_pressure(state, you)
    low_health = max(0.0, 35.0 - you.health)

    features = {
        "space": float(space),
        "exits": float(exits),
        "reaches_tail": reaches_tail,
        "tail_distance": float(tail_distance if tail_distance is not None else state.width * state.height),
        "territory": float(territory),
        "food": food_score,
        "health": float(you.health),
        "wall_pressure": float(wall_pressure),
        "head_danger": head_danger,
        "kill_pressure": kill_pressure,
        "low_health": low_health,
    }

    score = 0.0
    score += 14.0 * features["territory"]
    score += 6.0 * features["space"]
    score += 35.0 * features["exits"]
    score += 120.0 * features["reaches_tail"]
    score -= 1.5 * features["tail_distance"] if reaches_tail else 180.0
    score += features["food"]
    score += min(you.health, 80) * 1.5
    score -= 14.0 * features["low_health"]
    score -= 45.0 * features["wall_pressure"]
    score -= 900.0 * features["head_danger"]
    score += 80.0 * features["kill_pressure"]

    if space < you.length:
        score -= (you.length - space + 1) * 160.0
    if exits == 0:
        score -= 700.0
    elif exits == 1:
        score -= 140.0

    if previous_head is not None and you.head == previous_head:
        score -= 10.0

    return Evaluation(score, features)


def _tail_distance(state: BoardState, you: Snake, blocked: Set[Point]) -> Optional[int]:
    relaxed = set(blocked)
    relaxed.discard(you.tail)
    return shortest_distance(you.head, you.tail, relaxed, state.width, state.height)


def _voronoi_territory(state: BoardState, you: Snake) -> int:
    blocked = set(state.occupied())
    my_blocked = blocked - {you.head}
    my_dist = bfs_distances([you.head], my_blocked, state.width, state.height)

    enemy_heads = [enemy.head for enemy in state.enemies]
    if not enemy_heads:
        return len(my_dist)

    enemy_blocked = blocked - set(enemy_heads)
    enemy_dist = bfs_distances(enemy_heads, enemy_blocked, state.width, state.height)
    territory = 0
    for point, dist in my_dist.items():
        enemy = enemy_dist.get(point)
        if enemy is None or dist < enemy:
            territory += 1
    return territory


def _food_score(state: BoardState, you: Snake, blocked: Set[Point]) -> float:
    if not state.food:
        return 0.0

    distances = bfs_distances([you.head], blocked, state.width, state.height)
    reachable = [distances[food] for food in state.food if food in distances]
    if not reachable:
        return 0.0

    nearest = min(reachable)
    urgency = max(0.0, (80.0 - you.health) / 80.0)
    base = 260.0 * urgency
    opportunistic = 45.0 if you.health < 95 else 0.0
    return max(0.0, base + opportunistic - nearest * (8.0 + urgency * 8.0))


def _wall_pressure(state: BoardState, point: Point) -> float:
    x, y = point
    nearest_wall = min(x, state.width - 1 - x, y, state.height - 1 - y)
    center_distance = min(state.width - 1, state.height - 1) / 2.0
    return max(0.0, center_distance - nearest_wall)


def _kill_pressure(state: BoardState, you: Snake) -> float:
    pressure = 0.0
    for enemy in state.enemies:
        if enemy.length >= you.length:
            continue
        distance = abs(you.head[0] - enemy.head[0]) + abs(you.head[1] - enemy.head[1])
        if distance <= 2:
            pressure += 3 - distance
    return pressure
