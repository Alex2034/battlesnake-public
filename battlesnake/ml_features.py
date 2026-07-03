"""Feature extraction for the embedded ML move-ranking model."""

from __future__ import annotations

from typing import Dict

from .danger import build_threat_map
from .pathfinding import bfs_distances, count_exits, flood_fill
from .rules import next_head
from .types import BoardState, DIRECTIONS, Point, Snake

BIG_DISTANCE = 10_000
HUNGRY_THRESHOLD = 50


def candidate_features(state: BoardState, snake: Snake, move: str) -> Dict[str, float]:
    """Return ML-branch-compatible features for a candidate move."""
    head = snake.head
    nxt = next_head(snake, move)
    occupied = set(state.occupied())
    foods = state.food
    enemies = state.enemies if snake.id == state.you_id else tuple(
        other for other in state.alive_snakes if other.id != snake.id
    )
    enemy_heads = [enemy.head for enemy in enemies]
    bigger_heads = [enemy.head for enemy in enemies if enemy.length >= snake.length]
    threats = build_threat_map(state)

    my_dist = bfs_distances([nxt], occupied, state.width, state.height)
    enemy_dist = bfs_distances(enemy_heads, occupied, state.width, state.height) if enemy_heads else {}
    voronoi = sum(1 for cell, dist in my_dist.items() if dist < enemy_dist.get(cell, BIG_DISTANCE))

    tail_reach = bfs_distances([nxt], occupied - {snake.tail}, state.width, state.height)
    nearest_now = _nearest_distance(head, foods)
    nearest_next = _nearest_distance(nxt, foods)
    hungry = snake.health < HUNGRY_THRESHOLD

    return {
        "space_capped": float(flood_fill(nxt, occupied, state.width, state.height, limit=snake.length + 1)),
        "open_space": float(flood_fill(nxt, occupied, state.width, state.height, limit=state.width * state.height)),
        "voronoi": float(voronoi),
        "reaches_tail": 1.0 if snake.tail in tail_reach else 0.0,
        "escape": float(count_exits(nxt, occupied, state.width, state.height)),
        "h2h_danger": 1.0 if nxt in threats.lethal_head_to_head else 0.0,
        "near_bigger_head": float(_nearest_distance(nxt, bigger_heads, default=state.width + state.height)),
        "near_enemy_head": float(_nearest_distance(nxt, enemy_heads, default=state.width + state.height)),
        "wall_dist": float(min(nxt[0], state.width - 1 - nxt[0], nxt[1], state.height - 1 - nxt[1])),
        "food_score": float((state.width + state.height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (state.width - 1) / 2) + abs(nxt[1] - (state.height - 1) / 2),
    }


def _nearest_distance(point: Point, targets, default: int = BIG_DISTANCE) -> int:
    return min((abs(point[0] - target[0]) + abs(point[1] - target[1]) for target in targets), default=default)
