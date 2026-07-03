"""Small adversarial lookahead for move selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Sequence, Tuple

from .evaluation import DEATH_SCORE, evaluate_state
from .pathfinding import flood_fill
from .rules import move_dict, next_head, simulate_turn, survivable_moves
from .types import BoardState, Snake


@dataclass(frozen=True)
class SearchConfig:
    depth: int = 2
    enemy_branch_limit: int = 2
    max_enemy_combinations: int = 18


def score_move(state: BoardState, move: str, config: SearchConfig) -> float:
    you = state.you
    if you is None:
        return DEATH_SCORE
    return _score_move(state, you, move, config.depth, config)


def _score_move(state: BoardState, you: Snake, move: str, depth: int, config: SearchConfig) -> float:
    enemies = state.enemies
    enemy_options = [_rank_enemy_moves(state, enemy, config.enemy_branch_limit) for enemy in enemies]
    if not enemy_options:
        next_state = simulate_turn(state, {you.id: move})
        return _score_state(next_state, depth - 1, config, previous_head=you.head)

    worst = float("inf")
    for combo in _limited_product(enemy_options, config.max_enemy_combinations):
        next_state = simulate_turn(state, move_dict(you, move, enemies, combo))
        score = _score_state(next_state, depth - 1, config, previous_head=you.head)
        worst = min(worst, score)
    return worst


def _score_state(state: BoardState, depth: int, config: SearchConfig, previous_head=None) -> float:
    you = state.you
    if you is None:
        return DEATH_SCORE

    immediate = evaluate_state(state, previous_head=previous_head).score
    if depth <= 0:
        return immediate

    moves = survivable_moves(state, you)
    if not moves:
        return DEATH_SCORE

    future = max(_score_move(state, you, move, depth - 1, config) for move in moves)
    return immediate + 0.45 * future


def _rank_enemy_moves(state: BoardState, enemy: Snake, limit: int) -> Sequence[str]:
    moves = survivable_moves(state, enemy)
    if not moves:
        return ["up"]

    blocked = set(state.occupied(exclude={enemy.head}))
    scored: List[Tuple[float, str]] = []
    you = state.you
    for move in moves:
        head = next_head(enemy, move)
        space = flood_fill(head, blocked, state.width, state.height, limit=enemy.length + 6)
        food_bonus = 0.0
        if enemy.health < 45 and state.food:
            nearest = min(abs(head[0] - food[0]) + abs(head[1] - food[1]) for food in state.food)
            food_bonus = max(0.0, 30.0 - nearest * 4.0)
        pressure = 0.0
        if you is not None:
            dist = abs(head[0] - you.head[0]) + abs(head[1] - you.head[1])
            pressure = max(0.0, 5.0 - dist)
        scored.append((space * 3.0 + food_bonus + pressure, move))

    scored.sort(reverse=True)
    return [move for _, move in scored[:limit]]


def _limited_product(options: Iterable[Sequence[str]], limit: int):
    option_sets = [tuple(option) for option in options]
    if not option_sets or limit <= 0:
        return []

    total = 1
    for option in option_sets:
        total *= len(option)

    if total <= limit:
        return list(product(*option_sets))

    if limit == 1:
        return [_product_at_index(option_sets, 0)]

    indexes = {
        round(i * (total - 1) / (limit - 1))
        for i in range(limit)
    }
    return [_product_at_index(option_sets, index) for index in sorted(indexes)]


def _product_at_index(options: Sequence[Sequence[str]], index: int) -> Tuple[str, ...]:
    combo = []
    for option in reversed(options):
        index, remainder = divmod(index, len(option))
        combo.append(option[remainder])
    return tuple(reversed(combo))
