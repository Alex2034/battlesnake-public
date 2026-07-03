"""Battlesnake movement rules and deterministic turn simulation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterable, List, Set

from .types import BoardState, DIRECTIONS, Point, Snake

HAZARD_DAMAGE = 14


def next_head(snake: Snake, move: str) -> Point:
    dx, dy = DIRECTIONS[move]
    return snake.head[0] + dx, snake.head[1] + dy


def candidate_moves(state: BoardState, snake: Snake) -> List[str]:
    return [move for move in DIRECTIONS if state.in_bounds(next_head(snake, move))]


def survivable_moves(state: BoardState, snake: Snake) -> List[str]:
    return [move for move in candidate_moves(state, snake) if not is_body_collision(state, snake, move)]


def is_body_collision(state: BoardState, snake: Snake, move: str) -> bool:
    nxt = next_head(snake, move)
    if not state.in_bounds(nxt):
        return True

    eating = nxt in state.food
    occupied: Set[Point] = set()
    for other in state.alive_snakes:
        if other.id == snake.id:
            body = other.body if eating else other.body[:-1]
        else:
            body = other.body
        occupied.update(body)
    return nxt in occupied


def simulate_turn(state: BoardState, moves: Dict[str, str]) -> BoardState:
    """Apply one simultaneous Battlesnake turn.

    Missing enemy moves are filled with their first survivable move. This keeps
    search robust when an enemy is already trapped.
    """
    alive = state.alive_snakes
    chosen = {snake.id: moves.get(snake.id) for snake in alive}
    for snake in alive:
        if chosen[snake.id] not in DIRECTIONS:
            options = survivable_moves(state, snake) or candidate_moves(state, snake)
            chosen[snake.id] = options[0] if options else "up"

    new_heads = {snake.id: next_head(snake, chosen[snake.id]) for snake in alive}
    ate_food = {snake.id: new_heads[snake.id] in state.food for snake in alive}

    collision_bodies: Set[Point] = set()
    for snake in alive:
        body = snake.body if ate_food[snake.id] else snake.body[:-1]
        collision_bodies.update(body)

    dead: Set[str] = set()
    for snake in alive:
        head = new_heads[snake.id]
        if not state.in_bounds(head):
            dead.add(snake.id)
            continue
        if head in collision_bodies:
            dead.add(snake.id)

    head_groups: Dict[Point, List[Snake]] = defaultdict(list)
    for snake in alive:
        if snake.id not in dead:
            head_groups[new_heads[snake.id]].append(snake)

    for snakes in head_groups.values():
        if len(snakes) < 2:
            continue
        max_length = max(_future_length(snake, ate_food[snake.id]) for snake in snakes)
        winners = [snake for snake in snakes if _future_length(snake, ate_food[snake.id]) == max_length]
        if len(winners) == 1:
            dead.update(snake.id for snake in snakes if snake.id != winners[0].id)
        else:
            dead.update(snake.id for snake in snakes)

    new_food = set(state.food)
    new_snakes = []
    for snake in alive:
        if snake.id in dead:
            continue
        head = new_heads[snake.id]
        ate = ate_food[snake.id]
        health = 100 if ate else snake.health - 1
        if head in state.hazards:
            health -= HAZARD_DAMAGE
        if health <= 0:
            continue
        body = (head,) + (snake.body if ate else snake.body[:-1])
        new_snakes.append(replace(snake, health=health, body=body, length=len(body)))
        if ate:
            new_food.discard(head)

    return replace(state, food=frozenset(new_food), snakes=tuple(new_snakes), turn=state.turn + 1)


def _future_length(snake: Snake, ate: bool) -> int:
    return snake.length + (1 if ate else 0)


def move_dict(you: Snake, our_move: str, enemies: Iterable[Snake], enemy_moves: Iterable[str]) -> Dict[str, str]:
    moves = {you.id: our_move}
    moves.update({enemy.id: move for enemy, move in zip(enemies, enemy_moves)})
    return moves

