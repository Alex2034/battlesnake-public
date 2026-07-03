"""Enemy threat maps and tactical danger detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Set

from .rules import next_head, survivable_moves
from .types import BoardState, Point


@dataclass(frozen=True)
class ThreatMap:
    any_enemy: FrozenSet[Point]
    lethal_head_to_head: FrozenSet[Point]
    favorable_head_to_head: FrozenSet[Point]


def build_threat_map(state: BoardState) -> ThreatMap:
    you = state.you
    if you is None:
        return ThreatMap(frozenset(), frozenset(), frozenset())

    any_enemy: Set[Point] = set()
    lethal: Set[Point] = set()
    favorable: Set[Point] = set()

    for enemy in state.enemies:
        for move in survivable_moves(state, enemy):
            point = next_head(enemy, move)
            enemy_length = enemy.length + int(point in state.food)
            you_length = you.length + int(point in state.food)
            any_enemy.add(point)
            if enemy_length >= you_length:
                lethal.add(point)
            else:
                favorable.add(point)

    return ThreatMap(frozenset(any_enemy), frozenset(lethal), frozenset(favorable))
