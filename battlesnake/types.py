"""Shared domain types for the algorithmic Battlesnake engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class Snake:
    id: str
    health: int
    body: Tuple[Point, ...]
    length: int

    @property
    def alive(self) -> bool:
        return self.health > 0 and bool(self.body)

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def tail(self) -> Point:
        return self.body[-1]


@dataclass(frozen=True)
class BoardState:
    width: int
    height: int
    food: FrozenSet[Point]
    hazards: FrozenSet[Point]
    snakes: Tuple[Snake, ...]
    you_id: str
    turn: int

    @property
    def you(self) -> Optional[Snake]:
        return self.snake_by_id(self.you_id)

    @property
    def alive_snakes(self) -> Tuple[Snake, ...]:
        return tuple(snake for snake in self.snakes if snake.alive)

    @property
    def enemies(self) -> Tuple[Snake, ...]:
        return tuple(snake for snake in self.alive_snakes if snake.id != self.you_id)

    def snake_by_id(self, snake_id: str) -> Optional[Snake]:
        for snake in self.snakes:
            if snake.id == snake_id and snake.alive:
                return snake
        return None

    def in_bounds(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def occupied(self, include_tails: bool = True, exclude: Optional[Iterable[Point]] = None) -> FrozenSet[Point]:
        excluded = set(exclude or ())
        cells = set()
        for snake in self.alive_snakes:
            body = snake.body if include_tails else snake.body[:-1]
            cells.update(point for point in body if point not in excluded)
        return frozenset(cells)

