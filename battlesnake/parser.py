"""Translate Battlesnake JSON payloads into internal domain objects."""

from __future__ import annotations

from typing import Dict

from .types import BoardState, Point, Snake


def _point(raw: Dict) -> Point:
    return raw["x"], raw["y"]


def from_game_state(game_state: Dict) -> BoardState:
    board = game_state["board"]
    snakes = tuple(
        Snake(
            id=snake["id"],
            health=snake["health"],
            body=tuple(_point(segment) for segment in snake["body"]),
            length=snake.get("length", len(snake["body"])),
        )
        for snake in board["snakes"]
    )

    return BoardState(
        width=board["width"],
        height=board["height"],
        food=frozenset(_point(food) for food in board.get("food", [])),
        hazards=frozenset(_point(hazard) for hazard in board.get("hazards", [])),
        snakes=snakes,
        you_id=game_state["you"]["id"],
        turn=game_state.get("turn", 0),
    )

