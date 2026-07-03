"""Top-level algorithmic strategy."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .danger import build_threat_map
from .parser import from_game_state
from .rules import candidate_moves, next_head, survivable_moves
from .search import SearchConfig, score_move
from .types import BoardState, DIRECTIONS

log = logging.getLogger("battlesnake.strategy")

CONFIG = SearchConfig()


def choose_move(game_state: Dict) -> str:
    try:
        state = from_game_state(game_state)
    except (KeyError, TypeError, IndexError) as exc:
        log.warning("invalid game state payload: %s", exc)
        return "up"
    return choose_move_from_state(state)


def choose_move_from_state(state: BoardState) -> str:
    you = state.you
    if you is None:
        return "up"

    legal = survivable_moves(state, you)
    if not legal:
        return _least_bad_move(state)

    threats = build_threat_map(state)
    non_lethal = [move for move in legal if next_head(you, move) not in threats.lethal_head_to_head]
    candidates = non_lethal or legal

    scored: List[Tuple[float, str]] = []
    for move in candidates:
        scored.append((score_move(state, move, CONFIG), move))

    if not scored:
        return _least_bad_move(state)

    scored.sort(key=lambda item: (item[0], _direction_preference(item[1])), reverse=True)
    best_score, best_move = scored[0]
    log.info("scored moves=%s best=%s %.1f", scored, best_move, best_score)
    return best_move


def _least_bad_move(state: BoardState) -> str:
    you = state.you
    if you is None:
        return "up"
    moves = candidate_moves(state, you) or list(DIRECTIONS)
    moves.sort(key=lambda move: _direction_preference(move), reverse=True)
    return moves[0]


def _direction_preference(move: str) -> int:
    return {"up": 4, "right": 3, "left": 2, "down": 1}.get(move, 0)
