"""Top-level algorithmic strategy."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .danger import build_threat_map
from .ml_features import candidate_features
from .model import score_features
from .parser import from_game_state
from .rules import candidate_moves, next_head, survivable_moves
from .search import SearchConfig, score_move
from .types import BoardState, DIRECTIONS

log = logging.getLogger("battlesnake.strategy")

CONFIG = SearchConfig()
ML_RANK_BONUS = 25.0


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

    algorithm_scores: List[Tuple[float, str]] = []
    ml_scores: Dict[str, float] = {}
    for move in candidates:
        algorithm_scores.append((score_move(state, move, CONFIG), move))
        ml_score = _ml_score(state, move)
        if ml_score is not None:
            ml_scores[move] = ml_score

    ml_bonuses = _ml_rank_bonuses(ml_scores)
    scored = [(algorithm_score + ml_bonuses.get(move, 0.0), move) for algorithm_score, move in algorithm_scores]

    if not scored:
        return _least_bad_move(state)

    scored.sort(key=lambda item: (item[0], _direction_preference(item[1])), reverse=True)
    best_score, best_move = scored[0]
    log.info("scored moves=%s best=%s %.1f", scored, best_move, best_score)
    return best_move


def _ml_score(state: BoardState, move: str) -> Optional[float]:
    you = state.you
    if you is None:
        return None
    try:
        return score_features(candidate_features(state, you, move))
    except Exception as exc:  # noqa: BLE001 - model advice must never break play
        log.warning("ml scoring failed for move=%s: %s", move, exc)
        return None


def _ml_rank_bonuses(scores: Dict[str, float]) -> Dict[str, float]:
    ranked = sorted(scores.items(), key=lambda item: (item[1], _direction_preference(item[0])), reverse=True)
    return {move: (len(ranked) - rank) * ML_RANK_BONUS for rank, (move, _) in enumerate(ranked, start=1)}


def _least_bad_move(state: BoardState) -> str:
    you = state.you
    if you is None:
        return "up"
    moves = candidate_moves(state, you) or list(DIRECTIONS)
    moves.sort(key=lambda move: _direction_preference(move), reverse=True)
    return moves[0]


def _direction_preference(move: str) -> int:
    return {"up": 4, "right": 3, "left": 2, "down": 1}.get(move, 0)
