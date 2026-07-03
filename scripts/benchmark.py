"""Run deterministic move-selection benchmarks for the Battlesnake bot."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from battlesnake import strategy
from battlesnake.ml_features import candidate_features
from battlesnake.model import score_features
from battlesnake.parser import from_game_state
from battlesnake.rules import candidate_moves, next_head, simulate_turn, survivable_moves
from battlesnake.types import DIRECTIONS, Point


@dataclass(frozen=True)
class Case:
    name: str
    state: Dict
    expected: Optional[str] = None
    forbidden: Optional[str] = None
    curated: bool = False


def point(x: int, y: int) -> Dict[str, int]:
    return {"x": x, "y": y}


def snake(snake_id: str, body: Sequence[Point], health: int = 90) -> Dict:
    return {
        "id": snake_id,
        "name": snake_id,
        "health": health,
        "body": [point(x, y) for x, y in body],
        "head": point(*body[0]),
        "length": len(body),
    }


def game_state(
    you_body: Sequence[Point],
    enemies: Iterable[Tuple[str, Sequence[Point], int]] = (),
    food: Iterable[Point] = (),
    width: int = 11,
    height: int = 11,
    health: int = 90,
    turn: int = 12,
) -> Dict:
    you = snake("you", you_body, health=health)
    enemy_snakes = [snake(enemy_id, body, enemy_health) for enemy_id, body, enemy_health in enemies]
    return {
        "game": {"id": "benchmark-game", "ruleset": {"name": "standard"}},
        "turn": turn,
        "board": {
            "height": height,
            "width": width,
            "food": [point(x, y) for x, y in food],
            "hazards": [],
            "snakes": [you] + enemy_snakes,
        },
        "you": you,
    }


def curated_cases() -> List[Case]:
    return [
        Case("corner_escape", game_state([(0, 0), (0, 1), (0, 2)]), expected="right", curated=True),
        Case(
            "avoid_equal_head_to_head",
            game_state([(5, 5), (5, 4), (5, 3)], enemies=(("enemy", [(5, 7), (5, 8), (5, 9)], 90),)),
            forbidden="up",
            curated=True,
        ),
        Case(
            "hungry_adjacent_food",
            game_state([(5, 5), (5, 4), (5, 3)], food=((6, 5),), health=12),
            expected="right",
            curated=True,
        ),
        Case(
            "own_tail_escape",
            game_state(
                [(2, 3), (2, 2), (1, 2), (1, 3)],
                enemies=(("blocker", [(3, 3), (3, 2), (3, 1)], 90),),
                width=4,
                height=4,
            ),
            expected="left",
            curated=True,
        ),
        Case("near_edge_open_board", game_state([(1, 5), (1, 4), (1, 3)]), expected="right", curated=True),
        Case("stacked_start", game_state([(5, 5), (5, 5), (5, 5)], health=100, turn=0), curated=True),
    ]


def random_cases(count: int, seed: int) -> List[Case]:
    rng = random.Random(seed)
    cases: List[Case] = []
    attempts = 0
    occupied: set[Point]
    while len(cases) < count and attempts < count * 80:
        attempts += 1
        width = rng.choice([7, 9, 11])
        height = rng.choice([7, 9, 11])
        occupied = set()
        you_body = _random_body(rng, width, height, rng.randint(3, 6), occupied)
        if you_body is None:
            continue
        occupied.update(you_body)

        enemies = []
        for enemy_index in range(rng.randint(0, 3)):
            body = _random_body(rng, width, height, rng.randint(3, 6), occupied)
            if body is None:
                continue
            occupied.update(body)
            enemies.append((f"enemy-{enemy_index}", body, rng.randint(30, 100)))

        empty = [(x, y) for x in range(width) for y in range(height) if (x, y) not in occupied]
        rng.shuffle(empty)
        food = empty[: rng.randint(0, min(4, len(empty)))]
        state = game_state(
            you_body,
            enemies=tuple(enemies),
            food=tuple(food),
            width=width,
            height=height,
            health=rng.randint(5, 100),
            turn=rng.randint(0, 250),
        )
        parsed = from_game_state(state)
        if parsed.you is not None and survivable_moves(parsed, parsed.you):
            cases.append(Case(f"random_{len(cases):03d}", state))
    return cases


def _random_body(rng: random.Random, width: int, height: int, length: int, occupied: set[Point]) -> Optional[Tuple[Point, ...]]:
    directions = list(DIRECTIONS.values())
    for _ in range(80):
        dx, dy = rng.choice(directions)
        head = (rng.randrange(width), rng.randrange(height))
        body = tuple((head[0] - dx * index, head[1] - dy * index) for index in range(length))
        if all(0 <= x < width and 0 <= y < height for x, y in body) and not occupied.intersection(body):
            return body
    return None


def benchmark(cases: Sequence[Case], mode: str) -> Dict:
    if mode == "model_only":
        return _run_cases(cases, mode)

    old_bonus = strategy.ML_RANK_BONUS
    strategy.ML_RANK_BONUS = 0.0 if mode == "algorithm_only" else old_bonus
    try:
        return _run_cases(cases, mode)
    finally:
        strategy.ML_RANK_BONUS = old_bonus


def _run_cases(cases: Sequence[Case], mode: str) -> Dict:
    latencies_ms: List[float] = []
    immediate_deaths = 0
    invalid_moves = 0
    unsafe_moves = 0
    expectation_failures = []
    moves_by_case = {}

    for case in cases:
        parsed = from_game_state(case.state)
        you = parsed.you
        start = time.perf_counter()
        move = _choose_move(case.state, mode)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        moves_by_case[case.name] = move

        if you is None or move not in DIRECTIONS or not parsed.in_bounds(next_head(you, move)):
            invalid_moves += 1
            immediate_deaths += 1
            continue

        if move not in survivable_moves(parsed, you):
            unsafe_moves += 1

        next_state = simulate_turn(parsed, {you.id: move})
        if next_state.you is None:
            immediate_deaths += 1

        if case.expected is not None and move != case.expected:
            expectation_failures.append({"case": case.name, "expected": case.expected, "actual": move})
        if case.forbidden is not None and move == case.forbidden:
            expectation_failures.append({"case": case.name, "forbidden": case.forbidden, "actual": move})

    return {
        "mode": mode,
        "cases": len(cases),
        "curated_cases": sum(1 for case in cases if case.curated),
        "safe_move_rate": _rate(len(cases) - unsafe_moves, len(cases)),
        "immediate_death_rate": _rate(immediate_deaths, len(cases)),
        "invalid_move_rate": _rate(invalid_moves, len(cases)),
        "expectation_failures": expectation_failures,
        "latency_ms": {
            "avg": round(statistics.mean(latencies_ms), 3),
            "p50": round(statistics.median(latencies_ms), 3),
            "p95": round(_percentile(latencies_ms, 0.95), 3),
            "max": round(max(latencies_ms), 3),
        },
        "curated_moves": {case.name: moves_by_case[case.name] for case in cases if case.curated},
    }


def _choose_move(state: Dict, mode: str) -> str:
    if mode == "model_only":
        return _choose_model_only(state)
    return strategy.choose_move(state)


def _choose_model_only(state: Dict) -> str:
    parsed = from_game_state(state)
    you = parsed.you
    if you is None:
        return "up"

    moves = survivable_moves(parsed, you)
    if not moves:
        options = candidate_moves(parsed, you)
        return options[0] if options else "up"

    scored = []
    for move in moves:
        scored.append((score_features(candidate_features(parsed, you, move)), move))
    scored.sort(key=lambda item: (item[0], _direction_preference(item[1])), reverse=True)
    return scored[0][1]


def _direction_preference(move: str) -> int:
    return {"up": 4, "right": 3, "left": 2, "down": 1}.get(move, 0)


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    cases = curated_cases() + random_cases(args.random_cases, args.seed)
    results = {
        "seed": args.seed,
        "requested_random_cases": args.random_cases,
        "actual_random_cases": len(cases) - len(curated_cases()),
        "results": [
            benchmark(cases, "algorithm_only"),
            benchmark(cases, "model_only"),
            benchmark(cases, "hybrid"),
        ],
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
