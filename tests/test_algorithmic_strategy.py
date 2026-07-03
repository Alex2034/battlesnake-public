import unittest
from unittest.mock import patch

from battlesnake.evaluation import _wall_pressure
from battlesnake.ml_features import candidate_features
from battlesnake.model import score_features
from battlesnake.parser import from_game_state
from battlesnake.rules import simulate_turn, survivable_moves
from battlesnake.search import _limited_product
from battlesnake.strategy import choose_move_from_state
from logic import choose_move


def point(x, y):
    return {"x": x, "y": y}


def snake(snake_id, body, health=90):
    return {
        "id": snake_id,
        "name": snake_id,
        "health": health,
        "body": [point(x, y) for x, y in body],
        "head": point(*body[0]),
        "length": len(body),
    }


def game_state(you_body, enemies=(), food=(), width=11, height=11, health=90):
    you = snake("you", you_body, health=health)
    enemy_snakes = [snake(enemy_id, body, health=enemy_health) for enemy_id, body, enemy_health in enemies]
    return {
        "game": {"id": "game-id", "ruleset": {"name": "standard"}},
        "turn": 12,
        "board": {
            "height": height,
            "width": width,
            "food": [point(x, y) for x, y in food],
            "hazards": [],
            "snakes": [you] + enemy_snakes,
        },
        "you": you,
    }


class AlgorithmicStrategyTest(unittest.TestCase):
    def test_corner_escape_avoids_walls_and_body(self):
        state = game_state([(0, 0), (0, 1), (0, 2)])

        self.assertEqual(choose_move(state), "right")

    def test_avoids_losing_equal_head_to_head(self):
        state = game_state(
            [(5, 5), (5, 4), (5, 3)],
            enemies=(("enemy", [(5, 7), (5, 8), (5, 9)], 90),),
        )

        self.assertNotEqual(choose_move(state), "up")

    def test_hungry_snake_takes_safe_adjacent_food(self):
        state = game_state([(5, 5), (5, 4), (5, 3)], food=((6, 5),), health=12)

        self.assertEqual(choose_move(state), "right")

    def test_wall_pressure_penalizes_edges_not_center(self):
        state = from_game_state(game_state([(5, 5), (5, 4), (5, 3)]))

        self.assertGreater(_wall_pressure(state, (0, 5)), _wall_pressure(state, (5, 5)))

    def test_open_board_near_edge_does_not_prefer_wall(self):
        state = game_state([(1, 5), (1, 4), (1, 3)])

        self.assertEqual(choose_move(state), "right")

    def test_ml_model_scores_safe_food_move_higher_when_hungry(self):
        state = from_game_state(game_state([(5, 5), (5, 4), (5, 3)], food=((6, 5),), health=12))

        scores = {
            move: score_features(candidate_features(state, state.you, move))
            for move in survivable_moves(state, state.you)
        }

        self.assertGreater(scores["right"], scores["up"])
        self.assertGreater(scores["right"], scores["left"])

    def test_ml_failure_falls_back_to_algorithmic_scoring(self):
        state = from_game_state(game_state([(1, 5), (1, 4), (1, 3)]))

        with patch("battlesnake.strategy.candidate_features", side_effect=RuntimeError("model boom")):
            with self.assertLogs("battlesnake.strategy", level="WARNING"):
                self.assertEqual(choose_move_from_state(state), "right")

    def test_own_tail_is_survivable_when_it_moves(self):
        state = from_game_state(
            game_state(
                [(2, 3), (2, 2), (1, 2), (1, 3)],
                enemies=(("blocker", [(3, 3), (3, 2), (3, 1)], 90),),
                width=4,
                height=4,
            )
        )

        self.assertIn("left", survivable_moves(state, state.you))
        self.assertEqual(choose_move_from_state(state), "left")

    def test_equal_head_to_head_kills_both_snakes(self):
        state = from_game_state(
            game_state(
                [(1, 1), (1, 0), (0, 0)],
                enemies=(("enemy", [(3, 1), (3, 0), (2, 0)], 90),),
                width=5,
                height=5,
            )
        )

        next_state = simulate_turn(state, {"you": "right", "enemy": "left"})

        self.assertIsNone(next_state.snake_by_id("you"))
        self.assertIsNone(next_state.snake_by_id("enemy"))

    def test_enemy_combination_limit_samples_full_range(self):
        options = [["A1", "A2"], ["B1", "B2"], ["C1", "C2"], ["D1", "D2"], ["E1", "E2"]]

        combos = list(_limited_product(options, 18))

        self.assertEqual(len(combos), 18)
        self.assertIn(("A1", "B1", "C1", "D1", "E1"), combos)
        self.assertIn(("A2", "B2", "C2", "D2", "E2"), combos)

    def test_invalid_payload_falls_back_to_safe_move(self):
        with self.assertLogs("battlesnake.strategy", level="WARNING"):
            self.assertEqual(choose_move({}), "up")


if __name__ == "__main__":
    unittest.main()
