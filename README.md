# Battlesnake Hybrid Bot

A [Battlesnake](https://play.battlesnake.com) written in Python and Flask.
This version combines a hand-built algorithmic engine with an embedded linear
ML move-ranking model.

## What It Does

Each turn, the bot:

- Parses the Battlesnake JSON payload into a normalized board state.
- Removes immediately losing moves.
- Builds enemy head-to-head threat maps.
- Scores space, exits, tail reachability, food urgency, walls, and territory.
- Runs a small adversarial lookahead over plausible enemy replies.
- Adds a bounded ML rank bonus for safe candidate moves.
- Returns the strongest risk-adjusted move.

## Architecture

- `backend.py` - Battlesnake HTTP server with `/`, `/start`, `/move`, and `/end`.
- `logic.py` - compatibility entrypoint used by `backend.py`.
- `battlesnake/types.py` - domain types: points, snakes, board state.
- `battlesnake/parser.py` - JSON payload to internal state.
- `battlesnake/rules.py` - legal moves and simultaneous turn simulation.
- `battlesnake/pathfinding.py` - BFS, flood fill, exits, shortest paths.
- `battlesnake/danger.py` - enemy threat maps and head-to-head danger.
- `battlesnake/evaluation.py` - hand-tuned position evaluation.
- `battlesnake/ml_features.py` - feature extraction for the embedded model.
- `battlesnake/model.py` - standardized linear move-ranking checkpoint.
- `battlesnake/search.py` - bounded adversarial lookahead.
- `battlesnake/strategy.py` - top-level move selection.
- `tests/` - focused tactical scenario tests.

## Run Locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backend.py
```

Test your battlesnake with the Battlesnake CLI:

```bash
battlesnake play -W 11 -H 11 \
  -n algorithmic -u http://localhost:8000 \
  -g solo \
  -v -c -d 300
```

Run the local test suite:

```bash
python3 -m unittest
```

## Deploy to Render

1. Push this repo to GitHub.
2. In the [Render dashboard](https://dashboard.render.com), create a Blueprint
   from the repo. Render reads `render.yaml` and provisions a free web service
   running `gunicorn backend:app`.
3. Or create a Web Service manually with build command `pip install -r
   requirements.txt` and start command `gunicorn backend:app --bind
   0.0.0.0:$PORT`.
4. Visit the public URL in a browser. You should see the Battlesnake appearance
   JSON.

## Register on Battlesnake

1. Create an account at [play.battlesnake.com](https://play.battlesnake.com).
2. Create a Battlesnake and paste your Render URL as the server URL.
3. Use it in a game.
