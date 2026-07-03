"""Battlesnake compatibility entrypoint.

The algorithmic engine lives in the ``battlesnake`` package. This module keeps
the public API expected by ``backend.py`` small and stable.
"""

from typing import Dict

from battlesnake.strategy import choose_move


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "maria",
        "color": "#1f8f5f",
        "head": "beluga",
        "tail": "bolt",
        "version": "1.1.0-hybrid",
    }
