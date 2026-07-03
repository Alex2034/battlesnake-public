"""Pathfinding primitives used by strategy and evaluation."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Optional, Set

from .types import DIRECTIONS, Point


def neighbors(point: Point):
    x, y = point
    for dx, dy in DIRECTIONS.values():
        yield x + dx, y + dy


def bfs_distances(
    starts: Iterable[Point],
    blocked: Set[Point],
    width: int,
    height: int,
    limit: Optional[int] = None,
) -> Dict[Point, int]:
    dist: Dict[Point, int] = {}
    queue = deque()
    for start in starts:
        if not _in_bounds(start, width, height) or start in dist:
            continue
        dist[start] = 0
        queue.append(start)

    while queue:
        point = queue.popleft()
        if limit is not None and len(dist) >= limit:
            break
        for nxt in neighbors(point):
            if nxt in dist or nxt in blocked or not _in_bounds(nxt, width, height):
                continue
            dist[nxt] = dist[point] + 1
            queue.append(nxt)
    return dist


def flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: Optional[int] = None) -> int:
    return len(bfs_distances([start], blocked, width, height, limit=limit))


def shortest_distance(start: Point, goal: Point, blocked: Set[Point], width: int, height: int) -> Optional[int]:
    distances = bfs_distances([start], blocked, width, height)
    return distances.get(goal)


def count_exits(point: Point, blocked: Set[Point], width: int, height: int) -> int:
    return sum(1 for nxt in neighbors(point) if _in_bounds(nxt, width, height) and nxt not in blocked)


def _in_bounds(point: Point, width: int, height: int) -> bool:
    x, y = point
    return 0 <= x < width and 0 <= y < height

