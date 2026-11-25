from __future__ import annotations

import random


def next_float() -> float:
    return random.random()


def index_below(length: int) -> int:
    return random.randrange(length)
