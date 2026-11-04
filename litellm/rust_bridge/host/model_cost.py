from __future__ import annotations

import litellm


def current() -> dict[str, object]:
    return litellm.model_cost
