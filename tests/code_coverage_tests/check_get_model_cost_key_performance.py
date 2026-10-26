"""Check the warmed public/native resolver path against a 100x catalog increase."""

import os
from collections.abc import Callable
from functools import partial
from statistics import median
from timeit import repeat
from typing import Final
from unittest.mock import patch

with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
    import litellm
    from litellm import utils


def _measure(lookup: Callable[[str], str | None], size: int) -> tuple[float, ...]:
    rows: Final = {
        f"hosted_vllm/Model-{index}": {"litellm_provider": "hosted_vllm", "mode": "chat"} for index in range(size)
    }
    key: Final = f"hosted_vllm/Model-{size - 1}"
    queries: Final = ((key, key), (key.lower(), key), ("hosted_vllm/missing-model", None))
    with patch.object(litellm, "model_cost", rows):
        utils._invalidate_model_cost_lowercase_map()
        for query, expected in queries:
            assert lookup(query) == expected, (query, expected)
        return tuple(median(repeat(partial(lookup, query), repeat=5, number=2000)) for query, _ in queries)


def check_scaling(lookup: Callable[[str], str | None]) -> tuple[float, ...]:
    try:
        small: Final = _measure(lookup, 1000)
        large: Final = _measure(lookup, 100000)
        ratios: Final = tuple(after / before for before, after in zip(small, large, strict=True))
        assert max(ratios) < 8, f"Resolver time scaled with catalog size: exact/lowercase/miss ratios={ratios}"
        return ratios
    finally:
        utils._invalidate_model_cost_lowercase_map()


def main() -> None:
    ratios: Final = check_scaling(utils._get_model_cost_key)
    print(  # noqa: T201  # standalone performance check output
        f"PASS: actual _get_model_cost_key/native path, 100x rows, exact/lowercase/miss time ratios={ratios}"
    )


if __name__ == "__main__":
    main()
