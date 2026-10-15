from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, cast

import litellm
from litellm.responses import main as python_responses

PYTHON_ARESPONSES: Final = cast(Callable[..., Awaitable[object]], python_responses.aresponses)  # noqa: TID251  # the Python baseline under comparison
PUBLIC_ARESPONSES: Final = cast(Callable[..., Awaitable[object]], litellm.aresponses)
PYTHON_RESPONSES: Final = cast(Callable[..., object], python_responses.responses)  # noqa: TID251  # the Python baseline under comparison
PUBLIC_RESPONSES: Final = cast(Callable[..., object], litellm.responses)


@dataclass(slots=True)
class FallbackProbe:
    calls: int = 0
