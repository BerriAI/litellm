# Copyright 2022 Amethyst Reese
# Licensed under the MIT license

import inspect
import sys
from typing import Awaitable, Union

from .types import T

if sys.version_info < (3, 8):  # pragma: no cover
    from typing_extensions import Protocol
else:  # pragma: no cover
    from typing import Protocol


class Orderable(Protocol):  # pragma: no cover
    def __lt__(self, other):
        ...

    def __gt__(self, other):
        ...


async def maybe_await(object: Union[Awaitable[T], T]) -> T:
    if inspect.isawaitable(object):
        return await object  # type: ignore
    return object  # type: ignore
