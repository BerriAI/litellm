from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

_lens_analysis: Final = ContextVar("litellm_lens_analysis", default=False)


def is_lens_analysis() -> bool:
    return _lens_analysis.get()


@contextmanager
def lens_analysis() -> Iterator[None]:
    token: Final = _lens_analysis.set(True)
    try:
        yield
    finally:
        _lens_analysis.reset(token)
