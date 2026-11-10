import itertools
from collections.abc import Iterator
from typing import Final, TypeVar

import httpx
from openai import APIError

_EXCEPTION_CHAIN_MAX_DEPTH: Final = 16
_REQUEST_FAILURE_TYPES: Final = (APIError, httpx.HTTPError)

_RejectionT: Final = TypeVar("_RejectionT", bound=BaseException)


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    link: BaseException | None = error  # rebind-ok: advances one link of the cause chain per iteration
    for _ in range(_EXCEPTION_CHAIN_MAX_DEPTH):
        if link is None:
            return
        yield link
        link = link.__cause__ or link.__context__


def _ends_the_failed_attempt(
    link: BaseException, failed_attempt: BaseException, rejection_type: type[BaseException]
) -> bool:
    return not isinstance(link, rejection_type) and (
        isinstance(link, _REQUEST_FAILURE_TYPES) or type(link) is type(failed_attempt)
    )


def _links_within_the_failed_attempt(
    error: BaseException, rejection_type: type[BaseException]
) -> Iterator[BaseException]:
    yield error
    yield from itertools.takewhile(
        lambda link: not _ends_the_failed_attempt(link, error, rejection_type),
        itertools.islice(_exception_chain(error), 1, None),
    )


def credential_rejection_in(error: BaseException, rejection_type: type[_RejectionT]) -> _RejectionT | None:
    return next(
        (link for link in _links_within_the_failed_attempt(error, rejection_type) if isinstance(link, rejection_type)),
        None,
    )
