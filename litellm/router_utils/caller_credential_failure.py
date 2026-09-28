"""
Tags a local auth failure raised because the request carried no provider credential at all.

A deployment configured without its own key relies on each caller forwarding one, for example a
Claude subscription OAuth bearer. A caller that forwards none fails before any provider call, so the
failure says nothing about the deployment's health and must not cool it down for the callers that do
send a credential.

Kept free of router imports so provider transformations can tag the exception without an import cycle.
"""

from typing import Final

_MISSING_CALLER_CREDENTIAL_ATTR: Final = "_litellm_missing_caller_credential"


def mark_missing_caller_credential(exception: BaseException) -> None:
    setattr(exception, _MISSING_CALLER_CREDENTIAL_ATTR, True)


def is_missing_caller_credential(exception: BaseException | None) -> bool:
    return bool(getattr(exception, _MISSING_CALLER_CREDENTIAL_ATTR, False))
