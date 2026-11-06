# pyright: reportUnnecessaryTypeIgnoreComment=error
"""Type checked by `make lint-e2e-basedpyright`, never run: the ignore errors once the stub accepts any argument."""

from litellm import Router


def accepts_a_constructor_argument() -> Router:
    return Router(model_list=[], num_retries=2)


def rejects_an_unknown_argument() -> Router:
    return Router(model_list=[], not_a_parameter=1)  # pyright: ignore[reportCallIssue]  # what this file checks
