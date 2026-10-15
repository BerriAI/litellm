import pytest


def gap(reason: str, raises: type[Exception] = AssertionError) -> pytest.MarkDecorator:
    return pytest.mark.xfail(reason=reason, raises=raises, strict=True)
