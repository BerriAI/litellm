import typing as t


T = t.TypeVar("T")


def choice(choices: t.Iterable) -> t.Callable[[T], None]:
    """
    A validator that checks if the value is one of the choices.
    """

    def validate(value):
        # type (T) -> None
        if value is not None and value not in choices:
            raise ValueError("value must be one of %r" % sorted(choices))

    return validate


def range(min_value: int, max_value: int) -> t.Callable[[T], None]:
    """
    A validator that checks if the value is in the range.
    """

    def validate(value):
        # type (T) -> None
        if value is not None and not (min_value <= value <= max_value):
            raise ValueError("value must be in range [%r, %r]" % (min_value, max_value))

    return validate
