from typing import Any, Literal


def my_custom_validate(token: dict[str, Any]) -> Literal[True]:
    raise Exception("Custom validate failed")
