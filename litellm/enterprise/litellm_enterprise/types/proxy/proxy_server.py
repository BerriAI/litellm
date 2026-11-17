from typing import Literal

from typing_extensions import TypedDict


class CustomAuthSettings(TypedDict):
    mode: Literal["on", "off", "auto"]
