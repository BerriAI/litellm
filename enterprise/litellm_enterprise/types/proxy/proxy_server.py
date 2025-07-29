from typing import Literal, TypedDict


class CustomAuthSettings(TypedDict):
    mode: Literal["on", "off", "auto"]
