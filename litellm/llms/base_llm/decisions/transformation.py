from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecisionsEndpoint:
    default_api_base: str
    path: str
    api_key_env: tuple[str, ...]
    api_base_env: str
