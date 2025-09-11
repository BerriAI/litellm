import time
from typing import Optional

from .api_jwk import PyJWKSet, PyJWTSetWithTimestamp


class JWKSetCache:
    def __init__(self, lifespan: int) -> None:
        self.jwk_set_with_timestamp: Optional[PyJWTSetWithTimestamp] = None
        self.lifespan = lifespan

    def put(self, jwk_set: PyJWKSet) -> None:
        if jwk_set is not None:
            self.jwk_set_with_timestamp = PyJWTSetWithTimestamp(jwk_set)
        else:
            # clear cache
            self.jwk_set_with_timestamp = None

    def get(self) -> Optional[PyJWKSet]:
        if self.jwk_set_with_timestamp is None or self.is_expired():
            return None

        return self.jwk_set_with_timestamp.get_jwk_set()

    def is_expired(self) -> bool:
        return (
            self.jwk_set_with_timestamp is not None
            and self.lifespan > -1
            and time.monotonic()
            > self.jwk_set_with_timestamp.get_timestamp() + self.lifespan
        )
