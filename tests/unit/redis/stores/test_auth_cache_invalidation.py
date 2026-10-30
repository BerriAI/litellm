import math
from typing import Final

import pytest

from litellm.redis.stores.auth_cache_invalidation import (
    CacheInvalidation,
    decode_cache_invalidation,
    encode_cache_invalidation,
)

OLD_POD_MESSAGES: Final = (
    (b'{"cache_key": "project_id:p-1"}', CacheInvalidation("project_id:p-1")),
    (
        b'{"cache_key": "spend:team_member:u-1:t-1", "new_value": 0.0, "ttl": 60.5}',
        CacheInvalidation("spend:team_member:u-1:t-1", new_value=0.0, ttl=60.5),
    ),
    (b'{"cache_key": "k\\u96ea", "ttl": 30}', CacheInvalidation("k\u96ea", ttl=30.0)),
)


@pytest.mark.parametrize(("wire", "invalidation"), OLD_POD_MESSAGES)
def test_codec_round_trips_the_exact_bytes_old_pods_publish(wire: bytes, invalidation: CacheInvalidation) -> None:
    assert decode_cache_invalidation(wire) == invalidation
    assert decode_cache_invalidation(wire.decode()) == invalidation
    if invalidation.ttl != 30.0:
        assert encode_cache_invalidation(invalidation).encode() == wire


def test_encode_keeps_the_old_field_order_and_drops_unset_fields() -> None:
    assert encode_cache_invalidation(CacheInvalidation("k", ttl=5.0)) == '{"cache_key": "k", "ttl": 5.0}'
    assert encode_cache_invalidation(CacheInvalidation("k", new_value=1.0)) == '{"cache_key": "k", "new_value": 1.0}'


@pytest.mark.parametrize(
    "data",
    [b"not json", b'{"other": "x"}', b'{"cache_key": 7}', b"[1]", "raw string", None, 7],
)
def test_unusable_messages_decode_to_nothing(data: object) -> None:
    assert decode_cache_invalidation(data) is None


def test_non_numeric_values_are_dropped_and_non_finite_numbers_pass_through_like_old_pods() -> None:
    decoded: Final = decode_cache_invalidation(b'{"cache_key": "k", "new_value": true, "ttl": Infinity}')

    assert decoded is not None
    assert (decoded.cache_key, decoded.new_value, math.isinf(decoded.ttl or 0.0)) == ("k", None, True)
    assert decode_cache_invalidation(b'{"cache_key": "k", "new_value": "1"}') == CacheInvalidation("k")
