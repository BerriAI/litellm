from __future__ import annotations

import os
import sys

import fakeredis
import pytest
from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.request import Request
from vcr.serializers import yamlserializer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests._vcr_redis_persister import (  # noqa: E402
    CASSETTE_TTL_SECONDS,
    filter_non_2xx_response,
    make_redis_persister,
    redis_key_for,
)


def _sample_cassette_dict():
    request = Request(
        method="POST",
        uri="https://api.anthropic.com/v1/messages",
        body=b'{"model":"claude","messages":[{"role":"user","content":"hi"}]}',
        headers={"content-type": "application/json"},
    )
    response = {
        "status": {"code": 200, "message": "OK"},
        "headers": {"content-type": ["application/json"]},
        "body": {"string": b'{"id":"msg_1","type":"message"}'},
    }
    return {"requests": [request], "responses": [response]}


def _persister_with_fake_redis():
    fake = fakeredis.FakeStrictRedis()
    return fake, make_redis_persister(client=fake)


def test_save_then_load_roundtrips_cassette_content():
    _, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_y"

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)
    requests, responses = persister.load_cassette(cassette_id, yamlserializer)

    assert len(requests) == 1
    assert len(responses) == 1
    assert requests[0].method == "POST"
    assert requests[0].uri == "https://api.anthropic.com/v1/messages"
    assert responses[0]["status"]["code"] == 200
    assert responses[0]["body"]["string"] == b'{"id":"msg_1","type":"message"}'


def test_saved_key_has_24h_ttl():
    fake, persister = _persister_with_fake_redis()
    cassette_id = "tests/llm_translation/test_x/test_ttl"

    persister.save_cassette(cassette_id, _sample_cassette_dict(), yamlserializer)

    ttl = fake.ttl(redis_key_for(cassette_id))
    assert CASSETTE_TTL_SECONDS - 5 <= ttl <= CASSETTE_TTL_SECONDS


def test_load_missing_key_raises_cassette_not_found():
    _, persister = _persister_with_fake_redis()
    with pytest.raises(CassetteNotFoundError):
        persister.load_cassette("never/recorded", yamlserializer)


def test_redis_key_normalizes_path_passed_by_pytest_recording():
    # pytest-recording passes paths shaped like
    # ``<test_dir>/cassettes/<module>/<test>.yaml``. The persister stores them
    # under a clean test-identifier key — no extension, no ``cassettes/``
    # directory segment — so ``redis-cli keys`` reads as test IDs.
    raw = "tests/llm_translation/cassettes/test_anthropic/test_streaming.yaml"
    assert (
        redis_key_for(raw)
        == "litellm:vcr:cassette:tests/llm_translation/test_anthropic/test_streaming"
    )


@pytest.mark.parametrize(
    ("status_code", "expect_dropped"),
    [
        (200, False),
        (201, False),
        (204, False),
        (299, False),
        (300, True),
        (400, True),
        (401, True),
        (404, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
    ],
)
def test_only_2xx_responses_are_cached(status_code, expect_dropped):
    response = {
        "status": {"code": status_code, "message": "X"},
        "headers": {},
        "body": {"string": ""},
    }
    result = filter_non_2xx_response(response)
    assert (result is None) == expect_dropped
    if not expect_dropped:
        assert result is response
