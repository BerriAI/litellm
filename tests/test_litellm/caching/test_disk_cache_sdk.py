"""SDK-level disk-cache behavior, deterministic via mock_response (no network).

Runs unchanged against both the diskcache baseline and the sqlite candidate.
Characterizes end-to-end caching: hit/miss per call type, cache-control matrix,
cache-key factors, TTL, namespace, and mode. Cache reset between tests is
handled by the autouse isolation fixture in tests/test_litellm/conftest.py.
"""

import asyncio

import pytest

import litellm
from litellm.caching import Cache

M = [{"role": "user", "content": "hello world"}]
CHAT = "gpt-3.5-turbo"
EMB = "text-embedding-ada-002"
TEXT = "gpt-3.5-turbo-instruct"


def _disk(tmp_path, **kw):
    litellm.cache = Cache(type="disk", disk_cache_dir=str(tmp_path / "dc"), **kw)


def _content(resp):
    return resp.choices[0].message.content


def _hit(resp):
    return resp._hidden_params.get("cache_hit") is True


# --------------------------------------------------------------------------- #
# completion: hit / miss, sync + async
# --------------------------------------------------------------------------- #


def test_completion_cache_hit_sync(tmp_path):
    _disk(tmp_path)
    r1 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    r2 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="B")
    assert _content(r1) == "A"
    assert _content(r2) == "A"  # cache hit, B ignored
    assert _hit(r2)
    assert r1.id == r2.id
    assert r1.created == r2.created


@pytest.mark.asyncio
async def test_completion_cache_hit_async(tmp_path):
    _disk(tmp_path)
    r1 = await litellm.acompletion(
        model=CHAT, messages=M, caching=True, mock_response="A"
    )
    await asyncio.sleep(0.1)
    r2 = await litellm.acompletion(
        model=CHAT, messages=M, caching=True, mock_response="B"
    )
    assert _content(r2) == "A"
    assert _hit(r2)


@pytest.mark.parametrize(
    "override",
    [
        {"temperature": 0.9},
        {"max_tokens": 7},
        {"top_p": 0.5},
        {"model": "gpt-4o-mini"},
        {"messages": [{"role": "user", "content": "different"}]},
    ],
)
def test_completion_cache_miss_on_differing_param(tmp_path, override):
    _disk(tmp_path)
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    kwargs = {"model": CHAT, "messages": M, "caching": True, "mock_response": "C"}
    kwargs.update(override)
    r = litellm.completion(**kwargs)
    assert _content(r) == "C"  # different key -> miss
    assert not _hit(r)


# --------------------------------------------------------------------------- #
# embedding: hit / miss, sync + async
# --------------------------------------------------------------------------- #


def test_embedding_cache_hit_sync(tmp_path):
    _disk(tmp_path)
    v1 = [[0.1, 0.2, 0.3]]
    v2 = [[0.9, 0.9, 0.9]]
    litellm.embedding(model=EMB, input=["hi"], caching=True, mock_response=v1)
    r2 = litellm.embedding(model=EMB, input=["hi"], caching=True, mock_response=v2)
    assert _hit(r2)
    assert r2.data[0]["embedding"] == v1  # cached first value, not v2


def test_embedding_cache_miss_on_different_input(tmp_path):
    _disk(tmp_path)
    litellm.embedding(
        model=EMB, input=["hi"], caching=True, mock_response=[[0.1, 0.2, 0.3]]
    )
    r = litellm.embedding(
        model=EMB, input=["bye"], caching=True, mock_response=[[0.9, 0.9, 0.9]]
    )
    assert not _hit(r)
    assert r.data[0]["embedding"] == [[0.9, 0.9, 0.9]]


@pytest.mark.asyncio
async def test_embedding_cache_hit_async(tmp_path):
    _disk(tmp_path)
    await litellm.aembedding(
        model=EMB, input=["hi"], caching=True, mock_response=[[0.1, 0.2, 0.3]]
    )
    await asyncio.sleep(0.1)
    r2 = await litellm.aembedding(
        model=EMB, input=["hi"], caching=True, mock_response=[[0.9, 0.9, 0.9]]
    )
    assert _hit(r2)
    assert r2.data[0]["embedding"] == [[0.1, 0.2, 0.3]]


# --------------------------------------------------------------------------- #
# text_completion: hit / miss, sync + async
# --------------------------------------------------------------------------- #


def test_text_completion_cache_hit_sync(tmp_path):
    _disk(tmp_path)
    litellm.text_completion(model=TEXT, prompt="hi", caching=True, mock_response="A")
    r2 = litellm.text_completion(
        model=TEXT, prompt="hi", caching=True, mock_response="B"
    )
    assert r2.choices[0].text == "A"


@pytest.mark.asyncio
async def test_text_completion_cache_hit_async(tmp_path):
    _disk(tmp_path)
    await litellm.atext_completion(
        model=TEXT, prompt="hi", caching=True, mock_response="A"
    )
    await asyncio.sleep(0.1)
    r2 = await litellm.atext_completion(
        model=TEXT, prompt="hi", caching=True, mock_response="B"
    )
    assert r2.choices[0].text == "A"


# --------------------------------------------------------------------------- #
# cache-control matrix
# --------------------------------------------------------------------------- #


def test_no_cache_skips_read(tmp_path):
    _disk(tmp_path)
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    r2 = litellm.completion(
        model=CHAT,
        messages=M,
        caching=True,
        mock_response="B",
        cache={"no-cache": True},
    )
    assert _content(r2) == "B"  # read skipped -> fresh mock
    assert not _hit(r2)


def test_no_store_skips_write(tmp_path):
    _disk(tmp_path)
    litellm.completion(
        model=CHAT,
        messages=M,
        caching=True,
        mock_response="A",
        cache={"no-store": True},
    )
    r2 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="B")
    assert _content(r2) == "B"  # nothing stored -> miss


def test_ttl_zero_via_cache_control_expires_immediately(tmp_path):
    _disk(tmp_path)
    litellm.completion(
        model=CHAT, messages=M, caching=True, mock_response="A", cache={"ttl": 0}
    )
    r2 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="B")
    assert _content(r2) == "B"  # stored entry already expired -> miss


def test_smaxage_fresh_within_window_hits(tmp_path):
    _disk(tmp_path)
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    r2 = litellm.completion(
        model=CHAT,
        messages=M,
        caching=True,
        mock_response="B",
        cache={"s-maxage": 1000},
    )
    assert _content(r2) == "A"
    assert _hit(r2)


def test_smaxage_too_old_misses(tmp_path):
    _disk(tmp_path)
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    # negative max-age: any positive age exceeds it -> rejected as too old
    r2 = litellm.completion(
        model=CHAT, messages=M, caching=True, mock_response="B", cache={"s-maxage": -1}
    )
    assert _content(r2) == "B"


def test_namespace_isolates_keys(tmp_path):
    _disk(tmp_path, namespace="ns1")
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    litellm.cache = Cache(
        type="disk", disk_cache_dir=str(tmp_path / "dc"), namespace="ns2"
    )
    r2 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="B")
    assert _content(r2) == "B"  # different namespace -> different key -> miss


# --------------------------------------------------------------------------- #
# mode + global ttl + persistence
# --------------------------------------------------------------------------- #


def test_mode_default_off_does_not_cache_without_optin(tmp_path):
    _disk(tmp_path, mode="default_off")
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    r2 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="B")
    assert _content(r2) == "B"  # default_off + no use-cache -> never cached


def test_mode_default_off_caches_with_use_cache(tmp_path):
    _disk(tmp_path, mode="default_off")
    litellm.completion(
        model=CHAT, messages=M, mock_response="A", cache={"use-cache": True}
    )
    r2 = litellm.completion(
        model=CHAT, messages=M, mock_response="B", cache={"use-cache": True}
    )
    assert _content(r2) == "A"  # opt-in -> cached


def test_global_ttl_zero_expires_immediately(tmp_path):
    _disk(tmp_path, ttl=0)
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    r2 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="B")
    assert _content(r2) == "B"  # global ttl=0 -> entries expire immediately -> miss


def test_persistence_across_new_cache_instance(tmp_path):
    _disk(tmp_path)
    litellm.completion(model=CHAT, messages=M, caching=True, mock_response="A")
    # brand-new Cache object over the same on-disk dir
    litellm.cache = Cache(type="disk", disk_cache_dir=str(tmp_path / "dc"))
    r2 = litellm.completion(model=CHAT, messages=M, caching=True, mock_response="B")
    assert _content(r2) == "A"  # survived re-instantiation
    assert _hit(r2)
