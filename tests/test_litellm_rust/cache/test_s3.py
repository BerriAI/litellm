import json
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Final, cast
from unittest.mock import Mock

import boto3
import botocore.config
import pytest

from litellm.caching.caching import Cache
from litellm.caching.s3_cache import S3Cache
from litellm.types.caching import LiteLLMCacheType
from tests.test_litellm_rust.support.cache import CacheTestHandle, CacheTestResolver, request
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.s3_stub import S3Stub

pytestmark: Final = pytest.mark.requires_rust_extension


def python_s3(url: str) -> S3Cache:
    return S3Cache(
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
    )


async def test_s3_reads_python_entries_and_writes_with_python_metadata(s3_stub: S3Stub) -> None:
    python_cache: Final = python_s3(s3_stub.url)
    response: Final = {"choices": [{"text": "cached"}], "usage": {"total_tokens": 3}}
    python_cache.set_cache("sync:key", {"timestamp": time.time(), "response": response}, ttl=90)
    python_cache.set_cache("plain", {"timestamp": time.time(), "response": response})
    s3_stub.put_object("team/malformed", b"not a cache entry")
    s3_stub.put_object(
        "team/expired",
        json.dumps({"timestamp": time.time(), "response": response}).encode(),
        {"expires": "Thu, 01 Jan 1970 00:00:00 GMT"},
    )
    binding: Final = CacheTestResolver(
        SimpleNamespace(
            cache=CacheTestHandle.s3(
                "cache-bucket",
                region="us-east-1",
                endpoint_url=s3_stub.url,
                key_prefix="team/",
                access_key_id="key",
                secret_access_key="secret",
            )
        )
    ).resolve()

    assert binding.lookup(request("sync:key")) == response
    assert await binding.async_lookup(request("plain")) == response
    assert binding.lookup(request("malformed")) is None
    assert binding.lookup(request("expired")) is None
    assert binding.lookup(request("absent")) is None

    binding.store({**request("native:key"), "ttl_seconds": 90.0}, response)
    await binding.async_store(request("no_ttl"), response)
    stored: Final = s3_stub.objects["team/native/key"]
    assert stored.headers["content-type"] == "application/json"
    assert stored.headers["content-language"] == "en"
    assert stored.headers["content-disposition"] == 'inline; filename="team/native/key.json"'
    assert stored.headers["cache-control"] == "immutable, max-age=90, s-maxage=90"
    expires: Final = cast(datetime, s3_stub.expires("team/native/key"))
    remaining: Final = (expires - datetime.now(expires.tzinfo)).total_seconds()
    assert 60 < remaining <= 91
    no_ttl: Final = s3_stub.objects["team/no_ttl"]
    assert no_ttl.headers["cache-control"] == "immutable, max-age=31536000, s-maxage=31536000"
    assert "expires" not in no_ttl.headers
    assert python_cache.get_cache("native:key")["response"] == response

    partial: Final = await binding.async_lookup_batch([request("native:key"), request("absent"), request("malformed")])
    assert partial == {"values": [response, None, None], "missing_indices": [1, 2]}


def test_s3_facade_binds_only_exact_configuration_and_falls_back_on_mutation(s3_stub: S3Stub) -> None:
    facade: Final = Cache(
        type=LiteLLMCacheType.S3,
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=s3_stub.url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
    )
    handle: Final = CacheTestHandle.s3(
        "cache-bucket",
        region="us-east-1",
        endpoint_url=s3_stub.url,
        key_prefix="team/",
        access_key_id="key",
        secret_access_key="secret",
    )
    with pytest.raises(TypeError, match="buckets must match"):
        CacheTestHandle.s3("other", region="us-east-1", endpoint_url=s3_stub.url)._bind_facade(facade)
    with pytest.raises(TypeError, match="key prefixes must match"):
        CacheTestHandle.s3(
            "cache-bucket", region="us-east-1", endpoint_url=s3_stub.url, key_prefix="other/"
        )._bind_facade(facade)
    handle._bind_facade(facade)
    resolver: Final = CacheTestResolver(SimpleNamespace(cache=facade))
    binding: Final = resolver.resolve()
    assert binding.kind == "native"

    handler: Final = Mock()
    facade.cache.s3_client.meta.events.register("before-call.s3.*", handler)
    binding.store(request("native"), {"answer": 1})
    assert binding.lookup(request("native")) == {"answer": 1}
    assert handler.call_count == 0
    assert "team/native" in s3_stub.objects

    with rebound(facade.cache, "bucket_name", "other"):
        assert resolver.resolve().kind == "python_callback"
    other_client: Final = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=s3_stub.url,
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )
    with rebound(facade.cache, "s3_client", other_client):
        assert resolver.resolve().kind == "python_callback"

    class CustomS3Cache(S3Cache):
        pass

    subclassed: Final = Cache(
        type=LiteLLMCacheType.S3,
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=s3_stub.url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
    )
    subclassed.cache = CustomS3Cache(
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=s3_stub.url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
    )
    with pytest.raises(TypeError):
        handle._bind_facade(subclassed)
    assert CacheTestResolver(SimpleNamespace(cache=subclassed)).resolve().kind == "python_callback"


def test_s3_facade_rejects_configurations_that_require_python(s3_stub: S3Stub) -> None:
    handle: Final = CacheTestHandle.s3(
        "cache-bucket",
        region="us-east-1",
        endpoint_url=s3_stub.url,
        key_prefix="team/",
        access_key_id="key",
        secret_access_key="secret",
    )
    unverified: Final = Cache(
        type=LiteLLMCacheType.S3,
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url="https://s3.example.test",
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
        s3_verify=False,
    )
    with pytest.raises(TypeError, match="requires Python"):
        handle._bind_facade(unverified)
    proxied: Final = Cache(
        type=LiteLLMCacheType.S3,
        s3_bucket_name="cache-bucket",
        s3_region_name="us-east-1",
        s3_endpoint_url=s3_stub.url,
        s3_aws_access_key_id="key",
        s3_aws_secret_access_key="secret",
        s3_path="team",
        s3_config=botocore.config.Config(proxies={"https": "http://proxy.test"}),
    )
    with pytest.raises(TypeError, match="requires Python"):
        handle._bind_facade(proxied)
