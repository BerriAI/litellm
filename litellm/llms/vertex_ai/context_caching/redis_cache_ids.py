"""Opt-in Vertex cache-name registry, independent of LiteLLM response caching.

Redis stores hashes, resource names and deadlines, never prompts or credentials.
No Google LIST/GET is used. All coordination operations have bounded timeouts.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody, RequestBody

if TYPE_CHECKING:
    from redis import Redis

    from .vertex_ai_context_caching import ContextCachingEndpoints
from datetime import datetime
from functools import lru_cache

import httpx


def redis_error_type() -> type[Exception]:
    try:
        from redis.exceptions import RedisError

        return RedisError
    except ImportError:
        return ImportError


RedisError: Final = redis_error_type()
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


Params = dict[str, JsonValue]
Provider = Literal["vertex_ai", "vertex_ai_beta", "gemini"]
CacheResult = tuple[list[AllMessageValues], Params, str | None]


class CacheArguments(TypedDict):
    messages: ReadOnly[list[AllMessageValues]]
    optional_params: ReadOnly[Params]
    api_key: ReadOnly[str]
    api_base: ReadOnly[str | None]
    model: ReadOnly[str]
    client: ReadOnly[HTTPHandler | AsyncHTTPHandler | None]
    timeout: ReadOnly[float | httpx.Timeout | None]
    logging_obj: ReadOnly[Logging]
    custom_llm_provider: ReadOnly[Provider]
    vertex_project: ReadOnly[str | None]
    vertex_location: ReadOnly[str | None]
    vertex_auth_header: ReadOnly[str | None]
    extra_headers: ReadOnly[dict[str, str] | None]
    cache_id_auth_scope: ReadOnly[str | None]


class RequestTransform(Protocol):
    def __call__(
        self,
        *,
        messages: list[AllMessageValues],
        model: str,
        optional_params: Params,
        custom_llm_provider: Provider,
        litellm_params: Params,
        cached_content: str | None,
    ) -> RequestBody: ...


@dataclass(frozen=True, slots=True)
class PreparedCache:
    registry: Registry
    url: str
    headers: dict[str, str]
    body: CachedContentRequestBody
    messages: list[AllMessageValues]
    params: Params


ENV = "LITELLM_VERTEX_CACHE_ID_REDIS_URL"
CACHE_NAME = re.compile(r"projects/[^/]+/locations/[^/]+/cachedContents/[^/]+")
# A bounded wait avoids turning cache contention into generation latency.
WAIT_SECONDS = 2.0
LEASE_MS = 60000
SAFETY_SECONDS = 60.0
RELEASE = "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) end return 0"
PUBLISH = """if redis.call('get',KEYS[1]) == ARGV[1] then
redis.call('psetex',KEYS[2],ARGV[3],ARGV[2]); redis.call('del',KEYS[1]); return 1 end return 0"""
INVALIDATE = """local v=redis.call('get',KEYS[1]); if v then
local ok,r=pcall(cjson.decode,v); if ok and r.name == ARGV[1] then return redis.call('del',KEYS[1]) end end return 0"""


def valid_name(name: object) -> bool:
    return isinstance(name, str) and CACHE_NAME.fullmatch(name) is not None


def enabled(provider: str) -> bool:
    return provider == "vertex_ai" and bool(os.environ.get(ENV))


@lru_cache(maxsize=4)
def connection(url: str) -> Redis:
    from redis import Redis

    return Redis.from_url(
        url,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
        retry_on_timeout=False,
        decode_responses=True,
    )


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def credential_scope(credentials: object) -> str | None:
    """Fingerprint explicit credentials, independent of short-lived access tokens.

    Unknown/ADC credentials use the token-scoped fallback in prepare(). Raw
    credentials never enter Redis. File contents are read only for a configured
    credential path already accepted by the Vertex authentication code.
    """
    if not enabled("vertex_ai"):
        return None
    try:
        if isinstance(credentials, str):
            if os.path.isfile(credentials):
                with open(credentials) as stream:
                    credentials = json.load(stream)
            else:
                credentials = json.loads(credentials)
        if isinstance(credentials, dict) and credentials:
            return digest(credentials)
    except (OSError, ValueError, TypeError):
        pass
    return None


def transient_creation_error(error: Exception) -> bool:
    import litellm

    if isinstance(error, (httpx.TransportError, litellm.Timeout)):
        return True
    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (
        429,
        500,
        502,
        503,
        504,
    )


class Registry:
    def __init__(
        self,
        client: Redis,
        scope: object,
        body: object,
        *,
        safety: float = SAFETY_SECONDS,
        wait: float = WAIT_SECONDS,
        lease_ms: int = LEASE_MS,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.client = client
        self.key = "litellm:vertex-cache-id:v1:{" + digest([scope, body]) + "}"
        self.lock = self.key + ":lock"
        self.cooldown = "litellm:vertex-cache-id:cooldown:" + digest(scope)
        self.safety, self.wait, self.lease_ms = safety, wait, lease_ms
        self.clock, self.monotonic = clock, monotonic
        self.sleep, self.async_sleep = sleep, async_sleep

    def read(self) -> str | None:
        raw = self.client.get(self.key)
        if not raw:
            return None
        try:
            record = JSON_OBJECT.validate_json(raw)
            if not isinstance(record.get("name"), str) or not valid_name(record["name"]):
                return None
            if not math.isfinite(float(record["expires"])) or float(record["expires"]) - self.safety <= self.clock():
                return None
            return record["name"]
        except (ValueError, TypeError, KeyError):
            return None

    def publish(self, token: str, result: Mapping[str, object]) -> str | None:
        try:
            name = result["name"]
            if not isinstance(result["expireTime"], str):
                return None
            expires = datetime.fromisoformat(result["expireTime"].replace("Z", "+00:00"))
            if expires.tzinfo is None or not isinstance(name, str) or not valid_name(name):
                return None
            deadline = expires.timestamp()
            ttl = int((deadline - self.safety - self.clock()) * 1000)
            if ttl <= 0:
                return None
            record = json.dumps({"name": name, "expires": deadline})
            if self.client.eval(PUBLISH, 2, self.lock, self.key, token, record, ttl):
                return name
        except (ValueError, TypeError, KeyError):
            pass
        return None

    def release(self, token: str) -> None:
        try:
            self.client.eval(RELEASE, 1, self.lock, token)
        except RedisError:
            pass

    def invalidate(self, name: str) -> None:
        try:
            self.client.eval(INVALIDATE, 1, self.key, str(name))
        except RedisError:
            pass

    def acquire(self, token: str) -> bool:
        return bool(self.client.set(self.lock, token, nx=True, px=self.lease_ms))

    def cool_down(self) -> None:
        # Shared per auth/project/endpoint scope. Do not retry cache creation on 429.
        try:
            self.client.set(self.cooldown, "1", ex=30)
        except RedisError:
            pass

    def lookup(self) -> tuple[bool, str | None]:
        """Return (finished, name); a cooldown finishes without a cache ID."""
        name = self.read()
        return bool(name or self.client.exists(self.cooldown)), name

    def resolve(self, create: Callable[[], Mapping[str, object]]) -> str | None:
        token = uuid.uuid4().hex
        end = self.monotonic() + self.wait
        try:
            while True:
                finished, name = self.lookup()
                if finished:
                    return name
                if self.acquire(token):
                    try:
                        # Prior owner may have published before acquisition.
                        finished, name = self.lookup()
                        if finished:
                            return name
                        return self.publish(token, create())
                    except Exception as error:
                        if not transient_creation_error(error):
                            raise
                        self.cool_down()
                        return None
                    finally:
                        self.release(token)
                if self.monotonic() >= end:
                    return None
                self.sleep(0.02)
        except RedisError:
            return None

    async def async_resolve(self, create: Callable[[], Awaitable[Mapping[str, object]]]) -> str | None:
        token = uuid.uuid4().hex
        end = self.monotonic() + self.wait
        try:
            while True:
                finished, name = await asyncio.to_thread(self.lookup)
                if finished:
                    return name
                if await asyncio.to_thread(self.acquire, token):
                    try:
                        finished, name = await asyncio.to_thread(self.lookup)
                        if finished:
                            return name
                        result = await create()
                        return await asyncio.to_thread(self.publish, token, result)
                    except Exception as error:
                        if not transient_creation_error(error):
                            raise
                        await asyncio.to_thread(self.cool_down)
                        return None
                    finally:
                        await asyncio.shield(asyncio.to_thread(self.release, token))
                if self.monotonic() >= end:
                    return None
                await self.async_sleep(0.02)
        except RedisError:
            return None


class CacheReference(str):
    """String on the wire; local metadata supports one stale-cache recovery."""

    registry: Registry
    messages: list[AllMessageValues]
    params: Params
    rebuild: Callable[[], RequestBody]

    def __new__(cls, name: str, registry: Registry, messages: list[AllMessageValues], params: Params) -> CacheReference:
        obj = super().__new__(cls, name)
        obj.registry = registry
        obj.messages = copy.deepcopy(messages)
        obj.params = copy.deepcopy(params)
        return obj

    def __deepcopy__(self, memo: dict[int, object]) -> CacheReference:
        return CacheReference(str(self), self.registry, self.messages, self.params)

    def restore(self, body: RequestBody) -> None:
        replacement = self.rebuild()
        for key in tuple(body):
            del body[key]
        body.update(replacement)


def prepare(endpoint: ContextCachingEndpoints, kw: CacheArguments) -> PreparedCache | None:
    from .transformation import (
        cached_messages_end_on_supported_turn,
        separate_cached_messages,
        transform_openai_messages_to_gemini_context_caching,
    )
    from .vertex_ai_context_caching import is_prompt_caching_valid_prompt

    original_messages = kw["messages"]
    original_params = kw["optional_params"]
    cached, uncached = separate_cached_messages(copy.deepcopy(original_messages))
    if (
        not cached
        or not cached_messages_end_on_supported_turn(cached)
        or not is_prompt_caching_valid_prompt(
            model=kw["model"],
            messages=cached,
            tools=original_params.get("tools"),
            custom_llm_provider=kw["custom_llm_provider"],
        )
    ):
        return None
    params = copy.deepcopy(original_params)
    tools, tool_choice = params.pop("tools", None), params.pop("tool_choice", None)
    token, url = endpoint._get_token_and_url_context_caching(
        gemini_api_key=kw["api_key"],
        custom_llm_provider=kw["custom_llm_provider"],
        api_base=kw["api_base"],
        vertex_project=kw["vertex_project"],
        vertex_location=kw["vertex_location"],
        vertex_auth_header=kw["vertex_auth_header"],
        model=kw["model"],
    )
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = token if token.startswith("Bearer ") else "Bearer " + token
    headers.update(kw.get("extra_headers") or {})
    scoped_headers = {key.lower(): value for key, value in headers.items()}
    extra = {key.lower(): value for key, value in (kw.get("extra_headers") or {}).items()}
    stable_auth = kw.get("cache_id_auth_scope")
    if stable_auth and "authorization" not in extra:
        scoped_headers.pop("authorization", None)
    else:
        stable_auth = None
    scope = [
        kw["custom_llm_provider"],
        kw["vertex_project"],
        kw["vertex_location"],
        url,
        stable_auth,
        digest(scoped_headers),
    ]
    body = transform_openai_messages_to_gemini_context_caching(
        model=kw["model"],
        messages=cached,
        cache_key="",
        custom_llm_provider=kw["custom_llm_provider"],
        vertex_project=kw["vertex_project"],
        vertex_location=kw["vertex_location"],
    )
    if "ttl" in body and float(body["ttl"][:-1]) <= SAFETY_SECONDS:
        return None
    body["tools"] = tools
    if tool_choice is not None:
        body["toolConfig"] = tool_choice
    registry = Registry(connection(os.environ[ENV]), scope, body)
    body["displayName"] = registry.key.split("{")[1].rstrip("}")
    return PreparedCache(registry, url, headers, body, uncached, params)


def check_and_create(endpoint: ContextCachingEndpoints, kw: CacheArguments) -> CacheResult:
    import litellm
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    original = (kw["messages"], kw["optional_params"], None)
    try:
        prepared = prepare(endpoint, kw)
        if prepared is None:
            return original
        registry, url, headers, body, messages, params = (
            prepared.registry,
            prepared.url,
            prepared.headers,
            prepared.body,
            prepared.messages,
            prepared.params,
        )

        def create() -> Mapping[str, object]:
            owned = not isinstance(kw["client"], HTTPHandler)
            supplied = kw["client"]
            client = supplied if isinstance(supplied, HTTPHandler) else HTTPHandler(timeout=20)
            try:
                response = client.post(url=url, headers=headers, data=json.dumps(body), timeout=20)
                response.raise_for_status()
                return record_creation(kw["logging_obj"], response)
            finally:
                if owned:
                    client.close()

        name = registry.resolve(create)
        if name:
            return (
                messages,
                params,
                CacheReference(name, registry, kw["messages"], kw["optional_params"]),
            )
    except (RedisError, httpx.TransportError, litellm.Timeout):
        pass
    return original


async def async_check_and_create(endpoint: ContextCachingEndpoints, kw: CacheArguments) -> CacheResult:
    import litellm
    from litellm.llms.custom_httpx.http_handler import (
        AsyncHTTPHandler,
        get_async_httpx_client,
    )

    original = (kw["messages"], kw["optional_params"], None)
    try:
        prepared = prepare(endpoint, kw)
        if prepared is None:
            return original
        registry, url, headers, body, messages, params = (
            prepared.registry,
            prepared.url,
            prepared.headers,
            prepared.body,
            prepared.messages,
            prepared.params,
        )

        async def create() -> Mapping[str, object]:
            # This helper returns a shared client; do not close it here.
            client = (
                kw["client"]
                if isinstance(kw["client"], AsyncHTTPHandler)
                else get_async_httpx_client(params={"timeout": 20}, llm_provider=litellm.LlmProviders.VERTEX_AI)
            )
            response = await client.post(url=url, headers=headers, data=json.dumps(body), timeout=20)
            response.raise_for_status()
            return record_creation(kw["logging_obj"], response)

        name = await registry.async_resolve(create)
        if name:
            return (
                messages,
                params,
                CacheReference(name, registry, kw["messages"], kw["optional_params"]),
            )
    except (RedisError, httpx.TransportError, litellm.Timeout):
        pass
    return original


def attach_recovery(
    reference: str | None, transform: RequestTransform, model: str, provider: Provider, litellm_params: Params
) -> None:
    if isinstance(reference, CacheReference):
        saved_params = copy.deepcopy(litellm_params)
        messages, params = reference.messages, reference.params
        reference.rebuild = lambda: transform(
            messages=copy.deepcopy(messages),
            optional_params=copy.deepcopy(params),
            model=model,
            custom_llm_provider=provider,
            litellm_params=copy.deepcopy(saved_params),
            cached_content=None,
        )


def stale_reference(body: RequestBody | None, response: httpx.Response) -> CacheReference | None:
    reference = body.get("cachedContent") if isinstance(body, dict) else None
    if (
        not isinstance(reference, CacheReference)
        or not hasattr(reference, "rebuild")
        or response.status_code not in (400, 404)
    ):
        return None
    try:
        message = response.json().get("error", {}).get("message", "").lower()
    except (ValueError, AttributeError):
        return None
    cache_error = any(word in message for word in ("cached content", "cachedcontent", "cached_content"))
    missing = any(word in message for word in ("expired", "not found", "does not exist"))
    return reference if cache_error and missing else None


def post_with_recovery(
    post: Callable[[RequestBody | None], httpx.Response], body: RequestBody | None
) -> httpx.Response:
    """Retry exactly once, uncached, only for a recognized stale managed cache."""
    if not isinstance((body or {}).get("cachedContent"), CacheReference):
        return post(body)
    try:
        response = post(body)
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as err:
        err.response.read()
        reference = stale_reference(body, err.response)
        if reference is None:
            raise
        err.response.close()
        reference.registry.invalidate(reference)
        reference.restore(body)
        response = post(body)
        response.raise_for_status()
        return response


async def async_post_with_recovery(
    post: Callable[[RequestBody | None], Awaitable[httpx.Response]], body: RequestBody | None
) -> httpx.Response:
    if not isinstance((body or {}).get("cachedContent"), CacheReference):
        return await post(body)
    try:
        response = await post(body)
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as err:
        await err.response.aread()
        reference = stale_reference(body, err.response)
        if reference is None:
            raise
        await err.response.aclose()
        await asyncio.to_thread(reference.registry.invalidate, reference)
        await asyncio.to_thread(reference.restore, body)
        response = await post(body)
        response.raise_for_status()
        return response


def record_creation(logging_obj: Logging, response: httpx.Response) -> Mapping[str, object]:
    from .vertex_ai_context_caching import _CreatedCachedContentRoot, _record_cache_creation_tokens

    raw = JSON_OBJECT.validate_json(response.content)
    created = _CreatedCachedContentRoot.model_validate(raw).root
    _record_cache_creation_tokens(logging_obj, created)
    return raw
