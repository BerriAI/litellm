"""Unit tests for litellm.proxy.shunt_endpoints.endpoints."""

import io

import pytest
from fastapi import HTTPException, Request, UploadFile

import litellm.proxy.shunt_endpoints.endpoints as endpoints_mod
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.guardrails.auto_router_shunt import ShuntConfig
from litellm.proxy.guardrails.shunt_capability_token import mint_shunt_capability_token
from litellm.proxy.shunt_endpoints.endpoints import _caller_from_capability_token, _worker_text


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234-test-salt-key")


class TestMissingOrMalformedHeader:
    @pytest.mark.asyncio
    async def test_no_header_is_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_non_bearer_header_is_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization="Basic abc123")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_token_is_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization="Bearer not-a-real-token")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_token_is_rejected(self, monkeypatch):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None, now=1_000_000)
        # 121s after mint, one second past the 120s TTL.
        import litellm.proxy.guardrails.shunt_capability_token as token_mod

        monkeypatch.setattr(token_mod.time, "time", lambda: 1_000_121)
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401


class TestKeyHashGrant:
    @pytest.mark.asyncio
    async def test_resolves_the_key_object_for_the_grants_hash(self, monkeypatch):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None)
        resolved = UserAPIKeyAuth(api_key="deadbeef", team_id="team-1")

        async def _fake_get_key_object(**kwargs):
            assert kwargs["hashed_token"] == "deadbeef"
            return resolved

        import litellm.proxy.auth.auth_checks as auth_checks

        monkeypatch.setattr(auth_checks, "get_key_object", _fake_get_key_object)
        result = await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert result is resolved

    @pytest.mark.asyncio
    async def test_a_lookup_failure_is_rejected_not_propagated(self, monkeypatch):
        token = mint_shunt_capability_token(key_hash="deadbeef", master_key=None)

        async def _raising_get_key_object(**kwargs):
            raise Exception("key not found")

        import litellm.proxy.auth.auth_checks as auth_checks

        monkeypatch.setattr(auth_checks, "get_key_object", _raising_get_key_object)
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401


class TestMasterKeyGrant:
    @pytest.mark.asyncio
    async def test_matching_master_key_resolves_as_proxy_admin(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-the-real-master-key")
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-the-real-master-key")
        result = await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert result.user_role == LitellmUserRoles.PROXY_ADMIN

    # Regression: a resolved master-key caller carried the raw master key as its own api_key,
    # which _worker_text later places in the outbound request's metadata["user_api_key"] --
    # reachable by any raw-metadata logging callback. Normal master-key auth substitutes a
    # stable alias there specifically to keep the real key out of that sink; this must match.
    @pytest.mark.asyncio
    async def test_resolved_caller_never_carries_the_raw_master_key(self, monkeypatch):
        from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS

        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-the-real-master-key")
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-the-real-master-key")
        result = await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert result.api_key == LITELLM_PROXY_MASTER_KEY_ALIAS
        assert result.api_key != "sk-the-real-master-key"

    @pytest.mark.asyncio
    async def test_master_key_mismatch_is_rejected(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-the-current-master-key")
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-a-stale-master-key")
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_configured_master_key_rejects_a_master_key_grant(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
        token = mint_shunt_capability_token(key_hash=None, master_key="sk-anything")
        with pytest.raises(HTTPException) as exc_info:
            await _caller_from_capability_token(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 401


# _worker_text no longer calls llm_router.acompletion directly -- the send function is
# injected instead -- so this only needs to satisfy the type annotation, not do anything.
class _FakeRouter:
    pass


class _FakeProxyLogging:
    """post_call_success_hook is the one real dependency _worker_text calls on this object;
    pre_call/rate-limit/budget enforcement lives inside common_processing_pre_call_logic,
    supplied per test by the injected processor factory."""

    def __init__(self):
        self.post_call_success_hook_calls = []

    async def post_call_success_hook(self, *, data, user_api_key_dict, response):
        self.post_call_success_hook_calls.append((data, user_api_key_dict, response))
        return response

    async def post_call_failure_hook(self, *, user_api_key_dict, original_exception, request_data):
        return None


def _fake_request() -> Request:
    return Request(scope={"type": "http", "headers": [], "method": "POST", "path": "/"})


# Regression: the worker call went straight to llm_router.acompletion, skipping every
# registered rate-limit/budget callback and guardrail (they only run inside
# common_processing_pre_call_logic + route_request, the same pipeline /chat/completions uses).
# A caller already over budget or rate-limited could keep spending through this endpoint.
class TestWorkerTextGoesThroughTheSharedPipeline:
    """_worker_text takes its processor factory and its send function as parameters, so these
    pass doubles in rather than patching methods onto ProxyBaseLLMRequestProcessing itself.
    Only the proxy_server startup globals are monkeypatched, which are module-level state with
    no injection point, not class attributes."""

    def _processor_factory(self, *, pre_call_error: Exception | None = None):
        """A stand-in for ProxyBaseLLMRequestProcessing that subclasses the real thing, so
        _handle_llm_api_exception stays the production implementation. That conversion is
        exactly what the blocked-pre-call test asserts on, so faking it would test nothing."""

        class _StubProcessor(ProxyBaseLLMRequestProcessing):
            async def common_processing_pre_call_logic(self, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]  # test double narrows to the kwargs _worker_text passes
                if pre_call_error is not None:
                    raise pre_call_error
                return self.data, object()

        return _StubProcessor

    def _sender(self, response_text: str):
        """Mirrors route_request's real contract: awaiting it resolves the deployment and
        hands back the provider coroutine *unawaited*, so the caller must await twice. A
        double that returned the ModelResponse directly would pass against a caller that
        forgets the second await and hands a raw coroutine to the rest of the pipeline."""

        async def _send(**kwargs):
            from litellm.types.utils import Choices, Message, ModelResponse

            async def _provider_call():
                return ModelResponse(
                    choices=[Choices(index=0, message=Message(role="assistant", content=response_text))]
                )

            return _provider_call()

        return _send

    def _patch_startup_globals(self, monkeypatch, fake_logging: _FakeProxyLogging):
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_logging)
        monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
        monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", object())

    @pytest.mark.asyncio
    async def test_runs_the_pipeline_and_the_post_call_success_hook(self, monkeypatch):
        fake_logging = _FakeProxyLogging()
        self._patch_startup_globals(monkeypatch, fake_logging)
        text = await _worker_text(
            _fake_request(),
            _FakeRouter(),
            model="claude-haiku-4-5",
            system_prompt="be precise",
            message="what does this do",
            user_api_key_dict=UserAPIKeyAuth(api_key="fakehash1234567890"),
            label="bulk_read",
            make_processor=self._processor_factory(),
            send=self._sender("the worker's answer"),
        )
        assert text == "the worker's answer"
        assert len(fake_logging.post_call_success_hook_calls) == 1

    @pytest.mark.asyncio
    async def test_a_blocked_pre_call_prevents_the_worker_call(self, monkeypatch):
        fake_logging = _FakeProxyLogging()
        self._patch_startup_globals(monkeypatch, fake_logging)
        # _worker_text lets a blocked pre-call raise through
        # ProxyBaseLLMRequestProcessing._handle_llm_api_exception, the same conversion every
        # other LLM route uses, so a raw HTTPException surfaces as the proxy-standard
        # ProxyException rather than passing through unmodified.
        with pytest.raises(ProxyException) as exc_info:
            await _worker_text(
                _fake_request(),
                _FakeRouter(),
                model="claude-haiku-4-5",
                system_prompt="be precise",
                message="what does this do",
                user_api_key_dict=UserAPIKeyAuth(api_key="fakehash1234567890"),
                label="bulk_read",
                make_processor=self._processor_factory(
                    pre_call_error=HTTPException(status_code=429, detail="rate limited")
                ),
                send=self._sender("should never be reached"),
            )
        assert exc_info.value.code == "429"
        assert len(fake_logging.post_call_success_hook_calls) == 0


# Regression: _read_upload_text called upload.read() with no size, loading each whole file into
# memory and retaining every decoded file before building the prompt. An authenticated caller
# with a valid capability token could exhaust a proxy worker's memory, since the global
# request-size middleware is opt-in and premium-gated.
class TestUploadLimits:
    def _upload(self, name: str, content: bytes) -> UploadFile:
        return UploadFile(file=io.BytesIO(content), filename=name)

    @pytest.mark.asyncio
    async def test_a_file_over_the_per_file_limit_is_rejected(self, monkeypatch):
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_PER_FILE", 10)
        with pytest.raises(ProxyException) as exc_info:
            await endpoints_mod._read_upload_texts([self._upload("big.py", b"x" * 50)])
        assert exc_info.value.code == "400"

    @pytest.mark.asyncio
    async def test_a_misconfigured_negative_per_file_limit_still_bounds_the_read(self, monkeypatch):
        """A negative _MAX_UPLOAD_BYTES_PER_FILE (e.g. from a bad env var override) must never
        reach UploadFile.read(): a negative size there means "read the whole file", which would
        silently defeat this limit for every upload rather than enforce it. The constant is
        range-validated at import time via get_env_int_in_range specifically to prevent this,
        but this asserts the read-path behavior directly regardless of how the value got here."""
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_PER_FILE", -5)
        upload = self._upload("big.py", b"this must never be read without a positive bound")
        real_read = upload.read
        read_calls: list[int] = []

        async def _tracking_read(size: int = -1):
            read_calls.append(size)
            return await real_read(size)

        upload.read = _tracking_read  # rebind-ok: test spy on this one instance
        with pytest.raises(ProxyException) as exc_info:
            await endpoints_mod._read_upload_texts([upload])
        assert exc_info.value.code == "400"
        assert all(size > 0 for size in read_calls), (
            f"read() was called with a non-positive size in {read_calls}, "
            "which UploadFile.read() treats as 'read the whole file'"
        )

    @pytest.mark.asyncio
    async def test_files_under_the_limits_are_read_in_full(self, monkeypatch):
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_PER_FILE", 100)
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_TOTAL", 100)
        texts = await endpoints_mod._read_upload_texts(
            [self._upload("a.py", b"hello"), self._upload("b.py", b"world")]
        )
        assert texts == ("hello", "world")

    @pytest.mark.asyncio
    async def test_too_many_files_is_rejected_before_any_read(self, monkeypatch):
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_FILE_COUNT", 2)
        with pytest.raises(ProxyException) as exc_info:
            await endpoints_mod._read_upload_texts(
                [self._upload(f"{i}.py", b"x") for i in range(3)]
            )
        assert exc_info.value.code == "400"

    @pytest.mark.asyncio
    async def test_files_individually_under_but_together_over_the_total_are_rejected(self, monkeypatch):
        """The aggregate budget is what a many-small-files flood would otherwise slip past."""
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_PER_FILE", 100)
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_TOTAL", 12)
        with pytest.raises(ProxyException) as exc_info:
            await endpoints_mod._read_upload_texts(
                [self._upload("a.py", b"x" * 10), self._upload("b.py", b"y" * 10)]
            )
        assert exc_info.value.code == "400"

    @pytest.mark.asyncio
    async def test_an_exhausted_total_budget_rejects_the_next_file(self, monkeypatch):
        """The running total in _read_upload_texts can only ever fall to exactly zero, never
        below it (each read is already bounded by whatever was left when it started), so this
        exercises that real, reachable boundary: two files exactly filling the total budget,
        then a third that must be rejected outright rather than read at all."""
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_PER_FILE", 100)
        monkeypatch.setattr(endpoints_mod, "_MAX_UPLOAD_BYTES_TOTAL", 10)
        third = self._upload("c.py", b"this file must never actually be read")
        real_read = third.read
        read_calls: list[int] = []

        async def _tracking_read(size: int = -1):
            read_calls.append(size)
            return await real_read(size)

        third.read = _tracking_read  # rebind-ok: test spy on this one instance
        with pytest.raises(ProxyException) as exc_info:
            await endpoints_mod._read_upload_texts(
                [self._upload("a.py", b"aaaaaa"), self._upload("b.py", b"bbbb"), third]
            )
        assert exc_info.value.code == "400"
        assert read_calls == [], f"third file's read() was called with sizes {read_calls}, expected no call at all"

    @pytest.mark.asyncio
    async def test_read_upload_text_itself_rejects_a_negative_budget_without_reading(self):
        """_read_upload_texts's own loop can never pass a negative remaining_total_bytes (see
        the test above), but _read_upload_text is called with a caller-supplied budget and
        must reject one directly rather than pass it to UploadFile.read(), which treats a
        negative size as "read the whole file" and would silently defeat this limit."""
        upload = self._upload("c.py", b"must never actually be read")
        real_read = upload.read
        read_calls: list[int] = []

        async def _tracking_read(size: int = -1):
            read_calls.append(size)
            return await real_read(size)

        upload.read = _tracking_read  # rebind-ok: test spy on this one instance
        with pytest.raises(ProxyException) as exc_info:
            await endpoints_mod._read_upload_text(upload, remaining_total_bytes=-5)
        assert exc_info.value.code == "400"
        assert read_calls == [], f"read() was called with sizes {read_calls}, expected no call at all"

    @pytest.mark.asyncio
    async def test_non_utf8_content_is_rejected(self):
        with pytest.raises(ProxyException) as exc_info:
            await endpoints_mod._read_upload_texts([self._upload("bin.dat", b"\xff\xfe\x00binary")])
        assert exc_info.value.code == "400"


class TestWorkerConfig:
    """_worker_config is what stops these endpoints becoming a way to run any model the
    caller names: the worker model comes from the marker's own config, and a router that
    isn't shunt-armed is refused outright."""

    def test_no_router_configured_is_a_500(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
        with pytest.raises(ProxyException) as exc_info:
            endpoints_mod._worker_config("shunt-router", UserAPIKeyAuth(api_key="h"), ())
        assert exc_info.value.code == "500"

    def test_a_router_without_shunt_armed_is_rejected(self, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", object())
        monkeypatch.setattr(endpoints_mod, "shunt_config_for_model", lambda **kwargs: None)
        with pytest.raises(ProxyException) as exc_info:
            endpoints_mod._worker_config("plain-router", UserAPIKeyAuth(api_key="h"), ())
        assert exc_info.value.code == "400"
        assert "not a shunt-armed auto router" in exc_info.value.message

    def test_the_callers_team_and_tags_scope_the_lookup(self, monkeypatch):
        """A marker armed only under a tag must resolve here the same way it did when the
        original request was rewritten, so the lookup has to carry both through."""
        seen: dict[str, object] = {}
        fake_router = object()
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", fake_router)

        def _spy(**kwargs):
            seen.update(kwargs)
            return ShuntConfig(min_lines=350, bulk_read_model="w", code_write_model="w")

        monkeypatch.setattr(endpoints_mod, "shunt_config_for_model", _spy)
        router, config = endpoints_mod._worker_config(
            "shunt-router", UserAPIKeyAuth(api_key="h", team_id="team-9"), ("tag-a",)
        )
        assert router is fake_router
        assert config.bulk_read_model == "w"
        assert seen["team_id"] == "team-9"
        assert seen["request_tags"] == ("tag-a",)
        assert seen["model_alias"] == "shunt-router"


class TestHandlers:
    """The two route handlers, exercised directly. Both are thin, but each owns one thing
    worth pinning: bulk_read pairs every upload with its own filename before the worker sees
    it, and code_write strips the fences a chat model wraps code in."""

    def _upload(self, name: str, content: bytes) -> UploadFile:
        return UploadFile(file=io.BytesIO(content), filename=name)

    def _patch(self, monkeypatch, *, worker_reply: str) -> dict[str, object]:
        captured: dict[str, object] = {}
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", object())
        monkeypatch.setattr(
            endpoints_mod,
            "shunt_config_for_model",
            lambda **kwargs: ShuntConfig(min_lines=350, bulk_read_model="cheap-read", code_write_model="cheap-write"),
        )

        async def _fake_worker_text(request, llm_router, **kwargs):
            captured.update(kwargs)
            return worker_reply

        monkeypatch.setattr(endpoints_mod, "_worker_text", _fake_worker_text)
        return captured

    @pytest.mark.asyncio
    async def test_bulk_read_labels_each_file_and_uses_the_configured_read_model(self, monkeypatch):
        captured = self._patch(monkeypatch, worker_reply="the summary")
        result = await endpoints_mod.bulk_read(
            _fake_request(),
            router_name="shunt-router",
            question="what do these do",
            paths=[self._upload("a.py", b"AAA"), self._upload("b.py", b"BBB")],
            user_api_key_dict=UserAPIKeyAuth(api_key="h"),
        )
        assert result == "the summary"
        assert captured["model"] == "cheap-read"
        message = captured["message"]
        assert isinstance(message, str)
        # Each file's own content must travel under its own name: swapping or dropping a
        # filename here would silently attribute one file's code to another in the answer.
        assert "a.py" in message and "AAA" in message
        assert "b.py" in message and "BBB" in message
        assert message.index("a.py") < message.index("b.py")
        assert "what do these do" in message

    @pytest.mark.asyncio
    async def test_code_write_strips_fences_and_uses_the_configured_write_model(self, monkeypatch):
        captured = self._patch(monkeypatch, worker_reply="```python\nprint('hi')\n```")
        result = await endpoints_mod.code_write(
            _fake_request(),
            router_name="shunt-router",
            spec="write a greeter",
            reference=self._upload("ref.py", b"REFERENCE"),
            user_api_key_dict=UserAPIKeyAuth(api_key="h"),
        )
        # The client redirects this straight into a file, so a stray ``` line would end up
        # in the written source.
        assert result == "print('hi')"
        assert captured["model"] == "cheap-write"
        message = captured["message"]
        assert isinstance(message, str)
        assert "write a greeter" in message and "REFERENCE" in message
