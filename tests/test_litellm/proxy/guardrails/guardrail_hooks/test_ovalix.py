"""
Unit tests for Ovalix guardrail: config resolution and apply_guardrail behavior
with mocked Tracker service responses (allow, anonymize, block).
"""

import base64
import gzip
import json as json_lib
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix import (
    CheckpointTarget,
    OvalixGuardrail,
    OvalixGuardrailBlockedException,
    OvalixGuardrailMissingSecrets,
    ResolvedRouting,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs


@pytest.fixture(autouse=True)
def _clear_ovalix_env(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("OVALIX_"):
            monkeypatch.delenv(key, raising=False)


# Example Tracker responses (as returned by the checkpoint API)
TRACKER_RESPONSE_ALLOW = {
    "action_type": "allow",
    "data_type": "TEXT",
    "original_data": {"content": "how are you?"},
    "modified_data": {"content": "how are you?"},
    "alerts": [],
}

TRACKER_RESPONSE_ANONYMIZE = {
    "action_type": "anonymize",
    "data_type": "TEXT",
    "original_data": {"content": "Hello, my name is David."},
    "modified_data": {"content": "Hello, my name is {Name}. How are you?"},
    "alerts": [
        {
            "title": "Sensitive Data Alert",
            "subtitle": "We've identified that you were trying to share sensitive information",
            "alerts": ["Name:\tDavid\nRedacted to:\t{Name}"],
        }
    ],
}

TRACKER_RESPONSE_BLOCK = {
    "action_type": "block",
    "data_type": "TEXT",
    "original_data": {"content": "I am 15 YO"},
    "modified_data": {"content": "This message was blocked by Ovalix"},
    "alerts": [
        {
            "title": "Sensitive Data Alert",
            "subtitle": "We've identified that you were trying to share sensitive information",
            "alerts": ["Age:\t15\nBlocked"],
        }
    ],
}


def _ovalix_env():
    return {
        "OVALIX_TRACKER_API_BASE": "https://tracker.test",
        "OVALIX_TRACKER_API_KEY": "key",
        "OVALIX_APPLICATION_ID": "app-1",
        "OVALIX_PRE_CHECKPOINT_ID": "pre-1",
        "OVALIX_POST_CHECKPOINT_ID": "post-1",
    }


def _guardrail_kwargs():
    return {
        "guardrail_name": "ovalix-test",
        "event_hook": "pre_call",
        "default_on": True,
    }


def test_discovery_mode_initializes_without_application_id():
    guardrail = OvalixGuardrail(
        tracker_api_base="https://tracker.test",
        tracker_api_key="key",
        guardrail_name="ovalix-test",
        event_hook="pre_call",
        default_on=True,
    )
    assert guardrail._application_id is None
    assert guardrail._enable_routing_cache is True


def test_routing_cache_defaults_true_and_can_disable():
    on = OvalixGuardrail(
        tracker_api_base="https://t", tracker_api_key="k", guardrail_name="o", event_hook="pre_call", default_on=True
    )
    assert on._enable_routing_cache is True
    off = OvalixGuardrail(
        tracker_api_base="https://t",
        tracker_api_key="k",
        enable_routing_cache=False,
        guardrail_name="o",
        event_hook="pre_call",
        default_on=True,
    )
    assert off._enable_routing_cache is False


def test_new_config_fields_from_params():
    guardrail = OvalixGuardrail(
        tracker_api_base="https://tracker.test",
        tracker_api_key="key",
        application_id="app-1",
        pre_checkpoint_id="pre-1",
        file_checkpoint_id="file-1",
        enable_routing_cache=True,
        guardrail_name="ovalix-test",
        event_hook="pre_call",
        default_on=True,
    )
    assert guardrail._file_checkpoint_id == "file-1"
    assert guardrail._enable_routing_cache is True


def test_static_mode_requires_a_checkpoint():
    with pytest.raises(OvalixGuardrailMissingSecrets):
        OvalixGuardrail(
            tracker_api_base="https://tracker.test",
            tracker_api_key="key",
            application_id="app-1",
            guardrail_name="ovalix-test",
            event_hook="pre_call",
            default_on=True,
        )


def test_static_one_sided_config_registers_only_that_hook():
    pre_only = OvalixGuardrail(
        tracker_api_base="https://tracker.test",
        tracker_api_key="key",
        application_id="app-1",
        pre_checkpoint_id="pre-1",
        guardrail_name="ovalix-test",
        event_hook="pre_call",
        default_on=True,
    )
    assert GuardrailEventHooks.pre_call in pre_only.supported_event_hooks
    assert GuardrailEventHooks.post_call not in pre_only.supported_event_hooks

    post_only = OvalixGuardrail(
        tracker_api_base="https://tracker.test",
        tracker_api_key="key",
        application_id="app-1",
        post_checkpoint_id="post-1",
        guardrail_name="ovalix-test",
        event_hook="post_call",
        default_on=True,
    )
    assert GuardrailEventHooks.post_call in post_only.supported_event_hooks
    assert GuardrailEventHooks.pre_call not in post_only.supported_event_hooks


def test_discovery_mode_registers_both_hooks():
    guardrail = OvalixGuardrail(
        tracker_api_base="https://tracker.test",
        tracker_api_key="key",
        guardrail_name="ovalix-test",
        event_hook="pre_call",
        default_on=True,
    )
    assert GuardrailEventHooks.pre_call in guardrail.supported_event_hooks
    assert GuardrailEventHooks.post_call in guardrail.supported_event_hooks


class TestOvalixGuardrailConfigModel:
    """Minimal config model tests: wiring only."""

    def test_get_config_model_returns_ovalix_config_model(self):
        """get_config_model returns OvalixGuardrailConfigModel for proxy/config wiring."""
        config_model = OvalixGuardrail.get_config_model()
        assert config_model is not None
        assert config_model.__name__ == "OvalixGuardrailConfigModel"
        assert config_model.ui_friendly_name() == "Ovalix Guardrail"


class TestOvalixGuardrail:
    """Behavioral tests with mocked Tracker checkpoint API."""

    def setup_method(self):
        for key in list(os.environ.keys()):
            if key.startswith("OVALIX_"):
                del os.environ[key]

    def teardown_method(self):
        for key in list(os.environ.keys()):
            if key.startswith("OVALIX_"):
                del os.environ[key]

    @pytest.fixture
    def guardrail_with_env(self):
        """Guardrail with OVALIX_* env set; cleans up in teardown."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            yield OvalixGuardrail(**_guardrail_kwargs())
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    def test_initialization_requires_secrets(self):
        """Initialization raises when required Tracker/application/checkpoint config is missing."""
        with pytest.raises(OvalixGuardrailMissingSecrets):
            OvalixGuardrail(
                guardrail_name="ovalix-test",
                event_hook="pre_call",
                default_on=True,
            )

    def test_initialization_with_explicit_params(self):
        """Guardrail initializes with explicit tracker base, key, app and checkpoint IDs."""
        guardrail = OvalixGuardrail(
            tracker_api_base="https://tracker.example",
            tracker_api_key="secret",
            application_id="app-x",
            pre_checkpoint_id="pre-x",
            post_checkpoint_id="post-x",
            **_guardrail_kwargs(),
        )
        assert guardrail._tracker_api_base == "https://tracker.example"
        assert guardrail._application_id == "app-x"
        assert guardrail._pre_checkpoint_id == "pre-x"
        assert guardrail._post_checkpoint_id == "post-x"

    def test_initialization_with_env_vars(self):
        """Guardrail picks up OVALIX_* env vars when params not passed."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            assert guardrail._tracker_api_base == "https://tracker.test"
            assert guardrail._tracker_api_key == "key"
            assert guardrail._application_id == "app-1"
            assert guardrail._pre_checkpoint_id == "pre-1"
            assert guardrail._post_checkpoint_id == "post-1"
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_call_checkpoint_sends_correct_payload_and_returns_json(self):
        """_call_checkpoint POSTs to tracker with application_id, checkpoint_id, actor, session_id, data, tool."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            mock_response = MagicMock()
            mock_response.json.return_value = TRACKER_RESPONSE_ALLOW
            mock_response.raise_for_status = MagicMock()

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                result = await guardrail._call_checkpoint(
                    data_type="TEXT",
                    data={"content": "hello"},
                    checkpoint_id="pre-1",
                    actor="a1b2c3d4",
                    session_id="session-1",
                    target=CheckpointTarget("app-1", "request"),
                )

            assert result == TRACKER_RESPONSE_ALLOW
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args.args[0] == ("https://tracker.test/tracking/beta/checkpoint")
            body = call_args.kwargs["json"]
            assert body["application_id"] == "app-1"
            assert body["checkpoint_id"] == "pre-1"
            assert body["actor"] == "a1b2c3d4"
            assert body["session_id"] == "session-1"
            assert body["data_type"] == "TEXT"
            assert body["data"] == {"content": "hello"}
            assert body["tool"] == "LiteLLM"
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_allow_passes_through(self):
        """When Tracker returns allow, apply_guardrail returns inputs with texts set to modified_data content."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[{"role": "user", "content": "how are you?"}],
                texts=["how are you?"],
            )
            request_data = {}

            mock_response = MagicMock()
            mock_response.json.return_value = TRACKER_RESPONSE_ALLOW
            mock_response.raise_for_status = MagicMock()

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                result = await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="request",
                    logging_obj=None,
                )

            assert result.get("texts") == ["how are you?"]
            assert mock_post.call_count == 1
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_anonymize_returns_modified_text(self):
        """When Tracker returns anonymize, apply_guardrail returns texts with modified_data content."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[{"role": "user", "content": "Hello, my name is David."}],
                texts=["Hello, my name is David."],
            )
            request_data = {}

            mock_response = MagicMock()
            mock_response.json.return_value = TRACKER_RESPONSE_ANONYMIZE
            mock_response.raise_for_status = MagicMock()

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                result = await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="request",
                    logging_obj=None,
                )

            assert result.get("texts") == ["Hello, my name is {Name}. How are you?"]
            assert mock_post.call_count == 1
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_block_raises_with_tracker_message(self):
        """When Tracker returns block on the (chronologically) last user message, OvalixGuardrailBlockedException is raised."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[{"role": "user", "content": "I am 15 YO"}],
                texts=["I am 15 YO"],
            )
            request_data = {}

            mock_response = MagicMock()
            mock_response.json.return_value = TRACKER_RESPONSE_BLOCK
            mock_response.raise_for_status = MagicMock()

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                with pytest.raises(OvalixGuardrailBlockedException) as exc_info:
                    await guardrail.apply_guardrail(
                        inputs=inputs,
                        request_data=request_data,
                        input_type="request",
                        logging_obj=None,
                    )

            assert "This message was blocked by Ovalix" in str(exc_info.value.message)
            assert exc_info.value.status_code == 400
            assert mock_post.call_count == 1
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_block_non_last_replaced_in_texts(self):
        """When Tracker returns block on a non-last user message, that message is replaced in texts and no exception is raised."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[
                    {"role": "user", "content": "I am 15 YO"},
                    {"role": "user", "content": "how are you?"},
                ],
                texts=["I am 15 YO", "how are you?"],
            )
            request_data = {}

            def side_effect(*args, **kwargs):
                body = kwargs.get("json", {})
                content = (body.get("data") or {}).get("content", "")
                resp = MagicMock()
                if "15" in content:
                    resp.json.return_value = TRACKER_RESPONSE_BLOCK
                else:
                    resp.json.return_value = TRACKER_RESPONSE_ALLOW
                resp.raise_for_status = MagicMock()
                return resp

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                mock_post.side_effect = side_effect
                result = await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="request",
                    logging_obj=None,
                )

            assert result.get("texts") == [
                "This message was blocked by Ovalix",
                "how are you?",
            ]
            assert mock_post.call_count == 2
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_allow_returns_inputs(self):
        """When input_type is response and Tracker allows, apply_guardrail leaves texts unchanged (allow never rewrites)."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[{"role": "assistant", "content": "Safe assistant reply"}],
                texts=["Safe assistant reply"],
            )
            request_data = {}

            mock_response = MagicMock()
            mock_response.json.return_value = TRACKER_RESPONSE_ALLOW
            mock_response.raise_for_status = MagicMock()

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                result = await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="response",
                    logging_obj=None,
                )

            assert result.get("texts") == ["Safe assistant reply"]
            assert mock_post.call_count == 1
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_block_raises(self, guardrail_with_env):
        """When Tracker blocks on response, apply_guardrail raises OvalixGuardrailBlockedException."""
        guardrail = guardrail_with_env
        inputs = GenericGuardrailAPIInputs(
            structured_messages=[{"role": "user", "content": "I am 15 YO"}],
            texts=["I am 15 YO"],
        )
        request_data = {}

        mock_response = MagicMock()
        mock_response.json.return_value = TRACKER_RESPONSE_BLOCK
        mock_response.raise_for_status = MagicMock()

        with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(OvalixGuardrailBlockedException) as exc_info:
                await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="response",
                    logging_obj=None,
                )

        assert "This message was blocked by Ovalix" in str(exc_info.value.message)
        assert exc_info.value.status_code == 400
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_missing_modified_data_uses_original_content(self, guardrail_with_env):
        """When Tracker response has no modified_data.content, original content is used."""
        guardrail = guardrail_with_env
        inputs = GenericGuardrailAPIInputs(
            structured_messages=[{"role": "user", "content": "original text"}],
            texts=["original text"],
        )
        request_data = {}

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "action_type": "allow",
            "data_type": "TEXT",
            "original_data": {"content": "original text"},
            "modified_data": {},
            "alerts": [],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
                logging_obj=None,
            )

        assert result.get("texts") == ["original text"]
        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_apply_guardrail_tracker_http_error_raises_guardrail_exception(self, guardrail_with_env):
        """When Tracker returns HTTP error (e.g. 400), GuardrailRaisedException is raised."""
        guardrail = guardrail_with_env
        inputs = GenericGuardrailAPIInputs(
            structured_messages=[{"role": "user", "content": "hello"}],
            texts=["hello"],
        )
        request_data = {}

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=MagicMock(),
            response=MagicMock(status_code=400),
        )

        with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(GuardrailRaisedException):
                await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="request",
                    logging_obj=None,
                )

        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_apply_guardrail_checkpoint_error_raises_guardrail_exception(self):
        """When Tracker checkpoint call fails, GuardrailRaisedException is raised."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[{"role": "user", "content": "hello"}],
                texts=["hello"],
            )
            request_data = {}

            with patch.object(
                guardrail._async_handler,
                "post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("Connection refused"),
            ):
                with pytest.raises(GuardrailRaisedException):
                    await guardrail.apply_guardrail(
                        inputs=inputs,
                        request_data=request_data,
                        input_type="request",
                        logging_obj=None,
                    )
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    @pytest.mark.asyncio
    async def test_apply_guardrail_request_empty_messages_returns_inputs(self):
        """When request has no messages, apply_guardrail returns inputs without calling Tracker."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs(structured_messages=[], texts=[])
            request_data = {}

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                result = await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="request",
                    logging_obj=None,
                )

            assert result == inputs
            mock_post.assert_not_called()
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    def test_get_actor_from_metadata(self):
        """Actor is taken from metadata.user_api_key_user_email or user_api_key_user_id."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            assert guardrail._get_actor({"metadata": {"user_api_key_user_email": "a@b.com"}}) == "a@b.com"
            assert guardrail._get_actor({"metadata": {"user_api_key_user_id": "uid-1"}}) == "uid-1"
            assert guardrail._get_actor({"litellm_metadata": {"user_api_key_user_id": "uid-2"}}) == "uid-2"
            assert guardrail._get_actor({}) == ""
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]

    def test_get_actor_prefers_email_over_id(self, guardrail_with_env):
        """When both user_api_key_user_email and user_api_key_user_id exist, email is used."""
        guardrail = guardrail_with_env
        data = {
            "metadata": {
                "user_api_key_user_email": "primary@test.com",
                "user_api_key_user_id": "uid-99",
            }
        }
        assert guardrail._get_actor(data) == "primary@test.com"

    def test_get_tracker_actor_id_is_hash_not_raw_pii(self, guardrail_with_env):
        """Tracker API actor field uses a short hash of _get_actor, not email/user id."""
        guardrail = guardrail_with_env
        data = {"metadata": {"user_api_key_user_email": "user@example.com"}}
        raw = guardrail._get_actor(data)
        hashed = guardrail._get_tracker_actor_id(data)
        assert raw == "user@example.com"
        assert hashed != raw
        assert len(hashed) == 8
        assert all(c in "0123456789abcdef" for c in hashed)

    def test_get_session_id_deterministic_and_includes_app_id(self, guardrail_with_env):
        """Session ID is stable for same actor/day and includes application_id."""
        guardrail = guardrail_with_env
        data = {"metadata": {"user_api_key_user_id": "user-1"}}
        session_id_1 = guardrail._get_session_id(data)
        session_id_2 = guardrail._get_session_id(data)
        assert session_id_1 == session_id_2
        assert "app-1" in session_id_1

    def test_block_current_message_raises_ovalix_blocked_exception(self, guardrail_with_env):
        """_block_current_message raises OvalixGuardrailBlockedException with status_code 400."""
        guardrail = guardrail_with_env
        with pytest.raises(OvalixGuardrailBlockedException) as exc_info:
            guardrail._block_current_message("Custom block reason")
        assert "Custom block reason" in str(exc_info.value.message)
        assert exc_info.value.status_code == 400

    def test_get_trackers_corrected_message(self, guardrail_with_env):
        """_get_trackers_corrected_message returns modified_data.content or None."""
        guardrail = guardrail_with_env
        assert (
            guardrail._get_trackers_corrected_message({"modified_data": {"content": "corrected text"}})
            == "corrected text"
        )
        assert guardrail._get_trackers_corrected_message({"modified_data": {}}) is None
        assert guardrail._get_trackers_corrected_message({"modified_data": "not-a-dict"}) is None

    @pytest.mark.asyncio
    async def test_apply_guardrail_response_no_texts_returns_unchanged(self):
        """When input_type is response and inputs have no texts, apply_guardrail returns inputs without calling Tracker."""
        for k, v in _ovalix_env().items():
            os.environ[k] = v
        try:
            guardrail = OvalixGuardrail(**_guardrail_kwargs())
            inputs = GenericGuardrailAPIInputs()
            request_data = {}

            with patch.object(guardrail._async_handler, "post", new_callable=AsyncMock) as mock_post:
                result = await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="response",
                    logging_obj=None,
                )

            assert result == inputs
            mock_post.assert_not_called()
        finally:
            for k in _ovalix_env():
                if k in os.environ:
                    del os.environ[k]


_REGEX = r"^\s*\[([^\]]+)\]"
_ROUTING_BODY = {
    "application_id": "app-9",
    "checkpoint_id_pre": "pre",
    "checkpoint_id_post": "post",
    "checkpoint_id_pre_file": None,
    "checkpoint_id_post_file": None,
}


def _discovery_guardrail(enable_cache=True):
    return OvalixGuardrail(
        tracker_api_base="https://tracker.test",
        tracker_api_key="key",
        enable_routing_cache=enable_cache,
        guardrail_name="ovalix-test",
        event_hook="pre_call",
        default_on=True,
    )


def _alias_request_data(alias="[Weather App] prod"):
    return {"metadata": {"user_api_key_alias": alias, "user_api_key_user_email": "u@x.com"}}


def _routing_body(pre, post, pre_file, post_file, application_id="app-9"):
    return {
        "application_id": application_id,
        "checkpoint_id_pre": pre,
        "checkpoint_id_post": post,
        "checkpoint_id_pre_file": pre_file,
        "checkpoint_id_post_file": post_file,
    }


def _checkpoint_calls(mock_post):
    """(body, url) of the tracker checkpoint calls only, excluding regex/resolve traffic."""
    return [
        (c.kwargs["json"], c.args[0])
        for c in mock_post.call_args_list
        if c.args and c.args[0].endswith(("/checkpoint", "/file_checkpoint"))
    ]


def _checkpoint_bodies(mock_post):
    return [body for body, _ in _checkpoint_calls(mock_post)]


def _route_of(url):
    return url.rsplit("/", 1)[-1]


def _mock_handler(g, routing=None):
    get_resp = MagicMock()
    get_resp.json.return_value = {"regex": _REGEX}
    get_resp.raise_for_status = MagicMock()
    post_resp = MagicMock()
    post_resp.json.return_value = routing or _ROUTING_BODY
    post_resp.raise_for_status = MagicMock()
    g._async_handler.get = AsyncMock(return_value=get_resp)
    g._async_handler.post = AsyncMock(return_value=post_resp)
    return g._async_handler.get, g._async_handler.post


@pytest.mark.asyncio
async def test_static_mode_uses_config_routing():
    g = OvalixGuardrail(
        tracker_api_base="https://t",
        tracker_api_key="k",
        application_id="app-1",
        pre_checkpoint_id="pre-1",
        post_checkpoint_id="post-1",
        file_checkpoint_id="file-1",
        guardrail_name="o",
        event_hook="pre_call",
        default_on=True,
    )
    routing = await g._resolve_routing({})
    assert routing == ResolvedRouting("app-1", "pre-1", "post-1", "file-1", "file-1")


@pytest.mark.asyncio
async def test_discovery_extracts_name_and_resolves():
    g = _discovery_guardrail(enable_cache=False)
    mock_get, mock_post = _mock_handler(g)
    routing = await g._resolve_routing(_alias_request_data("[Weather App] prod"))
    assert routing.application_id == "app-9"
    assert mock_post.call_args.args[0].endswith("/tracking/beta/resolve_application")
    assert mock_post.call_args.kwargs["json"] == {"application_name": "Weather App"}
    assert mock_get.call_args.args[0].endswith("/tracking/beta/app_name_regex")


@pytest.mark.asyncio
async def test_regex_fetched_once_even_with_cache_off():
    g = _discovery_guardrail(enable_cache=False)
    mock_get, mock_post = _mock_handler(g)
    await g._resolve_routing(_alias_request_data())
    await g._resolve_routing(_alias_request_data())
    assert mock_get.call_count == 1
    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_no_bracket_alias_fails_closed():
    g = _discovery_guardrail()
    _mock_handler(g)
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data("no brackets here"))


@pytest.mark.asyncio
async def test_discovery_missing_alias_raises():
    g = _discovery_guardrail()
    _mock_handler(g)
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing({"metadata": {"user_api_key_user_email": "u@x.com"}})


@pytest.mark.asyncio
async def test_regex_fetch_failure_raises_guardrail_exception():
    g = _discovery_guardrail(enable_cache=False)
    g._async_handler.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data())


@pytest.mark.asyncio
async def test_resolve_endpoint_failure_raises_guardrail_exception():
    g = _discovery_guardrail(enable_cache=False)
    _mock_handler(g)
    g._async_handler.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data())


@pytest.mark.asyncio
async def test_resolve_missing_application_id_raises_guardrail_exception():
    g = _discovery_guardrail(enable_cache=False)
    _mock_handler(g, routing={"checkpoint_id_pre": "pre", "checkpoint_id_post": "post"})
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data())


@pytest.mark.asyncio
async def test_routing_cache_hit_and_ttl_expiry(monkeypatch):
    g = _discovery_guardrail(enable_cache=True)
    mock_get, mock_post = _mock_handler(g)
    clock = [1000.0]
    monkeypatch.setattr("litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix.time.monotonic", lambda: clock[0])
    await g._resolve_routing(_alias_request_data())
    clock[0] = 1000.0 + 3599
    await g._resolve_routing(_alias_request_data())
    assert mock_post.call_count == 1
    clock[0] = 1000.0 + 3601
    await g._resolve_routing(_alias_request_data())
    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_routing_cache_lru_eviction(monkeypatch):
    monkeypatch.setattr("litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix._ROUTING_CACHE_MAX_SIZE", 2)
    g = _discovery_guardrail(enable_cache=True)
    _mock_handler(g)
    for name in ("[App A] x", "[App B] x", "[App C] x"):
        await g._resolve_routing(_alias_request_data(name))
    assert "App A" not in g._routing_cache and len(g._routing_cache) == 2


_ALLOW = {"action_type": "allow", "modified_data": {"content": "x"}}
_BLOCK = {"action_type": "block", "modified_data": {"content": "stop-reason"}}
_ANON = {"action_type": "anonymize", "modified_data": {"content": "redacted"}}


def _static_guardrail():
    return OvalixGuardrail(
        tracker_api_base="https://t",
        tracker_api_key="k",
        application_id="app-1",
        pre_checkpoint_id="pre-1",
        post_checkpoint_id="post-1",
        file_checkpoint_id="file-1",
        guardrail_name="o",
        event_hook="pre_call",
        default_on=True,
    )


def _post_returning(mapping_fn):
    resp_factory = mapping_fn

    async def _post(url, headers=None, json=None):
        r = MagicMock()
        r.json.return_value = resp_factory(json)
        r.raise_for_status = MagicMock()
        return r

    return _post


@pytest.mark.asyncio
async def test_file_block_raises():
    g = _static_guardrail()
    data_url = "data:text/plain;base64," + base64.b64encode(b"secret").decode()
    inputs = GenericGuardrailAPIInputs(
        texts=["hi"],
        structured_messages=[
            {"role": "user", "content": [{"type": "file", "file": {"filename": "s.txt", "file_data": data_url}}]}
        ],
    )
    with patch.object(
        g._async_handler, "post", new=_post_returning(lambda body: _BLOCK if body["data_type"] == "FILE" else _ALLOW)
    ):
        with pytest.raises(OvalixGuardrailBlockedException) as exc:
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert "stop-reason" in str(exc.value.message)


@pytest.mark.asyncio
async def test_file_uses_file_checkpoint_and_gzip_wire():
    g = _static_guardrail()
    data_url = "data:text/plain;base64," + base64.b64encode(b"secret").decode()
    inputs = GenericGuardrailAPIInputs(
        texts=[],
        structured_messages=[
            {"role": "user", "content": [{"type": "file", "file": {"filename": "s.txt", "file_data": data_url}}]}
        ],
    )
    seen = {}

    async def _post(url, headers=None, json=None):
        seen["last"] = json
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert seen["last"]["data_type"] == "FILE"
    assert seen["last"]["checkpoint_id"] == "file-1"
    assert gzip.decompress(base64.b64decode(seen["last"]["data"]["content"])) == b"secret"


@pytest.mark.asyncio
async def test_response_side_file_uses_file_checkpoint():
    g = _static_guardrail()
    data_url = "data:image/png;base64," + base64.b64encode(b"img").decode()
    inputs = GenericGuardrailAPIInputs(texts=[], images=[data_url])
    seen = {}

    async def _post(url, headers=None, json=None):
        seen["last"] = json
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="response", logging_obj=None)
    assert seen["last"]["data_type"] == "FILE"
    assert seen["last"]["checkpoint_id"] == "file-1"


@pytest.mark.asyncio
async def test_file_checkpoint_call_routes_to_litellm_file_endpoint():
    g = _static_guardrail()
    seen = {}

    async def _post(url, headers=None, json=None):
        seen["url"] = url
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g._call_checkpoint(
            "FILE", {"name": "f.txt", "content": "x"}, "file-1", "a", "s", CheckpointTarget("app-1", "request")
        )
    assert seen["url"] == "https://t/tracking/beta/file_checkpoint"


@pytest.mark.asyncio
async def test_tool_call_block_raises():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=[], tool_calls=[{"id": "c1", "type": "function", "function": {"name": "exfil", "arguments": "{}"}}]
    )
    with patch.object(
        g._async_handler, "post", new=_post_returning(lambda body: _BLOCK if body["data_type"] == "TOOL" else _ALLOW)
    ):
        with pytest.raises(OvalixGuardrailBlockedException):
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type="response", logging_obj=None)


@pytest.mark.asyncio
async def test_tool_call_anonymize_escalates_to_block():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=[], tool_calls=[{"id": "c1", "type": "function", "function": {"name": "exfil", "arguments": "{}"}}]
    )
    with patch.object(
        g._async_handler, "post", new=_post_returning(lambda body: _ANON if body["data_type"] == "TOOL" else _ALLOW)
    ):
        with pytest.raises(OvalixGuardrailBlockedException) as exc:
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type="response", logging_obj=None)
    assert "tool call anonymization isn't possible" in str(exc.value.message)


@pytest.mark.asyncio
async def test_tool_result_block_raises():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=["sunny"],
        structured_messages=[
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
    )
    with patch.object(
        g._async_handler, "post", new=_post_returning(lambda body: _BLOCK if body["data_type"] == "TOOL" else _ALLOW)
    ):
        with pytest.raises(OvalixGuardrailBlockedException):
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)


@pytest.mark.asyncio
async def test_newest_text_block_raises_older_anonymized():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(texts=["old", "new"])

    def _map(body):
        if body["data_type"] != "TEXT":
            return _ALLOW
        return _BLOCK if body["data"]["content"] == "new" else _ALLOW

    with patch.object(g._async_handler, "post", new=_post_returning(_map)):
        with pytest.raises(OvalixGuardrailBlockedException):
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)


@pytest.mark.asyncio
async def test_older_text_anonymized_returns_modified_texts():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(texts=["make me anon", "safe newest"])

    def _map(body):
        if body["data_type"] != "TEXT":
            return _ALLOW
        return (
            {"action_type": "anonymize", "modified_data": {"content": "ANON"}}
            if body["data"]["content"] == "make me anon"
            else _ALLOW
        )

    with patch.object(g._async_handler, "post", new=_post_returning(_map)):
        result = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result["texts"] == ["ANON", "safe newest"]


@pytest.mark.asyncio
async def test_all_allow_passes_through():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(texts=["hi"])
    with patch.object(g._async_handler, "post", new=_post_returning(lambda body: _ALLOW)):
        result = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result["texts"] == ["hi"]


@pytest.mark.asyncio
async def test_actor_sent_to_tracker_is_raw_identifier_not_hash():
    g = _static_guardrail()
    request_data = {"metadata": {"user_api_key_user_email": "user@example.com"}}
    inputs = GenericGuardrailAPIInputs(texts=["hi"])
    seen = {}

    async def _post(url, headers=None, json=None):
        seen["last"] = json
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request", logging_obj=None)
    assert seen["last"]["actor"] == "user@example.com"
    assert seen["last"]["session_id"] != "user@example.com"
    assert g._get_tracker_actor_id(request_data) in seen["last"]["session_id"]


@pytest.mark.asyncio
async def test_empty_user_sends_empty_actor_matching_reference():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(texts=["hi"])
    seen = {}

    async def _post(url, headers=None, json=None):
        seen["last"] = json
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert seen["last"]["actor"] == ""


def _recording_post(mapping_fn=lambda body: _ALLOW):
    calls = []

    async def _post(url, headers=None, json=None):
        calls.append((json["data_type"], json["data"].get("content")))
        r = MagicMock()
        r.json.return_value = mapping_fn(json)
        r.raise_for_status = MagicMock()
        return r

    return _post, calls


@pytest.mark.asyncio
async def test_every_checkpoint_payload_is_json_serializable():
    """httpx json-encodes the checkpoint body, so a non-dict mapping in it would 500 at runtime."""
    g = _static_guardrail()
    data_url = "data:text/plain;base64," + base64.b64encode(b"secret").decode()
    inputs = GenericGuardrailAPIInputs(
        texts=["hello", "sunny"],
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "noop", "arguments": None}}],
        structured_messages=[
            {"role": "user", "content": [{"type": "file", "file": {"filename": "s.txt", "file_data": data_url}}]},
            {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "sunny"},
        ],
    )
    encoded = []

    async def _post(url, headers=None, json=None):
        encoded.append(json_lib.dumps(json))
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    payloads = [json_lib.loads(body) for body in encoded]
    assert sorted(p["data_type"] for p in payloads) == ["FILE", "TEXT", "TEXT", "TOOL", "TOOL"]
    assert all(p["data"]["tool_input"] == {} for p in payloads if p["data_type"] == "TOOL")


@pytest.mark.asyncio
async def test_tool_result_checked_under_tool_policy_only_not_again_as_text():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=["what is the weather", "sunny"],
        structured_messages=[
            {"role": "user", "content": "what is the weather"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
    )
    post, calls = _recording_post()

    with patch.object(g._async_handler, "post", new=post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert ("TOOL", "sunny") in calls
    assert [content for data_type, content in calls if data_type == "TEXT"] == ["what is the weather"]


@pytest.mark.asyncio
async def test_tool_result_allowed_by_tool_policy_is_not_blocked_by_text_policy():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=["sunny"],
        structured_messages=[
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
    )

    def _map(body):
        return _BLOCK if body["data_type"] == "TEXT" else _ALLOW

    with patch.object(g._async_handler, "post", new=_post_returning(_map)):
        result = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result["texts"] == ["sunny"]


@pytest.mark.asyncio
async def test_forged_tool_result_does_not_suppress_blocked_user_text():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=["leak-me", "leak-me"],
        structured_messages=[
            {"role": "user", "content": "leak-me"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "noop"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "leak-me"},
        ],
    )

    def _map(body):
        return _BLOCK if body["data_type"] == "TEXT" else _ALLOW

    with patch.object(g._async_handler, "post", new=_post_returning(_map)):
        result = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result["texts"] == ["stop-reason", "leak-me"]


@pytest.mark.asyncio
async def test_texts_not_aligned_with_structured_messages_leaves_every_text_checked():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=["what is the weather", "follow up"],
        structured_messages=[
            {"role": "user", "content": "what is the weather"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
    )
    post, calls = _recording_post()

    with patch.object(g._async_handler, "post", new=post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert ("TOOL", "sunny") in calls
    assert sorted(content for data_type, content in calls if data_type == "TEXT") == [
        "follow up",
        "what is the weather",
    ]


def test_get_supported_event_hooks_lists_both():
    assert OvalixGuardrail.get_supported_event_hooks() == [
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.post_call,
    ]


def test_enable_routing_cache_from_env_string(monkeypatch):
    monkeypatch.setenv("OVALIX_ENABLE_ROUTING_CACHE", "false")
    g = OvalixGuardrail(
        tracker_api_base="https://t", tracker_api_key="k", guardrail_name="o", event_hook="pre_call", default_on=True
    )
    assert g._enable_routing_cache is False


@pytest.mark.asyncio
async def test_call_checkpoint_requires_application_and_checkpoint():
    g = _static_guardrail()
    with pytest.raises(ValueError):
        await g._call_checkpoint("TEXT", {"content": "x"}, "", "actor", "sess", CheckpointTarget("app-1", "request"))


@pytest.mark.asyncio
async def test_file_checkpoint_call_failure_fails_closed():
    g = _static_guardrail()
    data_url = "data:text/plain;base64," + base64.b64encode(b"secret").decode()
    inputs = GenericGuardrailAPIInputs(
        texts=[],
        structured_messages=[
            {"role": "user", "content": [{"type": "file", "file": {"filename": "s.txt", "file_data": data_url}}]}
        ],
    )
    with patch.object(g._async_handler, "post", new=AsyncMock(side_effect=httpx.ConnectError("boom"))):
        with pytest.raises(GuardrailRaisedException):
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)


@pytest.mark.asyncio
async def test_discovery_resolved_without_any_checkpoint_raises():
    """An application with no checkpoints in any direction is a tracker misconfiguration, so fail closed."""
    g = _discovery_guardrail(enable_cache=False)
    _mock_handler(g, routing=_routing_body(None, None, None, None))
    inputs = GenericGuardrailAPIInputs(texts=["hi"])
    with pytest.raises(GuardrailRaisedException) as exc:
        await g.apply_guardrail(
            inputs=inputs, request_data=_alias_request_data(), input_type="request", logging_obj=None
        )
    assert "no checkpoints configured" in str(exc.value.message)


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type, inspected", [("request", ["request"]), ("response", [])])
async def test_one_sided_discovery_inspects_configured_direction_only(input_type, inspected):
    """Discovery registers both hooks speculatively, so a direction the app does not inspect must pass through."""
    g = _discovery_guardrail(enable_cache=False)
    _, mock_post = _mock_handler(g, routing=_routing_body("pre-9", None, None, None))
    inputs = GenericGuardrailAPIInputs(texts=["hi"])
    result = await g.apply_guardrail(
        inputs=inputs, request_data=_alias_request_data(), input_type=input_type, logging_obj=None
    )
    assert result["texts"] == ["hi"]
    assert [b["input_type"] for b in _checkpoint_bodies(mock_post)] == inspected


@pytest.mark.asyncio
async def test_file_only_checkpoint_inspects_files_and_skips_text():
    """A direction with just a file checkpoint still scans files; text and tools need a prompt checkpoint."""
    g = _discovery_guardrail(enable_cache=False)
    _, mock_post = _mock_handler(g, routing=_routing_body(None, None, "pre-file-9", None))
    data_url = "data:text/plain;base64," + base64.b64encode(b"secret").decode()
    inputs = GenericGuardrailAPIInputs(
        texts=["hi"],
        structured_messages=[
            {"role": "user", "content": [{"type": "file", "file": {"filename": "s.txt", "file_data": data_url}}]}
        ],
    )
    result = await g.apply_guardrail(
        inputs=inputs, request_data=_alias_request_data(), input_type="request", logging_obj=None
    )
    assert result["texts"] == ["hi"]
    assert [(b["data_type"], _route_of(c)) for b, c in _checkpoint_calls(mock_post)] == [("FILE", "file_checkpoint")]


def test_initialize_guardrail_wires_new_params(monkeypatch):
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.ovalix import initialize_guardrail

    monkeypatch.setattr(litellm.logging_callback_manager, "add_litellm_callback", lambda callback: None)

    class _Params:
        tracker_api_base = "https://t"
        tracker_api_key = "k"
        application_id = "app-1"
        pre_checkpoint_id = "pre-1"
        post_checkpoint_id = "post-1"
        file_checkpoint_id = "file-1"
        enable_routing_cache = False
        mode = "pre_call"
        default_on = True

    guardrail = initialize_guardrail(_Params(), {"guardrail_name": "ovalix"})
    assert guardrail._file_checkpoint_id == "file-1"
    assert guardrail._enable_routing_cache is False
    assert guardrail.guardrail_name == "ovalix"


def test_fail_if_no_application_defaults_true():
    g = OvalixGuardrail(
        tracker_api_base="https://t", tracker_api_key="k", guardrail_name="o", event_hook="pre_call", default_on=True
    )
    assert g._fail_if_no_application is True


def test_fail_if_no_application_from_param():
    g = OvalixGuardrail(
        tracker_api_base="https://t",
        tracker_api_key="k",
        fail_if_no_application=False,
        guardrail_name="o",
        event_hook="pre_call",
        default_on=True,
    )
    assert g._fail_if_no_application is False


def test_fail_if_no_application_from_env_string(monkeypatch):
    monkeypatch.setenv("OVALIX_FAIL_IF_NO_APPLICATION", "false")
    g = OvalixGuardrail(
        tracker_api_base="https://t", tracker_api_key="k", guardrail_name="o", event_hook="pre_call", default_on=True
    )
    assert g._fail_if_no_application is False


def test_explicit_param_beats_env_for_fail_if_no_application(monkeypatch):
    monkeypatch.setenv("OVALIX_FAIL_IF_NO_APPLICATION", "false")
    g = OvalixGuardrail(
        tracker_api_base="https://t",
        tracker_api_key="k",
        fail_if_no_application=True,
        guardrail_name="o",
        event_hook="pre_call",
        default_on=True,
    )
    assert g._fail_if_no_application is True


def test_initialize_guardrail_wires_fail_if_no_application(monkeypatch):
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.ovalix import initialize_guardrail

    monkeypatch.setattr(litellm.logging_callback_manager, "add_litellm_callback", lambda callback: None)

    class _Params:
        tracker_api_base = "https://t"
        tracker_api_key = "k"
        application_id = None
        pre_checkpoint_id = None
        post_checkpoint_id = None
        file_checkpoint_id = None
        enable_routing_cache = None
        fail_if_no_application = False
        mode = "pre_call"
        default_on = True

    guardrail = initialize_guardrail(_Params(), {"guardrail_name": "ovalix"})
    assert guardrail._fail_if_no_application is False


def test_config_model_declares_fail_if_no_application():
    from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import OvalixGuardrailConfigModel

    assert OvalixGuardrailConfigModel.model_fields["fail_if_no_application"].default is None


def _fail_open_discovery_guardrail(enable_cache=False):
    return OvalixGuardrail(
        tracker_api_base="https://tracker.test",
        tracker_api_key="key",
        enable_routing_cache=enable_cache,
        fail_if_no_application=False,
        guardrail_name="ovalix-test",
        event_hook="pre_call",
        default_on=True,
    )


def _http_status_error(status_code):
    request = httpx.Request("POST", "https://tracker.test/x")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _mock_handler_resolve_status(g, status_code):
    get_resp = MagicMock()
    get_resp.json.return_value = {"regex": _REGEX}
    get_resp.raise_for_status = MagicMock()
    post_resp = MagicMock()
    post_resp.raise_for_status = MagicMock(side_effect=_http_status_error(status_code))
    g._async_handler.get = AsyncMock(return_value=get_resp)
    g._async_handler.post = AsyncMock(return_value=post_resp)
    return g._async_handler.post


@pytest.mark.asyncio
async def test_fail_open_missing_alias_returns_none():
    g = _fail_open_discovery_guardrail()
    _mock_handler(g)
    assert await g._resolve_routing({"metadata": {"user_api_key_user_email": "u@x.com"}}) is None


@pytest.mark.asyncio
async def test_fail_open_unparseable_alias_returns_none():
    g = _fail_open_discovery_guardrail()
    _mock_handler(g)
    assert await g._resolve_routing(_alias_request_data("no brackets here")) is None


@pytest.mark.asyncio
async def test_fail_open_tracker_404_returns_none():
    g = _fail_open_discovery_guardrail()
    _mock_handler_resolve_status(g, 404)
    assert await g._resolve_routing(_alias_request_data("[Ghost App] prod")) is None


@pytest.mark.asyncio
async def test_fail_closed_tracker_404_still_raises():
    g = _discovery_guardrail(enable_cache=False)
    _mock_handler_resolve_status(g, 404)
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data("[Ghost App] prod"))


@pytest.mark.asyncio
async def test_fail_open_tracker_500_still_raises():
    g = _fail_open_discovery_guardrail()
    _mock_handler_resolve_status(g, 500)
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data("[Weather App] prod"))


@pytest.mark.asyncio
async def test_fail_open_tracker_unreachable_still_raises():
    g = _fail_open_discovery_guardrail()
    _mock_handler(g)
    g._async_handler.post = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data())


@pytest.mark.asyncio
async def test_fail_open_regex_fetch_failure_still_raises():
    g = _fail_open_discovery_guardrail()
    g._async_handler.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(GuardrailRaisedException):
        await g._resolve_routing(_alias_request_data())


@pytest.mark.asyncio
async def test_fail_open_no_application_makes_no_checkpoint_call():
    g = _fail_open_discovery_guardrail()
    _mock_handler_resolve_status(g, 404)
    checkpoint = AsyncMock()
    with patch.object(g, "_call_checkpoint", new=checkpoint):
        inputs = GenericGuardrailAPIInputs(texts=["hello"])
        result = await g.apply_guardrail(
            inputs=inputs,
            request_data=_alias_request_data("[Ghost App] prod"),
            input_type="request",
            logging_obj=None,
        )
    assert result == inputs
    checkpoint.assert_not_awaited()


@pytest.mark.asyncio
async def test_fail_open_resolved_application_without_checkpoint_still_raises():
    g = _fail_open_discovery_guardrail()
    _mock_handler(
        g,
        routing={
            "application_id": "app-9",
            "checkpoint_id_pre": None,
            "checkpoint_id_post": None,
            "checkpoint_id_pre_file": None,
            "checkpoint_id_post_file": None,
        },
    )
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=["hi"]),
            request_data=_alias_request_data(),
            input_type="request",
            logging_obj=None,
        )


@pytest.mark.asyncio
async def test_fail_open_still_guards_when_application_resolves():
    g = _fail_open_discovery_guardrail()
    _mock_handler(g)
    checkpoint = AsyncMock(return_value=_BLOCK)
    with patch.object(g, "_call_checkpoint", new=checkpoint):
        with pytest.raises(OvalixGuardrailBlockedException):
            await g.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(texts=["hi"]),
                request_data=_alias_request_data(),
                input_type="request",
                logging_obj=None,
            )


@pytest.mark.asyncio
async def test_negative_routing_cache_hit_and_expiry(monkeypatch):
    g = _fail_open_discovery_guardrail(enable_cache=True)
    mock_post = _mock_handler_resolve_status(g, 404)
    clock = [1000.0]
    monkeypatch.setattr("litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix.time.monotonic", lambda: clock[0])

    assert await g._resolve_routing(_alias_request_data("[Ghost App] prod")) is None
    clock[0] = 1000.0 + 299
    assert await g._resolve_routing(_alias_request_data("[Ghost App] prod")) is None
    assert mock_post.call_count == 1

    clock[0] = 1000.0 + 301
    assert await g._resolve_routing(_alias_request_data("[Ghost App] prod")) is None
    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_negative_cache_expires_sooner_than_positive(monkeypatch):
    g = _discovery_guardrail(enable_cache=True)
    _mock_handler(g)
    clock = [1000.0]
    monkeypatch.setattr("litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix.time.monotonic", lambda: clock[0])
    await g._resolve_routing(_alias_request_data("[Weather App] prod"))
    clock[0] = 1000.0 + 301
    hit, cached = g._routing_cache_get("Weather App")
    assert hit is True and cached is not None


@pytest.mark.asyncio
async def test_negative_result_not_cached_when_cache_disabled():
    g = _fail_open_discovery_guardrail(enable_cache=False)
    mock_post = _mock_handler_resolve_status(g, 404)
    await g._resolve_routing(_alias_request_data("[Ghost App] prod"))
    await g._resolve_routing(_alias_request_data("[Ghost App] prod"))
    assert mock_post.call_count == 2
    assert len(g._routing_cache) == 0


@pytest.mark.asyncio
async def test_cached_404_still_raises_when_failing_closed(monkeypatch):
    g = _discovery_guardrail(enable_cache=True)
    mock_post = _mock_handler_resolve_status(g, 404)
    clock = [1000.0]
    monkeypatch.setattr("litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix.time.monotonic", lambda: clock[0])
    for _ in range(2):
        with pytest.raises(GuardrailRaisedException):
            await g._resolve_routing(_alias_request_data("[Ghost App] prod"))
    assert mock_post.call_count == 1


def _alias_guardrail():
    return OvalixGuardrail(
        tracker_api_base="https://t",
        tracker_api_key="k",
        guardrail_name="o",
        event_hook="pre_call",
        default_on=True,
    )


def _capturing_post(response=None):
    seen = {}

    async def _post(url, headers=None, json=None):
        seen["url"] = url
        seen["body"] = json
        r = MagicMock()
        r.json.return_value = response if response is not None else _ALLOW
        r.raise_for_status = MagicMock()
        return r

    return seen, _post


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_type,expected_route",
    [("TEXT", "checkpoint"), ("TOOL", "checkpoint"), ("FILE", "file_checkpoint")],
)
async def test_call_checkpoint_by_name_sends_name_and_direction_not_ids(data_type, expected_route):
    """Name routing must put application_name and input_type on the wire and no ids at all.

    The two forms are mutually exclusive server-side, so leaking an id alongside the name is rejected
    """
    g = _alias_guardrail()
    seen, post = _capturing_post()

    with patch.object(g._async_handler, "post", new=post):
        await g._call_checkpoint(
            data_type,
            {"content": "x"},
            "unused-cp",
            "actor",
            "sess",
            CheckpointTarget("app-9", "response", application_name="Weather App"),
        )

    assert seen["url"] == f"https://t/tracking/beta/{expected_route}"
    assert seen["body"]["application_name"] == "Weather App"
    assert seen["body"]["input_type"] == "response"
    assert "application_id" not in seen["body"]
    assert "checkpoint_id" not in seen["body"]


@pytest.mark.asyncio
async def test_call_checkpoint_by_ids_sends_ids_and_no_name_or_direction():
    g = _static_guardrail()
    seen, post = _capturing_post()

    with patch.object(g._async_handler, "post", new=post):
        await g._call_checkpoint(
            "TEXT", {"content": "x"}, "pre-1", "actor", "sess", CheckpointTarget("app-1", "request")
        )

    assert seen["body"]["application_id"] == "app-1"
    assert seen["body"]["checkpoint_id"] == "pre-1"
    assert "application_name" not in seen["body"]
    assert "input_type" not in seen["body"]


@pytest.mark.asyncio
async def test_call_checkpoint_by_name_does_not_require_a_checkpoint_id():
    """The tracker chooses the checkpoint under name routing, so an empty id must not be rejected."""
    g = _alias_guardrail()
    seen, post = _capturing_post()

    with patch.object(g._async_handler, "post", new=post):
        await g._call_checkpoint(
            "TEXT",
            {"content": "x"},
            "",
            "actor",
            "sess",
            CheckpointTarget("", "request", application_name="Weather App"),
        )

    assert seen["body"]["application_name"] == "Weather App"


@pytest.mark.asyncio
async def test_checkpoint_routing_name_is_none_when_application_is_pinned():
    """A deployment that pins application_id reads no alias, so it must keep routing by ids."""
    g = _static_guardrail()

    assert await g._checkpoint_routing_name({"metadata": {"user_api_key_alias": "[Weather App] k"}}) is None


@pytest.mark.asyncio
async def test_checkpoint_routing_name_extracted_from_key_alias():
    g = _alias_guardrail()
    with patch.object(g, "_get_app_name_regex", new=AsyncMock(return_value=re.compile(r"^\s*\[([^\]]+)\]"))):
        name = await g._checkpoint_routing_name({"metadata": {"user_api_key_alias": "[Weather App] free text"}})

    assert name == "Weather App"


@pytest.mark.asyncio
async def test_checkpoint_routing_name_is_none_without_an_alias():
    g = _alias_guardrail()

    assert await g._checkpoint_routing_name({"metadata": {}}) is None


@pytest.mark.asyncio
async def test_apply_guardrail_resolved_by_alias_routes_checkpoints_by_name():
    """End to end: an alias-resolved application sends its name per checkpoint, never the resolved id.

    This is what keeps a just-created application inspectable: the ids from resolution may name an
    application the tracker's process-wide config has not picked up yet
    """
    g = _alias_guardrail()
    seen, post = _capturing_post()

    with (
        patch.object(
            g, "_resolve_routing", new=AsyncMock(return_value=ResolvedRouting("app-9", "pre-9", "post-9", None, None))
        ),
        patch.object(g, "_get_app_name_regex", new=AsyncMock(return_value=re.compile(r"^\s*\[([^\]]+)\]"))),
        patch.object(g._async_handler, "post", new=post),
    ):
        await g.apply_guardrail(
            GenericGuardrailAPIInputs(texts=["hello"]),
            {"metadata": {"user_api_key_alias": "[Weather App] k", "user_api_key_user_email": "u@e.com"}},
            "request",
        )

    assert seen["url"] == "https://t/tracking/beta/checkpoint"
    assert seen["body"]["application_name"] == "Weather App"
    assert seen["body"]["input_type"] == "request"
    assert "application_id" not in seen["body"]
