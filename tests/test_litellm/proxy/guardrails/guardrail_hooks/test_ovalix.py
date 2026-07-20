"""
Unit tests for Ovalix guardrail: config resolution and apply_guardrail behavior
with mocked Tracker service responses (allow, anonymize, block).
"""

import os
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix import (
    OvalixGuardrail,
    OvalixGuardrailBlockedException,
    OvalixGuardrailMissingSecrets,
    ResolvedRouting,
)
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
        """_call_checkpoint POSTs to tracker with application_id, checkpoint_id, actor, session_id, data."""
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
                    content="hello",
                    checkpoint_id="pre-1",
                    actor="a1b2c3d4",
                    session_id="session-1",
                )

            assert result == TRACKER_RESPONSE_ALLOW
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args.args[0] == ("https://tracker.test/tracking/custom_application/checkpoint")
            body = call_args.kwargs["json"]
            assert body["application_id"] == "app-1"
            assert body["checkpoint_id"] == "pre-1"
            assert body["actor"] == "a1b2c3d4"
            assert body["session_id"] == "session-1"
            assert body["data_type"] == "TEXT"
            assert body["data"] == {"content": "hello"}
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
        """When input_type is response and Tracker allows, apply_guardrail returns inputs with texts updated from Tracker."""
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

            assert result.get("texts") == ["how are you?"]
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
            assert guardrail._get_actor({}) == "unknown"
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
    assert routing == ResolvedRouting("app-1", "pre-1", "post-1", "file-1", None)


@pytest.mark.asyncio
async def test_discovery_extracts_name_and_resolves():
    g = _discovery_guardrail(enable_cache=False)
    mock_get, mock_post = _mock_handler(g)
    routing = await g._resolve_routing(_alias_request_data("[Weather App] prod"))
    assert routing.application_id == "app-9"
    assert mock_post.call_args.args[0].endswith("/tracking/custom_application/resolve_litellm_application")
    assert mock_post.call_args.kwargs["json"] == {"application_name": "Weather App"}
    assert mock_get.call_args.args[0].endswith("/tracking/custom_application/litellm_app_name_regex")


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
