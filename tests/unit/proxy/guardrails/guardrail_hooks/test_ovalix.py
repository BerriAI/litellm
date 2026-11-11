"""
Unit tests for Ovalix guardrail: config resolution and apply_guardrail behavior
with mocked Tracker service responses (allow, anonymize, block).
"""

import base64
import gzip
import json as json_lib
import os
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix import (
    OvalixGuardrail,
    OvalixGuardrailBlockedException,
    OvalixGuardrailMissingSecrets,
)
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix_extraction import FilePart
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
            assert guardrail._get_actor({}) == "unknown"
            assert (
                guardrail._get_actor({"metadata": {"user_api_key_user_email": "", "user_api_key_user_id": ""}})
                == "unknown"
            )
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


def _checkpoint_calls(mock_post):
    """(body, url) of every tracker call the guardrail made."""
    return [(c.kwargs["json"], c.args[0]) for c in mock_post.call_args_list]


def _route_of(url):
    return url.rsplit("/", 1)[-1]


def _mock_tracker(g):
    post_resp = MagicMock()
    post_resp.json.return_value = TRACKER_RESPONSE_ALLOW
    post_resp.raise_for_status = MagicMock()
    g._async_handler.get = AsyncMock()
    g._async_handler.post = AsyncMock(return_value=post_resp)
    return g._async_handler.get, g._async_handler.post


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

    async def _post(url, headers=None, json=None, timeout=None):
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

    async def _post(url, headers=None, json=None, timeout=None):
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

    async def _post(url, headers=None, json=None, timeout=None):
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

    async def _post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g._call_checkpoint("FILE", {"name": "f.txt", "content": "x"}, "file-1", "a", "s")
    assert seen["url"] == "https://t/tracking/beta/file_checkpoint"


@pytest.mark.asyncio
@pytest.mark.parametrize("oversize", [True, False])
async def test_file_checkpoint_flags_oversize_parts(oversize):
    g = _static_guardrail()
    part = FilePart(name="huge.bin", data=None, mime_hint=None, inline=oversize, oversize=oversize, message_index=0)
    seen = {}

    async def _post(url, headers=None, json=None, timeout=None):
        seen["last"] = json
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g._check_files_for_block([part], "file-1", "a", "s")
    assert seen["last"]["data"]["content"] is None
    assert seen["last"].get("oversize", False) is oversize


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
async def test_texts_are_checked_newest_first_and_a_blocked_newest_stops_the_scan():
    g: Final = _static_guardrail()
    post, calls = _recording_post(lambda body: _BLOCK if body["data"].get("content") == "new" else _ALLOW)

    with patch.object(g._async_handler, "post", new=post):
        with pytest.raises(OvalixGuardrailBlockedException):
            await g.apply_guardrail(
                inputs=GenericGuardrailAPIInputs(texts=["old", "middle", "new"]),
                request_data={},
                input_type="request",
                logging_obj=None,
            )

    assert [content for data_type, content in calls if data_type == "TEXT"] == ["new"]


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

    async def _post(url, headers=None, json=None, timeout=None):
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
@pytest.mark.parametrize(
    "request_data",
    [
        {},
        {"metadata": {"user_id": "client-chosen"}, "litellm_metadata": {"user_api_key_user_id": "uid-1"}},
    ],
)
async def test_request_with_no_user_field_sends_unknown_actor(request_data):
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(texts=["hi"])
    seen = {}

    async def _post(url, headers=None, json=None, timeout=None):
        seen["last"] = json
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data=request_data, input_type="request", logging_obj=None)
    assert seen["last"]["actor"] == "unknown"


def _recording_post(mapping_fn=lambda body: _ALLOW):
    calls = []

    async def _post(url, headers=None, json=None, timeout=None):
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

    async def _post(url, headers=None, json=None, timeout=None):
        encoded.append(json_lib.dumps(json))
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    payloads = [json_lib.loads(body) for body in encoded]
    assert sorted(p["data_type"] for p in payloads) == ["FILE", "TEXT", "TEXT", "TOOL", "TOOL", "TOOL"]
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


@pytest.mark.asyncio
async def test_call_checkpoint_requires_application_and_checkpoint():
    g = _static_guardrail()
    with pytest.raises(ValueError, match="application_id or checkpoint_id not resolved"):
        await g._call_checkpoint("TEXT", {"content": "x"}, "", "actor", "sess")


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


_GIF_B64 = base64.b64encode(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00;").decode()
_CORRUPT_GIF_B64 = f"{_GIF_B64[:8]}!{_GIF_B64[8:]}"


def _user_turn_with(block):
    return GenericGuardrailAPIInputs(
        texts=["what is this?"],
        structured_messages=[{"role": "user", "content": [{"type": "text", "text": "what is this?"}, block]}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inputs", "input_type"),
    [
        (
            _user_turn_with({"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{_CORRUPT_GIF_B64}"}}),
            "request",
        ),
        (_user_turn_with({"type": "file", "file": {"filename": "a.gif", "file_data": _CORRUPT_GIF_B64}}), "request"),
        (_user_turn_with({"type": "input_file", "filename": "a.gif", "file_data": _CORRUPT_GIF_B64}), "request"),
        (
            _user_turn_with({"type": "input_audio", "input_audio": {"data": _CORRUPT_GIF_B64, "format": "wav"}}),
            "request",
        ),
        (GenericGuardrailAPIInputs(texts=[], images=[f"data:image/gif;base64,{_CORRUPT_GIF_B64}"]), "response"),
    ],
)
async def test_inline_content_that_cannot_be_decoded_blocks_instead_of_skipping_inspection(inputs, input_type):
    g = _static_guardrail()
    with patch.object(g._async_handler, "post", new=_post_returning(lambda body: _ALLOW)):
        with pytest.raises(OvalixGuardrailBlockedException) as exc:
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type=input_type, logging_obj=None)
    assert "couldn't be decoded" in str(exc.value.message)


def _pinned_guardrail(**overrides):
    return OvalixGuardrail(
        **{
            "tracker_api_base": "https://tracker.test",
            "tracker_api_key": "key",
            "application_id": "app-1",
            "guardrail_name": "ovalix-test",
            "event_hook": "pre_call",
            "default_on": True,
            **overrides,
        }
    )


def test_missing_application_id_fails_at_startup():
    with pytest.raises(OvalixGuardrailMissingSecrets, match="Application ID"):
        _pinned_guardrail(application_id=None, pre_checkpoint_id="pre-1")


@pytest.mark.parametrize(
    "hook, checkpoints, missing",
    [
        (GuardrailEventHooks.pre_call, {"post_checkpoint_id": "post-1"}, "Pre-checkpoint ID"),
        (GuardrailEventHooks.post_call, {"pre_checkpoint_id": "pre-1"}, "Post-checkpoint ID"),
    ],
)
def test_requested_hook_without_its_checkpoint_fails_at_startup(hook, checkpoints, missing):
    with pytest.raises(OvalixGuardrailMissingSecrets, match=missing):
        _pinned_guardrail(supported_event_hooks=[hook], **checkpoints)


@pytest.mark.asyncio
async def test_key_alias_does_not_change_the_application_checked():
    """The application comes from config alone, so a key alias naming another app changes nothing on the wire."""
    g: Final = _pinned_guardrail(pre_checkpoint_id="pre-1")
    mock_get, mock_post = _mock_tracker(g)

    await g.apply_guardrail(
        GenericGuardrailAPIInputs(texts=["hi"]),
        {"metadata": {"user_api_key_alias": "[Other App] prod", "user_api_key_user_email": "u@x.com"}},
        "request",
    )

    mock_get.assert_not_called()
    [(body, url)] = _checkpoint_calls(mock_post)
    assert url == "https://tracker.test/tracking/beta/checkpoint"
    assert (body["application_id"], body["checkpoint_id"]) == ("app-1", "pre-1")
    assert "application_name" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("input_type, inspected", [("request", ["pre-1"]), ("response", [])])
async def test_one_sided_config_inspects_configured_direction_only(input_type, inspected):
    g: Final = _pinned_guardrail(pre_checkpoint_id="pre-1")
    _, mock_post = _mock_tracker(g)

    result: Final = await g.apply_guardrail(
        inputs=GenericGuardrailAPIInputs(texts=["hi"]), request_data={}, input_type=input_type, logging_obj=None
    )

    assert result["texts"] == ["hi"]
    assert [body["checkpoint_id"] for body, _ in _checkpoint_calls(mock_post)] == inspected


@pytest.mark.asyncio
async def test_every_tracker_call_carries_the_configured_timeout():
    """litellm_params.timeout bounds the checkpoint call."""
    g: Final = _pinned_guardrail(pre_checkpoint_id="pre-1", timeout=2.5)
    mock_get, mock_post = _mock_tracker(g)

    await g.apply_guardrail(
        inputs=GenericGuardrailAPIInputs(texts=["hi"]), request_data={}, input_type="request", logging_obj=None
    )

    mock_get.assert_not_called()
    assert [(_route_of(c.args[0]), c.kwargs["timeout"]) for c in mock_post.call_args_list] == [("checkpoint", 2.5)]


@pytest.mark.asyncio
async def test_file_only_checkpoint_inspects_files_and_skips_text():
    """A direction with just a file checkpoint still scans files; text and tools need a prompt checkpoint."""
    g: Final = _pinned_guardrail(post_checkpoint_id="post-1", file_checkpoint_id="file-1", event_hook="post_call")
    _, mock_post = _mock_tracker(g)
    data_url: Final = "data:text/plain;base64," + base64.b64encode(b"secret").decode()
    inputs: Final = GenericGuardrailAPIInputs(
        texts=["hi"],
        structured_messages=[
            {"role": "user", "content": [{"type": "file", "file": {"filename": "s.txt", "file_data": data_url}}]}
        ],
    )

    result: Final = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)

    assert result["texts"] == ["hi"]
    assert [(b["data_type"], _route_of(url)) for b, url in _checkpoint_calls(mock_post)] == [
        ("FILE", "file_checkpoint")
    ]


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
        mode = "pre_call"
        default_on = True
        timeout = None

    guardrail = initialize_guardrail(_Params(), {"guardrail_name": "ovalix"})
    assert guardrail._file_checkpoint_id == "file-1"
    assert guardrail.guardrail_name == "ovalix"


def _capturing_post(response=None):
    seen = {}

    async def _post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["body"] = json
        r = MagicMock()
        r.json.return_value = response if response is not None else _ALLOW
        r.raise_for_status = MagicMock()
        return r

    return seen, _post


@pytest.mark.asyncio
async def test_call_checkpoint_sends_the_configured_ids_and_no_name_or_direction():
    g = _static_guardrail()
    seen, post = _capturing_post()

    with patch.object(g._async_handler, "post", new=post):
        await g._call_checkpoint("TEXT", {"content": "x"}, "pre-1", "actor", "sess")

    assert seen["body"]["application_id"] == "app-1"
    assert seen["body"]["checkpoint_id"] == "pre-1"
    assert "application_name" not in seen["body"]
    assert "input_type" not in seen["body"]


@pytest.mark.asyncio
async def test_message_tool_calls_are_scanned_without_top_level_tool_calls():
    g = _static_guardrail()
    inputs = GenericGuardrailAPIInputs(
        texts=[],
        structured_messages=[
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "exfil", "arguments": "{}"}}],
            }
        ],
    )
    with patch.object(
        g._async_handler, "post", new=_post_returning(lambda body: _BLOCK if body["data_type"] == "TOOL" else _ALLOW)
    ):
        with pytest.raises(OvalixGuardrailBlockedException):
            await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)


@pytest.mark.asyncio
async def test_tool_call_present_in_both_sources_is_scanned_once():
    g = _static_guardrail()
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "sf"}'}}
    inputs = GenericGuardrailAPIInputs(
        texts=[],
        tool_calls=[call],
        structured_messages=[{"role": "assistant", "tool_calls": [call]}],
    )
    tool_bodies = []

    async def _post(url, headers=None, json=None, timeout=None):
        if json["data_type"] == "TOOL":
            tool_bodies.append(json)
        r = MagicMock()
        r.json.return_value = _ALLOW
        r.raise_for_status = MagicMock()
        return r

    with patch.object(g._async_handler, "post", new=_post):
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert len(tool_bodies) == 1


def test_non_string_actor_metadata_does_not_crash_hashing():
    g = _static_guardrail()
    assert g._get_tracker_actor_id({"metadata": {"user_api_key_user_email": 12345}}) == g._get_tracker_actor_id({})


def test_non_string_corrected_content_is_ignored():
    g = _static_guardrail()
    assert g._get_trackers_corrected_message({"modified_data": {"content": {"nested": "obj"}}}) is None
    assert g._get_trackers_corrected_message({"modified_data": {"content": ""}}) == ""
