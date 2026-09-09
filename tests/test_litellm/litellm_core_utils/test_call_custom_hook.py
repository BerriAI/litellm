import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.call_custom_hook import call_custom_hook, call_custom_hook_sync
from litellm.litellm_core_utils.hook_filter_utils import parse_hook_filters


class _RecordingLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls.append(("async_pre_call_hook", data, call_type))
        return data

    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        self.calls.append(("async_moderation_hook", data, call_type))
        return None

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.calls.append(("log_success_event", kwargs))
        return "logged"


@pytest.fixture
def callback():
    return _RecordingLogger()


class TestCallCustomHookAsync:
    @pytest.mark.asyncio
    async def test_runs_when_no_hook_filters_attached_at_all(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        assert callback.hook_filters is None
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            model="gpt-4o",
            key_alias=None,
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={"model": "gpt-4o"},
            call_type="acompletion",
        )
        assert result == {"model": "gpt-4o"}
        assert callback.calls == [("async_pre_call_hook", {"model": "gpt-4o"}, "acompletion")]

    @pytest.mark.asyncio
    async def test_runs_when_this_hook_has_no_filter_entry_but_sibling_hook_does(self, monkeypatch, callback):
        """Per-hook-method granularity: a filter on async_moderation_hook must not
        affect async_pre_call_hook on the same instance."""
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters(
            "recording", {"async_moderation_hook": {"models": ["claude-*"]}}
        )
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            model="gpt-4o",
            key_alias=None,
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={"model": "gpt-4o"},
            call_type="acompletion",
        )
        assert result == {"model": "gpt-4o"}
        assert len(callback.calls) == 1

    @pytest.mark.asyncio
    async def test_skips_when_enable_hook_filters_true_and_models_filter_excludes(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters("recording", {"async_pre_call_hook": {"models": ["claude-*"]}})
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            model="gpt-4o",
            key_alias=None,
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={"model": "gpt-4o"},
            call_type="acompletion",
        )
        assert result is None
        assert callback.calls == []

    @pytest.mark.asyncio
    async def test_runs_when_enable_hook_filters_true_and_models_filter_matches(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters("recording", {"async_pre_call_hook": {"models": ["gpt-4o*"]}})
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            model="gpt-4o",
            key_alias=None,
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={"model": "gpt-4o"},
            call_type="acompletion",
        )
        assert result == {"model": "gpt-4o"}
        assert len(callback.calls) == 1

    @pytest.mark.asyncio
    async def test_skips_on_key_aliases_filter_mismatch(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters(
            "recording", {"async_pre_call_hook": {"key_aliases": ["prod-*"]}}
        )
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            model=None,
            key_alias="dev-key-1",
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={},
            call_type="acompletion",
        )
        assert result is None
        assert callback.calls == []

    @pytest.mark.asyncio
    async def test_runs_on_request_tags_filter_match(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters(
            "recording", {"async_pre_call_hook": {"request_tags": ["scan-me"]}}
        )
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            model=None,
            key_alias=None,
            model_tags=(),
            request_tags=("unrelated", "scan-me"),
            user_api_key_dict=None,
            cache=None,
            data={},
            call_type="acompletion",
        )
        assert result == {}
        assert len(callback.calls) == 1

    @pytest.mark.asyncio
    async def test_ignores_hook_filters_entirely_when_enable_hook_filters_false(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", False)
        callback.hook_filters = parse_hook_filters("recording", {"async_pre_call_hook": {"models": ["claude-*"]}})
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            model="gpt-4o",
            key_alias=None,
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={"model": "gpt-4o"},
            call_type="acompletion",
        )
        assert result == {"model": "gpt-4o"}
        assert len(callback.calls) == 1

    @pytest.mark.asyncio
    async def test_target_is_invoked_but_filters_are_checked_against_callback(self, monkeypatch, callback):
        """Covers ProxyLogging's unified-guardrail wrapper: the config-owning
        object (callback) and the object actually invoked (target) can differ."""
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters("recording", {"async_pre_call_hook": {"models": ["claude-*"]}})
        wrapper = _RecordingLogger()

        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            target=wrapper,
            model="gpt-4o",
            key_alias=None,
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={"model": "gpt-4o"},
            call_type="acompletion",
        )
        assert result is None
        assert callback.calls == []
        assert wrapper.calls == []

        callback.hook_filters = parse_hook_filters("recording", {"async_pre_call_hook": {"models": ["gpt-4o*"]}})
        result = await call_custom_hook(
            callback,
            "async_pre_call_hook",
            target=wrapper,
            model="gpt-4o",
            key_alias=None,
            model_tags=(),
            request_tags=(),
            user_api_key_dict=None,
            cache=None,
            data={"model": "gpt-4o"},
            call_type="acompletion",
        )
        assert result == {"model": "gpt-4o"}
        assert callback.calls == []
        assert len(wrapper.calls) == 1

    @pytest.mark.asyncio
    async def test_forwards_hook_kwargs_verbatim_to_the_named_hook(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", False)
        sentinel_data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        await call_custom_hook(
            callback,
            "async_pre_call_hook",
            user_api_key_dict="fake-key-dict",
            cache="fake-cache",
            data=sentinel_data,
            call_type="acompletion",
        )
        assert callback.calls == [("async_pre_call_hook", sentinel_data, "acompletion")]


class TestCallCustomHookSync:
    def test_runs_when_filter_matches(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters("recording", {"log_success_event": {"key_aliases": ["prod-*"]}})
        result = call_custom_hook_sync(
            callback,
            "log_success_event",
            model=None,
            key_alias="prod-key-1",
            model_tags=(),
            request_tags=(),
            kwargs={},
            response_obj=None,
            start_time=0,
            end_time=1,
        )
        assert result == "logged"
        assert len(callback.calls) == 1

    def test_skips_when_filter_excludes(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", True)
        callback.hook_filters = parse_hook_filters("recording", {"log_success_event": {"key_aliases": ["prod-*"]}})
        result = call_custom_hook_sync(
            callback,
            "log_success_event",
            model=None,
            key_alias="dev-key-1",
            model_tags=(),
            request_tags=(),
            kwargs={},
            response_obj=None,
            start_time=0,
            end_time=1,
        )
        assert result is None
        assert callback.calls == []

    def test_ignores_hook_filters_when_enable_hook_filters_false(self, monkeypatch, callback):
        monkeypatch.setattr("litellm.enable_hook_filters", False)
        callback.hook_filters = parse_hook_filters("recording", {"log_success_event": {"key_aliases": ["prod-*"]}})
        result = call_custom_hook_sync(
            callback,
            "log_success_event",
            model=None,
            key_alias="dev-key-1",
            model_tags=(),
            request_tags=(),
            kwargs={},
            response_obj=None,
            start_time=0,
            end_time=1,
        )
        assert result == "logged"
        assert len(callback.calls) == 1
