"""Tests for the Sensitive Data Routing guardrail."""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.sensitive_data_routing import (
    SensitiveDataRoutingGuardrail,
    initialize_guardrail,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.sensitive_data_routing import (
    SensitiveDataRoutingConfigModel,
)

ON_PREM = "on-prem-model"
USER_KEY = UserAPIKeyAuth()


class RecordingCache(DualCache):
    """DualCache that records writes so tests can assert pinning behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.sets: List[Tuple[str, Any, Optional[int]]] = []

    async def async_set_cache(self, key, value, local_only: bool = False, **kwargs):
        self.sets.append((key, value, kwargs.get("ttl")))
        await super().async_set_cache(key, value, local_only=local_only, **kwargs)


def _request(text, model="gpt-4o", **extra) -> Dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


def _make_guardrail(**overrides) -> SensitiveDataRoutingGuardrail:
    params: Dict[str, Any] = dict(
        guardrail_name="sdr",
        on_premise_model=ON_PREM,
        prebuilt_patterns=["us_ssn"],
        keywords=["confidential"],
    )
    params.update(overrides)
    return SensitiveDataRoutingGuardrail(**params)


async def _hook(guardrail, data, cache=None):
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=USER_KEY,
        cache=cache or DualCache(),
        data=data,
        call_type="acompletion",
    )


@pytest.mark.asyncio
async def test_clean_prompt_is_not_rerouted():
    g = _make_guardrail()
    data = _request("what's the weather today?")
    result = await _hook(g, data)
    assert result is None
    assert data["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_prebuilt_pattern_match_reroutes_to_on_prem():
    g = _make_guardrail()
    data = _request("my social security number is 123-45-6789")
    result = await _hook(g, data)
    assert result is not None
    assert result["model"] == ON_PREM
    assert data["model"] == ON_PREM


@pytest.mark.asyncio
async def test_custom_regex_match_reroutes():
    g = _make_guardrail(prebuilt_patterns=None, regex_patterns=[r"project\s+titan"])
    data = _request("notes on Project Titan rollout")
    result = await _hook(g, data)
    assert result is not None and result["model"] == ON_PREM


@pytest.mark.asyncio
async def test_keyword_match_is_case_insensitive():
    g = _make_guardrail(prebuilt_patterns=None, keywords=["confidential"])
    data = _request("this memo is CONFIDENTIAL")
    result = await _hook(g, data)
    assert result is not None and result["model"] == ON_PREM


@pytest.mark.asyncio
async def test_non_dict_messages_are_skipped():
    g = _make_guardrail()
    data = {
        "model": "gpt-4o",
        "messages": ["not-a-dict", {"role": "user", "content": "ssn 123-45-6789"}],
    }
    result = await _hook(g, data)
    assert result is not None and result["model"] == ON_PREM


@pytest.mark.asyncio
async def test_detects_text_in_content_parts_list():
    g = _make_guardrail()
    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ssn 123-45-6789"}],
            }
        ],
    }
    result = await _hook(g, data)
    assert result is not None and result["model"] == ON_PREM


@pytest.mark.asyncio
async def test_reroute_records_guardrail_logging_information():
    g = _make_guardrail()
    data = _request("ssn 123-45-6789")
    await _hook(g, data)
    entries = data["metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    assert entries[0]["guardrail_name"] == "sdr"


@pytest.mark.asyncio
async def test_sticky_session_pins_following_clean_turns():
    g = _make_guardrail(sticky_session=True, session_ttl_seconds=999)
    cache = RecordingCache()
    session = {"litellm_session_id": "sess-1"}

    first = await _hook(g, _request("ssn 123-45-6789", **session), cache)
    assert first is not None and first["model"] == ON_PREM
    assert cache.sets and cache.sets[0][0].endswith("sess-1")
    assert cache.sets[0][2] == 999  # ttl is honored

    follow_up = _request("just a normal follow-up question", **session)
    result = await _hook(g, follow_up, cache)
    assert result is not None
    assert follow_up["model"] == ON_PREM


@pytest.mark.asyncio
async def test_non_sticky_does_not_pin_session():
    g = _make_guardrail(sticky_session=False)
    cache = RecordingCache()
    session = {"litellm_session_id": "sess-2"}

    await _hook(g, _request("ssn 123-45-6789", **session), cache)
    assert cache.sets == []  # nothing pinned

    follow_up = _request("a normal follow-up", **session)
    result = await _hook(g, follow_up, cache)
    assert result is None
    assert follow_up["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_sessions_are_isolated_from_each_other():
    g = _make_guardrail(sticky_session=True)
    cache = RecordingCache()

    await _hook(g, _request("ssn 123-45-6789", litellm_session_id="flagged"), cache)

    other = _request("nothing sensitive here", litellm_session_id="other")
    result = await _hook(g, other, cache)
    assert result is None
    assert other["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_session_id_read_from_metadata():
    g = _make_guardrail(sticky_session=True)
    cache = RecordingCache()
    meta = {"metadata": {"session_id": "meta-sess"}}

    await _hook(g, _request("ssn 123-45-6789", **meta), cache)
    follow_up = _request("benign follow-up", **meta)
    result = await _hook(g, follow_up, cache)
    assert result is not None and follow_up["model"] == ON_PREM


def test_requires_at_least_one_detector():
    with pytest.raises(ValueError):
        SensitiveDataRoutingGuardrail(guardrail_name="sdr", on_premise_model=ON_PREM)


def test_only_supports_pre_call_event_hook():
    with pytest.raises(ValueError):
        _make_guardrail(event_hook=GuardrailEventHooks.post_call)


def test_initializer_requires_guardrail_name():
    params = LitellmParams(
        guardrail="sensitive_data_routing", mode="pre_call", on_premise_model=ON_PREM
    )
    with pytest.raises(ValueError):
        initialize_guardrail(
            litellm_params=params,
            guardrail={"guardrail_name": "", "litellm_params": {}},
        )


def test_initializer_requires_on_premise_model():
    params = LitellmParams(guardrail="sensitive_data_routing", mode="pre_call")
    with pytest.raises(ValueError):
        initialize_guardrail(
            litellm_params=params,
            guardrail={"guardrail_name": "sdr", "litellm_params": {}},
        )


def test_config_model_is_exposed_for_ui():
    config_model = SensitiveDataRoutingGuardrail.get_config_model()
    assert config_model is SensitiveDataRoutingConfigModel
    assert config_model.ui_friendly_name() == "Sensitive Data Routing"


def test_initializer_builds_guardrail_from_config():
    params = LitellmParams(
        guardrail="sensitive_data_routing",
        mode="pre_call",
        on_premise_model=ON_PREM,
        prebuilt_patterns=["us_ssn"],
        keywords=["confidential"],
        session_ttl_seconds=120,
    )
    instance = initialize_guardrail(
        litellm_params=params,
        guardrail={"guardrail_name": "sdr", "litellm_params": {}},
    )
    assert isinstance(instance, SensitiveDataRoutingGuardrail)
    assert instance.on_premise_model == ON_PREM
    assert instance.session_ttl_seconds == 120
