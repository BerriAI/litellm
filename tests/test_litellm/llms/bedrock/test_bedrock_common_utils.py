import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.bedrock.common_utils import BedrockModelInfo

# --------------------------------------------------------------------------- #
# get_bedrock_response_stream_shape lazy-load tests                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_bedrock_response_stream_shape_cache():
    """Prevent lru_cache leakage between tests in this module."""
    import litellm.llms.bedrock.common_utils as mod

    mod.get_bedrock_response_stream_shape.cache_clear()
    mod._get_local_model_cost_map.cache_clear()
    yield
    mod.get_bedrock_response_stream_shape.cache_clear()
    mod._get_local_model_cost_map.cache_clear()


def test_bedrock_response_stream_shape_lazy_loads_once():
    """
    get_bedrock_response_stream_shape() loads from botocore at most once per process.
    """
    from unittest.mock import MagicMock, patch

    import litellm.llms.bedrock.common_utils as mod

    sentinel = MagicMock()
    with patch.object(
        mod, "_load_bedrock_response_stream_shape", return_value=sentinel
    ) as mock_load:
        assert mod.get_bedrock_response_stream_shape() is sentinel
        assert mod.get_bedrock_response_stream_shape() is sentinel
        mock_load.assert_called_once()


def test_bedrock_response_stream_shape_loaded_on_first_access():
    """
    get_bedrock_response_stream_shape() loads once on first use.
    In a standard environment with botocore installed it must be non-None.
    """
    pytest.importorskip("botocore")
    from litellm.llms.bedrock.common_utils import get_bedrock_response_stream_shape

    assert get_bedrock_response_stream_shape() is not None


def test_bedrock_response_stream_shape_load_failure_returns_none():
    """
    If botocore's Loader raises (e.g. missing data files), _load_bedrock_response_stream_shape
    should return None rather than propagating the exception, so the module
    still imports cleanly.
    """
    from unittest.mock import patch

    import litellm.llms.bedrock.common_utils as mod

    pytest.importorskip("botocore")
    with patch(
        "botocore.loaders.Loader.load_service_model",
        side_effect=Exception("no data"),
    ):
        shape = mod._load_bedrock_response_stream_shape()
        assert shape is None


def test_bedrock_response_stream_shape_is_structure_shape():
    """
    The loaded shape should be the botocore StructureShape for ResponseStream,
    not a plain dict or any other type.
    """
    pytest.importorskip("botocore")
    from botocore.model import StructureShape

    from litellm.llms.bedrock.common_utils import get_bedrock_response_stream_shape

    loaded_shape = get_bedrock_response_stream_shape()
    assert (
        loaded_shape is not None
    ), "get_bedrock_response_stream_shape() is None — botocore may not be installed"
    shape: StructureShape = loaded_shape
    assert isinstance(shape, StructureShape)
    assert shape.name == "ResponseStream"


def test_bedrock_response_stream_shape_same_object_across_calls():
    """
    Repeated calls must return the identical cached object.
    """
    from litellm.llms.bedrock.common_utils import get_bedrock_response_stream_shape

    first = get_bedrock_response_stream_shape()
    second = get_bedrock_response_stream_shape()
    assert first is second


def test_bedrock_event_stream_decoder_base_uses_module_shape():
    """
    BedrockEventStreamDecoderBase instances no longer carry their own
    per-instance cache — _parse_message_from_event uses the module constant
    directly, so there is no instance-level _response_stream_shape_cache attr.
    """
    from litellm.llms.bedrock.common_utils import BedrockEventStreamDecoderBase

    decoder_a = BedrockEventStreamDecoderBase()
    decoder_b = BedrockEventStreamDecoderBase()

    assert "_response_stream_shape_cache" not in decoder_a.__dict__
    assert "_response_stream_shape_cache" not in decoder_b.__dict__


def test_bedrock_parse_message_from_event_raises_on_none_shape():
    """
    When get_bedrock_response_stream_shape() returns None (botocore unavailable),
    _parse_message_from_event must raise BedrockError before touching the
    botocore parser — not an opaque AttributeError from inside botocore.
    """
    from unittest.mock import MagicMock, patch

    import litellm.llms.bedrock.common_utils as mod
    from litellm.llms.bedrock.common_utils import (
        BedrockError,
        BedrockEventStreamDecoderBase,
    )

    decoder = BedrockEventStreamDecoderBase.__new__(BedrockEventStreamDecoderBase)
    decoder.parser = MagicMock()
    mock_event = MagicMock()

    with patch.object(mod, "get_bedrock_response_stream_shape", return_value=None):
        with pytest.raises(BedrockError) as exc_info:
            decoder._parse_message_from_event(mock_event)

    assert exc_info.value.status_code == 500
    assert "botocore" in str(exc_info.value.message).lower()
    # The botocore parser must never have been called
    mock_event.to_response_dict.assert_not_called()


def test_deepseek_cris():
    """
    Test that DeepSeek models with cross-region inference prefix use converse route
    """
    bedrock_model_info = BedrockModelInfo
    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/us.deepseek.r1-v1:0"
    )
    assert bedrock_route == "converse"


def test_application_inference_profile_arn_routes_to_converse():
    """
    Regression for #18258: a bare application-inference-profile ARN passed as
    `bedrock/arn:...` must route to converse. The ARN ends in an opaque id with
    no provider substring, so the invoke path cannot build a provider-native
    body and raises "Unknown provider=None". Converse needs no provider, so it
    is the correct route.
    """
    route = BedrockModelInfo.get_bedrock_route(
        model="bedrock/arn:aws:bedrock:us-west-2:123412341234:application-inference-profile/a1b2c3"
    )
    assert route == "converse"


def test_explicit_invoke_prefix_wins_over_application_inference_profile_arn():
    """
    An explicit invoke/ prefix is respected even for an application-inference-profile
    ARN; only the bare `bedrock/arn:...` form is auto-routed to converse. The
    explicit invoke path remains a dead end for these ARNs (no provider can be
    derived, so completion raises "Unknown provider=None") by design: a caller
    that explicitly asks for invoke gets invoke. The auto-route only rescues the
    documented bare form.
    """
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    model = "bedrock/invoke/arn:aws:bedrock:us-west-2:123412341234:application-inference-profile/a1b2c3"
    assert BedrockModelInfo.get_bedrock_route(model) == "invoke"
    assert BaseAWSLLM.get_bedrock_invoke_provider(model) is None


def test_system_defined_inference_profile_arn_still_routes_to_converse():
    """
    A system-defined cross-region inference-profile ARN embeds a known model, so
    get_base_model resolves it and it already routes to converse. Guards that the
    application-inference-profile fix does not change this working case.
    """
    route = BedrockModelInfo.get_bedrock_route(
        model="bedrock/arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    assert route == "converse"


def test_other_opaque_arn_types_still_route_to_invoke():
    """
    Only application-inference-profile ARNs are auto-routed to converse. Other
    opaque ARNs (provisioned-model, imported-model, custom-model-deployment)
    also yield no invoke provider, but they are frequently invoke-only with
    provider-specific body formats, so routing them to converse could break
    them. Guards the deliberate scope against an over-broad "any opaque ARN ->
    converse" generalization.
    """
    for arn_segment in (
        "provisioned-model/abcdefgh1234",
        "imported-model/abcdefgh1234",
        "custom-model-deployment/abcdefgh1234",
    ):
        route = BedrockModelInfo.get_bedrock_route(
            model=f"bedrock/arn:aws:bedrock:us-east-1:123412341234:{arn_segment}"
        )
        assert route == "invoke", f"{arn_segment} should stay on invoke route"


def test_govcloud_cross_region_inference_prefix():
    """
    Test that GovCloud models with cross-region inference prefix (us-gov.) are parsed correctly
    """
    bedrock_model_info = BedrockModelInfo

    # Test us-gov prefix is stripped correctly for Claude models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert base_model == "anthropic.claude-haiku-4-5-20251001-v1:0"

    # Test us-gov prefix is stripped correctly for different Claude versions
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    assert base_model == "anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Test us-gov prefix is stripped correctly for Haiku models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.anthropic.claude-3-haiku-20240307-v1:0"
    )
    assert base_model == "anthropic.claude-3-haiku-20240307-v1:0"

    # Test us-gov prefix is stripped correctly for Meta models
    base_model = bedrock_model_info.get_base_model(
        model="bedrock/us-gov.meta.llama3-8b-instruct-v1:0"
    )
    assert base_model == "meta.llama3-8b-instruct-v1:0"


def test_context_window_suffix_stripped_for_cost_lookup():
    """
    Test that [1m], [200k] etc. context window suffixes are stripped from
    Bedrock model names before cost lookup.

    Models configured like `bedrock/us.anthropic.claude-opus-4-6-v1[1m]`
    should resolve to the base model name so pricing can be found.
    """
    from litellm.llms.bedrock.common_utils import get_bedrock_base_model

    assert (
        get_bedrock_base_model("us.anthropic.claude-opus-4-6-v1[1m]")
        == "anthropic.claude-opus-4-6-v1"
    )
    assert (
        get_bedrock_base_model("us.anthropic.claude-sonnet-4-6[1m]")
        == "anthropic.claude-sonnet-4-6"
    )
    assert (
        get_bedrock_base_model("global.anthropic.claude-opus-4-5-20251101-v1:0[1m]")
        == "anthropic.claude-opus-4-5-20251101-v1:0"
    )
    # Ensure models without suffix are unaffected
    assert (
        get_bedrock_base_model("us.anthropic.claude-opus-4-6-v1")
        == "anthropic.claude-opus-4-6-v1"
    )
    # Ensure :51k throughput suffix still works
    assert (
        get_bedrock_base_model("anthropic.claude-3-5-sonnet-20241022-v2:0:51k")
        == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )


def test_output_config_effort_normalization_uses_model_info_ceiling(monkeypatch):
    import litellm.llms.bedrock.common_utils as mod

    calls = []

    def fake_get_model_info(model, custom_llm_provider=None):
        calls.append((model, custom_llm_provider))
        return {"bedrock_output_config_effort_ceiling": "max"}

    monkeypatch.setattr(mod, "_get_model_info", fake_get_model_info)
    output_config = {"effort": "xhigh"}

    mod.normalize_bedrock_opus_output_config_effort(
        model="custom-bedrock-alias-without-opus-pattern",
        output_config=output_config,
    )

    assert output_config == {"effort": "max"}
    assert calls == [("custom-bedrock-alias-without-opus-pattern", "bedrock")]


@pytest.mark.parametrize(
    "model,expected_ceiling",
    [
        ("anthropic.claude-opus-4-5-20251101-v1:0", "high"),
        ("anthropic.claude-opus-4-6-v1", "max"),
        ("anthropic.claude-opus-4-7", "xhigh"),
        ("us.anthropic.claude-opus-4-5-20251101-v1:0", "high"),
        ("us.anthropic.claude-opus-4-6-v1", "max"),
        ("us.anthropic.claude-opus-4-7", "xhigh"),
    ],
)
def test_bundled_bedrock_opus_model_info_declares_output_config_effort_ceiling(
    model, expected_ceiling
):
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    model_info = GetModelCostMap.load_local_model_cost_map()[model]

    assert model_info["bedrock_output_config_effort_ceiling"] == expected_ceiling


def test_route_prefix_matched_as_path_segment_not_substring():
    """Route tokens like ``mantle/`` must match only at a path-segment boundary.

    The ``bedrock_mantle/`` provider prefix contains the substring ``mantle/``;
    a substring match misroutes ``bedrock_mantle/openai.gpt-5.5`` to the Claude
    Mythos mantle config, whose request transform strips ``mantle/`` and mangles
    the body model into ``bedrock_openai.gpt-5.5``. These assertions fail under
    the old substring matching and pass once matching is anchored to ``startswith``
    or a ``/`` boundary.
    """
    # The bedrock_mantle/ provider prefix must NOT be read as the mantle/ route.
    assert (
        BedrockModelInfo.get_bedrock_route("bedrock_mantle/openai.gpt-5.5") != "mantle"
    )
    assert (
        BedrockModelInfo.get_bedrock_route("bedrock_mantle/openai.gpt-5.4") == "invoke"
    )
    assert (
        BedrockModelInfo._explicit_mantle_route("bedrock_mantle/openai.gpt-5.5")
        is False
    )

    # A genuine mantle route still resolves, via the startswith branch...
    assert (
        BedrockModelInfo.get_bedrock_route("mantle/anthropic.claude-mythos-preview")
        == "mantle"
    )
    # ...and via the mid-path "/mantle/" branch (after the bedrock/ provider prefix).
    assert (
        BedrockModelInfo.get_bedrock_route(
            "bedrock/mantle/anthropic.claude-mythos-preview"
        )
        == "mantle"
    )


def test_model_has_route_prefix_exercises_both_branches():
    """``_model_has_route_prefix`` matches on ``startswith`` or a ``/`` boundary only."""
    # startswith branch
    assert (
        BedrockModelInfo._model_has_route_prefix(
            "mantle/anthropic.claude-mythos-preview", "mantle/"
        )
        is True
    )
    # f"/{prefix}" boundary branch
    assert (
        BedrockModelInfo._model_has_route_prefix(
            "bedrock/mantle/anthropic.claude-mythos-preview", "mantle/"
        )
        is True
    )
    # neither branch: the token only appears glued to another segment
    assert (
        BedrockModelInfo._model_has_route_prefix(
            "bedrock_mantle/openai.gpt-5.5", "mantle/"
        )
        is False
    )


@pytest.mark.parametrize(
    "route_method, token",
    [
        (BedrockModelInfo._explicit_converse_route, "converse"),
        (BedrockModelInfo._explicit_converse_like_route, "converse_like"),
        (BedrockModelInfo._explicit_invoke_route, "invoke"),
        (BedrockModelInfo._explicit_async_invoke_route, "async_invoke"),
        (BedrockModelInfo._explicit_agent_route, "agent"),
        (BedrockModelInfo._explicit_agentcore_route, "agentcore"),
        (BedrockModelInfo._explicit_claude_platform_route, "claude_platform"),
        (BedrockModelInfo._explicit_openai_route, "openai"),
    ],
    ids=[
        "converse",
        "converse_like",
        "invoke",
        "async_invoke",
        "agent",
        "agentcore",
        "claude_platform",
        "openai",
    ],
)
def test_explicit_route_helpers_match_token_only_as_path_segment(route_method, token):
    """Each migrated ``_explicit_*_route`` matches its token only as a path segment.

    A leading segment (start of the id or right after a ``/``) matches; the token
    glued onto a preceding segment does not. Reverting any method to the old
    ``"<token>/" in model`` substring check makes the non-segment case return True
    and fails this test.
    """
    # leading-segment forms match
    assert route_method(f"{token}/some-model") is True
    assert route_method(f"bedrock/{token}/some-model") is True
    # the token only as a non-segment substring must not match
    assert route_method(f"x{token}/y") is False


def test_explicit_invoke_route_does_not_match_async_invoke():
    """``invoke/`` must not substring-match ``async_invoke/`` models.

    This is the concrete improvement of the segment-boundary migration: the old
    ``"invoke/" in model`` check wrongly classified async-invoke models as the
    invoke route.
    """
    async_invoke_model = "async_invoke/twelvelabs.marengo-embed-2-7-v1:0"
    assert BedrockModelInfo._explicit_invoke_route(async_invoke_model) is False
    assert (
        BedrockModelInfo._explicit_invoke_route(f"bedrock/{async_invoke_model}")
        is False
    )
    # ...while async_invoke/ is still detected as its own route.
    assert BedrockModelInfo._explicit_async_invoke_route(async_invoke_model) is True
    assert (
        BedrockModelInfo._explicit_async_invoke_route(f"bedrock/{async_invoke_model}")
        is True
    )
