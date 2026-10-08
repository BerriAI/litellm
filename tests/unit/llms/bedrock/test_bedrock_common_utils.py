
import asyncio, importlib, litellm, litellm.litellm_core_utils.get_model_cost_map as bedrock_govcloud_model_cost_map, pytest


from litellm.llms.bedrock.common_utils import(
    AmazonBedrockGlobalConfig,
    BedrockModelInfo,
    extract_model_name_from_bedrock_arn,
    get_bedrock_base_model,
    get_bedrock_cross_region_inference_regions,
    strip_bedrock_routing_prefix,
    strip_bedrock_throughput_suffix,
)
from collections.abc import Iterator
from litellm import completion
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from unittest.mock import Mock, patch

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


def test_legacy_mantle_route_prefix_stripped_for_cost_lookup():
    """The mantle/ route token is a routing prefix like openai/, so a bedrock/mantle/<model>
    deployment must resolve the bare Bedrock model for cost lookup while still routing to Mantle."""
    from litellm.llms.bedrock.common_utils import get_bedrock_base_model, strip_bedrock_routing_prefix

    assert strip_bedrock_routing_prefix("mantle/anthropic.claude-sonnet-5") == "anthropic.claude-sonnet-5"
    assert get_bedrock_base_model("bedrock/mantle/anthropic.claude-sonnet-5") == "anthropic.claude-sonnet-5"
    assert BedrockModelInfo.get_bedrock_route("bedrock/mantle/anthropic.claude-sonnet-5") == "mantle"


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
        BedrockModelInfo.get_bedrock_route("bedrock_mantle/openai.gpt-5.4") == "converse"
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


def test_capability_lookups_fall_back_to_base_model_when_regional_entry_lacks_field(monkeypatch):
    """
    Regression test: a regional model_cost entry without the capability field
    must not shadow a base entry that has it (`get(model) or get(base)` used to
    short-circuit on the truthy regional dict and drop the capability).
    """
    import litellm
    from litellm.llms.bedrock.common_utils import (
        bedrock_converse_supports_parallel_tool_use_config,
        is_claude_4_5_on_bedrock,
    )

    base = "anthropic.claude-fallback-test"
    regional = f"eu.{base}"
    monkeypatch.setitem(litellm.model_cost, regional, {"input_cost_per_token": 1e-06})
    monkeypatch.setitem(
        litellm.model_cost,
        base,
        {
            "cache_creation_input_token_cost_above_1hr": 1e-05,
            "supports_parallel_tool_use_config": True,
        },
    )

    assert is_claude_4_5_on_bedrock(regional) is True
    assert bedrock_converse_supports_parallel_tool_use_config(regional) is True


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            {"supports_prompt_caching": True, "supports_prompt_cache_breakpoint": False},
            False,
            id="priced-cached-tokens-but-rejects-the-explicit-marker",
        ),
        pytest.param(
            {"supports_prompt_caching": False, "supports_prompt_cache_breakpoint": True},
            True,
            id="explicit-marker-flag-wins-over-the-caching-flag",
        ),
        pytest.param({"supports_prompt_caching": True}, True, id="caching-flag-alone-keeps-emitting"),
        pytest.param({"supports_prompt_caching": False}, False, id="no-caching-and-no-marker-flag"),
    ],
)
def test_bedrock_model_accepts_cache_points_prefers_the_explicit_breakpoint_flag(monkeypatch, entry, expected):
    import litellm
    from litellm.llms.bedrock.common_utils import bedrock_model_accepts_cache_points

    base = "vendor.breakpoint-flag-test"
    monkeypatch.setitem(litellm.model_cost, f"us.{base}", {"input_cost_per_token": 1e-06})
    monkeypatch.setitem(litellm.model_cost, base, entry)

    assert bedrock_model_accepts_cache_points(f"us.{base}") is expected


@pytest.mark.parametrize("model", ["moonshotai.kimi-k3", "us.moonshotai.kimi-k3", "global.moonshotai.kimi-k3"])
def test_kimi_k3_keeps_cached_token_pricing_while_refusing_converse_cache_points(model, local_model_cost_map):
    import litellm
    from litellm.llms.bedrock.common_utils import bedrock_model_accepts_cache_points

    assert bedrock_model_accepts_cache_points(model) is False
    assert litellm.utils.supports_prompt_caching(model=model, custom_llm_provider="bedrock") is True
    assert litellm.model_cost[model]["cache_read_input_token_cost"] > 0


def test_deployment_model_info_breakpoint_flag_covers_an_unmapped_arn(local_model_cost_map):
    from litellm import Router
    from litellm.llms.bedrock.common_utils import bedrock_model_accepts_cache_points

    flagged_arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/flagged"
    unflagged_arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/unflagged"
    converse_arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/converse"
    Router(
        model_list=[
            {
                "model_name": "kimi-k3-profile-converse",
                "litellm_params": {"model": f"bedrock/converse/{converse_arn}", "aws_region_name": "us-east-1"},
                "model_info": {"supports_prompt_cache_breakpoint": False},
            },
            {
                "model_name": "kimi-k3-profile",
                "litellm_params": {"model": f"bedrock/{flagged_arn}", "aws_region_name": "us-east-1"},
                "model_info": {"supports_prompt_cache_breakpoint": False},
            },
            {
                "model_name": "kimi-k3-profile-unflagged",
                "litellm_params": {"model": f"bedrock/{unflagged_arn}", "aws_region_name": "us-east-1"},
            },
        ]
    )

    assert bedrock_model_accepts_cache_points(flagged_arn) is False
    assert bedrock_model_accepts_cache_points(converse_arn) is False
    assert bedrock_model_accepts_cache_points(unflagged_arn) is True


def test_merge_bedrock_aws_request_params_strips_caller_identity_when_deployment_has_static_credentials():
    from litellm.llms.bedrock.common_utils import merge_bedrock_aws_request_params

    merged = merge_bedrock_aws_request_params(
        litellm_params={
            "aws_access_key_id": "deployment-key",
            "aws_secret_access_key": "deployment-secret",
            "aws_region_name": "us-west-2",
            "s3_bucket_name": "deployment-bucket",
        },
        optional_params={
            "aws_access_key_id": "caller-key",
            "aws_profile_name": "caller-profile",
            "aws_role_name": "arn:aws:iam::123456789012:role/caller",
            "aws_session_token": "caller-token",
            "aws_web_identity_token": "caller-web-identity",
            "aws_session_tags": [{"Key": "team", "Value": "caller-chosen"}],
            "timeout": 600,
        },
    )

    assert merged["aws_access_key_id"] == "deployment-key"
    assert merged["aws_secret_access_key"] == "deployment-secret"
    assert merged["aws_region_name"] == "us-west-2"
    assert merged["s3_bucket_name"] == "deployment-bucket"
    assert merged["timeout"] == 600
    for stripped in (
        "aws_profile_name",
        "aws_role_name",
        "aws_session_token",
        "aws_web_identity_token",
        "aws_session_tags",
    ):
        assert stripped not in merged


def test_merge_bedrock_aws_request_params_keeps_caller_credentials_without_static_deployment_credentials():
    from litellm.llms.bedrock.common_utils import merge_bedrock_aws_request_params

    merged = merge_bedrock_aws_request_params(
        litellm_params={"aws_region_name": "us-west-2"},
        optional_params={
            "aws_access_key_id": "caller-key",
            "aws_secret_access_key": "caller-secret",
            "aws_session_token": "caller-token",
        },
    )

    assert merged["aws_access_key_id"] == "caller-key"
    assert merged["aws_secret_access_key"] == "caller-secret"
    assert merged["aws_session_token"] == "caller-token"
    assert merged["aws_region_name"] == "us-west-2"


def test_strip_unsupported_output_config_keeps_format_drops_effort(local_model_cost_map):
    """On a model with neither effort flag, only the ``format`` key survives."""
    from litellm.llms.bedrock.common_utils import (
        strip_unsupported_bedrock_invoke_output_config_keys,
    )

    schema_format = {"type": "json_schema", "schema": {"type": "object"}}
    body = {"output_config": {"effort": "high", "format": schema_format}}

    strip_unsupported_bedrock_invoke_output_config_keys(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        request_body=body,
    )

    assert body["output_config"] == {"format": schema_format}


def test_apply_structured_output_prefers_legacy_output_format(local_model_cost_map):
    """The legacy ``output_format`` wins over ``output_config.format`` when a
    request carries both, matching the pre-existing precedence."""
    from litellm.llms.bedrock.common_utils import (
        apply_bedrock_invoke_structured_output,
    )

    legacy = {"type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}
    newer = {"type": "json_schema", "schema": {"type": "object", "properties": {"b": {"type": "string"}}}}
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "output_format": legacy,
        "output_config": {"format": newer},
    }

    apply_bedrock_invoke_structured_output(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        request_body=body,
    )

    assert body["output_config"] == {"format": legacy}
    assert "output_format" not in body


def test_sign_aws_request_assumes_role_with_external_id(monkeypatch):
    """A trust policy requiring sts:ExternalId must be satisfied when signing batch API requests."""
    import datetime
    from unittest.mock import patch

    import boto3
    from botocore.exceptions import ClientError

    from litellm.llms.bedrock.common_utils import CommonBatchFilesUtils

    monkeypatch.delenv("AWS_EXTERNAL_ID", raising=False)

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if params.get("ExternalId") != "external-id-batch-sign":
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:AssumeRole"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIABATCHSIGNROLE",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
                }
            }

    optional_params = {
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "AKIABATCHSIGNCALLER",
        "aws_secret_access_key": "pod-caller-secret",
        "aws_role_name": "arn:aws:iam::999999999999:role/litellm-batch-sign-role",
        "aws_session_name": "litellm-batch-sign-session",
        "aws_external_id": "external-id-batch-sign",
    }

    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        signed_headers, signed_data = CommonBatchFilesUtils().sign_aws_request(
            service_name="bedrock",
            data={"jobName": "litellm-batch-job"},
            endpoint_url="https://bedrock.us-east-1.amazonaws.com/model-invocation-job",
            optional_params=optional_params,
        )

    authorization = {key.lower(): value for key, value in signed_headers.items()}["authorization"]
    assert "ASIABATCHSIGNROLE" in authorization
    assert signed_data == b'{"jobName": "litellm-batch-job"}'


def test_sign_aws_request_assumes_role_with_session_tags(monkeypatch):
    """Batch and file signing must carry the deployment's session tags into the AssumeRole call too."""
    import datetime
    from unittest.mock import patch

    import boto3
    from botocore.exceptions import ClientError

    from litellm.llms.bedrock.common_utils import CommonBatchFilesUtils

    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    tags = [{"Key": "team", "Value": "genai"}]

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if list(params.get("Tags", ())) != tags:
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:TagSession"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIABATCHSIGNTAGGED",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
                }
            }

    optional_params = {
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "AKIABATCHSIGNCALLER",
        "aws_secret_access_key": "pod-caller-secret",
        "aws_role_name": "arn:aws:iam::999999999999:role/litellm-batch-sign-role",
        "aws_session_name": "litellm-batch-sign-session",
        "aws_session_tags": tags,
    }

    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        signed_headers, _signed_data = CommonBatchFilesUtils().sign_aws_request(
            service_name="bedrock",
            data={"jobName": "litellm-batch-job"},
            endpoint_url="https://bedrock.us-east-1.amazonaws.com/model-invocation-job",
            optional_params=optional_params,
        )

    authorization = {key.lower(): value for key, value in signed_headers.items()}["authorization"]
    assert "Credential=ASIABATCHSIGNTAGGED/" in authorization


# --------------------------------------------------------------------------- #
# Provider error headers (LIT-5428)                                            #
# --------------------------------------------------------------------------- #


def _bedrock_chat_error_configs():
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from litellm.llms.bedrock.chat.invoke_agent.transformation import AmazonInvokeAgentConfig
    from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
        AmazonMoonshotConfig,
    )
    from litellm.llms.bedrock.chat.invoke_transformations.amazon_openai_transformation import (
        AmazonBedrockOpenAIConfig,
    )
    from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
        AmazonInvokeConfig,
    )

    return [
        AmazonInvokeConfig,
        AmazonConverseConfig,
        AmazonMoonshotConfig,
        AmazonBedrockOpenAIConfig,
        AmazonAgentCoreConfig,
        AmazonInvokeAgentConfig,
    ]


@pytest.mark.parametrize("config", _bedrock_chat_error_configs())
def test_bedrock_chat_get_error_class_keeps_provider_headers(config):
    """Every Bedrock chat route must carry x-amzn-RequestId out to the caller (LIT-5428).

    A config that drops the headers it is handed shadows the fix for its own models.
    """
    error = config().get_error_class(
        error_message="Amazon Bedrock is unable to process your request.",
        status_code=500,
        headers={"x-amzn-RequestId": "req-chat-500"},
    )

    assert error.response.headers["x-amzn-requestid"] == "req-chat-500"


def test_error_response_text_reads_a_read_response():
    import httpx

    from litellm.llms.bedrock.common_utils import error_response_text

    response = httpx.Response(status_code=500, text="Amazon Bedrock is unable to process your request.")

    assert error_response_text(response) == "Amazon Bedrock is unable to process your request."


def test_error_response_text_falls_back_when_a_streamed_response_was_never_read():
    """A retried streamed request raises HTTPStatusError over an unread body; reading it
    throws ResponseNotRead and would lose the status and headers this fix preserves."""
    import httpx

    from litellm.llms.bedrock.common_utils import error_response_text

    request = httpx.Request(method="POST", url="https://bedrock-runtime.amazonaws.com")
    response = httpx.Response(
        status_code=500,
        headers={"x-amzn-RequestId": "req-unread-500"},
        stream=httpx.ByteStream(b"never read"),
        request=request,
    )

    with pytest.raises(httpx.ResponseNotRead):
        _ = response.text

    assert error_response_text(response) == "Internal Server Error"


def test_bedrock_error_skips_header_values_httpx_cannot_carry():
    """The shared HTTP handler copies an arbitrary exception's header values in verbatim,
    so a non-str value must not take down the whole error (LIT-5428)."""
    import httpx

    from litellm.llms.bedrock.common_utils import BedrockError

    error = BedrockError(
        status_code=500,
        message="boom",
        headers={"x-amzn-RequestId": "req-mixed-500", "x-retry-count": 3, "x-nothing": None},
    )

    assert error.response.headers["x-amzn-requestid"] == "req-mixed-500"
    assert "x-retry-count" not in error.response.headers
    assert isinstance(error.response, httpx.Response)


def test_bedrock_error_keeps_duplicate_httpx_header_values():
    import httpx

    from litellm.llms.bedrock.common_utils import BedrockError

    error = BedrockError(
        status_code=500,
        message="boom",
        headers=httpx.Headers([("x-amzn-RequestId", "req-dup-500"), ("set-cookie", "a=1"), ("set-cookie", "b=2")]),
    )

    assert error.response.headers.get_list("set-cookie") == ["a=1", "b=2"]


def _bedrock_httpx_status_error_sites():
    """Every `except httpx.HTTPStatusError as err` that raises a BedrockError, across bedrock."""
    import ast
    import pathlib

    sites = []
    for path in sorted(pathlib.Path("litellm/llms/bedrock").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
            caught = ast.unparse(handler.type) if handler.type is not None else ""
            if "HTTPStatusError" not in caught or handler.name is None:
                continue
            for call in (
                n
                for n in ast.walk(handler)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "BedrockError"
            ):
                sites.append((str(path), call.lineno, handler.name, {k.arg for k in call.keywords}))
    return sites


def test_every_bedrock_httpx_status_error_site_keeps_provider_headers():
    """A raise site holding the provider's failed response must hand its headers on (LIT-5428).

    These sites are the only place x-amzn-RequestId still exists; a site that drops it
    silently shadows the fix for that whole surface.
    """
    sites = _bedrock_httpx_status_error_sites()

    assert len(sites) >= 12
    dropped = [f"{path}:{lineno}" for path, lineno, _, kwargs in sites if "headers" not in kwargs]
    assert dropped == []


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_bedrock_embedding_call_keeps_provider_headers(is_async):
    """The embeddings surface raises from the same shape as chat and lost the same header."""
    import httpx

    from litellm.llms.bedrock.common_utils import BedrockError
    from litellm.llms.bedrock.embed.embedding import BedrockEmbedding
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    failure = httpx.Response(
        status_code=500,
        headers={"x-amzn-RequestId": "req-embed-500"},
        text='{"message":"Amazon Bedrock is unable to process your request."}',
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/"),
    )

    class _SyncUpstream(HTTPHandler):
        def post(self, *args, **kwargs):
            return failure

    class _AsyncUpstream(AsyncHTTPHandler):
        async def post(self, *args, **kwargs):
            return failure

    async def _drive():
        embedding = BedrockEmbedding()
        kwargs = dict(
            timeout=None,
            api_base="https://bedrock-runtime.us-east-1.amazonaws.com/",
            headers={},
            data={},
        )
        if is_async:
            return await embedding._make_async_call(client=_AsyncUpstream(), **kwargs)
        return embedding._make_sync_call(client=_SyncUpstream(), **kwargs)

    with pytest.raises(BedrockError) as exc_info:
        await _drive()

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-embed-500"


def _bedrock_mantle_error_configs():
    from litellm.llms.bedrock_mantle.chat.transformation import BedrockMantleChatConfig
    from litellm.llms.bedrock_mantle.responses.transformation import BedrockMantleResponsesAPIConfig

    return [BedrockMantleChatConfig, BedrockMantleResponsesAPIConfig]


@pytest.mark.parametrize("config", _bedrock_mantle_error_configs())
def test_bedrock_mantle_get_error_class_keeps_provider_headers(config):
    """bedrock_mantle rides the OpenAI-compatible surfaces, whose errors drop the headers.

    A chat request for a responses-API model is bridged onto the responses config, so
    fixing only the chat one leaves the model the customer actually calls uncovered.
    """
    error = config().get_error_class(
        error_message="prompt tokens exceed model maximum",
        status_code=400,
        headers={"x-amzn-RequestId": "req-mantle-400"},
    )

    assert error.response.headers["x-amzn-requestid"] == "req-mantle-400"


def _bedrock_configs_with_get_error_class():
    import importlib
    import inspect
    import pathlib

    import litellm

    llms_root = pathlib.Path(inspect.getfile(litellm)).parent / "llms"
    configs = []
    for package in ("bedrock", "bedrock_mantle"):
        for path in sorted((llms_root / package).rglob("*.py")):
            module_name = "litellm.llms." + ".".join(path.relative_to(llms_root).with_suffix("").parts)
            module = importlib.import_module(module_name)
            for name, obj in vars(module).items():
                if not inspect.isclass(obj) or obj.__module__ != module_name:
                    continue
                if getattr(obj, "get_error_class", None) is None:
                    continue
                configs.append(pytest.param(obj, id=f"{module_name}.{name}"))
    return configs


@pytest.mark.parametrize("config", _bedrock_configs_with_get_error_class())
def test_every_bedrock_config_get_error_class_keeps_provider_headers(config):
    """Every bedrock surface must classify errors through BedrockError, not a header-dropping base.

    A config that inherits get_error_class from a provider-agnostic base builds a blank
    response, so the request id is gone before the proxy ever reads it.
    """
    try:
        instance = config()
    except Exception:
        instance = config.__new__(config)

    try:
        error = instance.get_error_class(
            error_message="boom",
            status_code=500,
            headers={"x-amzn-RequestId": "req-audit-500"},
        )
    except Exception as raised:  # some bases raise the exception instead of returning it
        error = raised

    assert error.response.headers["x-amzn-requestid"] == "req-audit-500"


def test_bedrock_get_error_class_audit_covers_every_surface():
    assert len(_bedrock_configs_with_get_error_class()) >= 30


def test_s3_static_key_pair_returns_the_pair_when_both_keys_are_set():
    from litellm.llms.bedrock.common_utils import s3_static_key_pair

    assert s3_static_key_pair(
        {
            "aws_access_key_id": "bedrock-key",
            "aws_secret_access_key": "bedrock-secret",
            "s3_access_key_id": "s3-key",
            "s3_secret_access_key": "s3-secret",
        }
    ) == ("s3-key", "s3-secret")


@pytest.mark.parametrize(
    "partial_s3_pair",
    [
        {},
        {"s3_access_key_id": "s3-key"},
        {"s3_secret_access_key": "s3-secret"},
        {"s3_access_key_id": "", "s3_secret_access_key": ""},
    ],
)
def test_s3_static_key_pair_is_none_without_a_full_pair(partial_s3_pair):
    from litellm.llms.bedrock.common_utils import s3_static_key_pair

    assert s3_static_key_pair({"aws_access_key_id": "bedrock-key", **partial_s3_pair}) is None


def test_unmapped_openai_family_model_routes_to_converse():
    """A Bedrock-native OpenAI model that is not in the cost map yet must not fall to the invoke route.

    The invoke ``openai`` provider is the imported-model path and sends ``max_tokens``, which Bedrock
    rejects for these models; Converse maps it to ``inferenceConfig.maxTokens``.
    """
    from typing import Final

    import litellm

    unmapped: Final = "bedrock/global.openai.gpt-99-unmapped"
    assert unmapped.removeprefix("bedrock/") not in litellm.bedrock_converse_models
    assert BedrockModelInfo.get_bedrock_route(unmapped) == "converse"
    imported: Final = "bedrock/openai/arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123"
    assert BedrockModelInfo.get_bedrock_route(imported) == "openai"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("converse/us.anthropic.claude-haiku-4-5-20251001-v1:0", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        ("chat_completions/us.xai.grok-4.6", "us.xai.grok-4.6"),
        ("global.openai.gpt-5.6-sol", "global.openai.gpt-5.6-sol"),
    ],
)
def test_without_bedrock_route_prefix_hands_converse_the_bare_model_id(model, expected):
    from litellm.llms.bedrock.common_utils import without_bedrock_route_prefix

    assert without_bedrock_route_prefix(model) == expected


def test_bedrock_stream_event_statuses_cover_every_modeled_member_of_both_stream_shapes():
    pytest.importorskip("botocore")
    from botocore.loaders import Loader
    from botocore.model import ServiceModel

    import litellm.llms.bedrock.common_utils as mod

    mod.get_bedrock_stream_event_statuses.cache_clear()
    statuses = mod.get_bedrock_stream_event_statuses()
    assert statuses is not None

    service_model = ServiceModel(Loader().load_service_model("bedrock-runtime", "service-2"))
    for shape_name in ("ConverseStreamOutput", "ResponseStream"):
        for name, member in service_model.shape_for(shape_name).members.items():
            modeled = (member.metadata or {}).get("error", {}).get("httpStatusCode")
            assert statuses[name] == (None if modeled is None else int(modeled))
            assert mod.bedrock_stream_event_error_status(name) == statuses[name]

    assert any(status is not None for status in statuses.values())
    assert any(status is None for status in statuses.values())
    assert mod.bedrock_stream_event_error_status("notAModeledEvent") is None
    assert mod.bedrock_stream_event_error_status(None) is None


def test_bedrock_stream_event_statuses_load_failure_returns_none():
    from unittest.mock import patch

    import litellm.llms.bedrock.common_utils as mod

    pytest.importorskip("botocore")
    mod.get_bedrock_stream_event_statuses.cache_clear()
    with patch("botocore.loaders.Loader.load_service_model", side_effect=Exception("no data")):
        assert mod._load_bedrock_stream_event_statuses() is None
        assert mod.get_bedrock_stream_event_statuses() is None
        assert mod.bedrock_stream_event_error_status("validationException") is None
    mod.get_bedrock_stream_event_statuses.cache_clear()


@pytest.mark.parametrize(
    ("headers", "expected_status", "expected_message"),
    [
        ({":message-type": "error"}, 400, '{"message":"upstream failed"}'),
        (
            {":message-type": "exception", ":exception-type": "somethingNotModeled"},
            400,
            'somethingNotModeled {"message":"upstream failed"}',
        ),
        (
            {":message-type": "exception", ":exception-type": "throttlingException"},
            429,
            'throttlingException {"message":"upstream failed"}',
        ),
    ],
)
def test_build_bedrock_stream_error_resolves_status_from_the_exception_type(
    headers: dict[str, str], expected_status: int, expected_message: str
):
    pytest.importorskip("botocore")
    from litellm.llms.bedrock.common_utils import build_bedrock_stream_error, get_bedrock_response_stream_shape

    error = build_bedrock_stream_error(
        {"status_code": 400, "headers": headers, "body": b'{"message":"upstream failed"}'},
        get_bedrock_response_stream_shape(),
    )

    assert error.status_code == expected_status
    assert error.message == expected_message


@pytest.mark.parametrize(
    ("header_value", "expected"),
    [
        (
            '["interleaved-thinking-2025-05-14", "claude-code-20250219"]',
            ["interleaved-thinking-2025-05-14", "claude-code-20250219"],
        ),
        (' [" context-1m-2025-08-07 "] ', ["context-1m-2025-08-07"]),
        ("[]", []),
        ("[not-json]", ["[not-json]"]),
    ],
)
def test_get_anthropic_beta_from_headers_reads_a_json_array_header(header_value: str, expected: list[str]):
    from litellm.llms.bedrock.common_utils import get_anthropic_beta_from_headers

    assert get_anthropic_beta_from_headers({"anthropic-beta": header_value}) == expected


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestStripBedrockRoutingPrefix:
    """Tests for strip_bedrock_routing_prefix function."""

    def test_strips_bedrock_prefix(self):
        assert strip_bedrock_routing_prefix("bedrock/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_converse_prefix(self):
        assert strip_bedrock_routing_prefix("converse/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_invoke_prefix(self):
        assert strip_bedrock_routing_prefix("invoke/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_openai_prefix(self):
        assert strip_bedrock_routing_prefix("openai/gpt-4") == "gpt-4"

    def test_strips_all_known_prefixes(self):
        # Function strips all known prefixes iteratively
        # bedrock/converse/model -> converse/model -> model
        assert strip_bedrock_routing_prefix("bedrock/converse/claude-3") == "claude-3"

    def test_no_prefix_unchanged(self):
        assert strip_bedrock_routing_prefix("claude-3-sonnet") == "claude-3-sonnet"

    def test_model_with_dots_unchanged(self):
        assert (
            strip_bedrock_routing_prefix("anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestStripBedrockThroughputSuffix:
    """Tests for strip_bedrock_throughput_suffix function."""

    @pytest.mark.parametrize(
        "input_model,expected",
        [
            (
                "anthropic.claude-haiku-4-5-20251001-v1:0:51k",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            (
                "anthropic.claude-haiku-4-5-20251001-v1:0:18k",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            ("model:1:51k", "model:1"),
            ("model:123:18k", "model:123"),
            (
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            ("anthropic.claude-3-sonnet", "anthropic.claude-3-sonnet"),
        ],
    )
    def test_strip_throughput_suffix(self, input_model, expected):
        assert strip_bedrock_throughput_suffix(input_model) == expected

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestExtractModelNameFromBedrockArn:
    """Tests for extract_model_name_from_bedrock_arn function."""

    def test_extracts_from_provisioned_model_arn(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-model-id"
        assert extract_model_name_from_bedrock_arn(arn) == "my-model-id"

    def test_extracts_from_foundation_model_arn(self):
        arn = "arn:aws:bedrock:us-west-2:123456789012:foundation-model/anthropic.claude-v2"
        assert extract_model_name_from_bedrock_arn(arn) == "anthropic.claude-v2"

    def test_non_arn_unchanged(self):
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
        assert extract_model_name_from_bedrock_arn(model) == model

    def test_case_insensitive_arn_detection(self):
        arn = "ARN:aws:bedrock:us-east-1:123456789012:model/my-model"
        assert extract_model_name_from_bedrock_arn(arn) == "my-model"

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestGetBedrockCrossRegionInferenceRegions:
    """Tests for get_bedrock_cross_region_inference_regions function."""

    def test_returns_expected_regions(self):
        regions = get_bedrock_cross_region_inference_regions()
        assert "us" in regions
        assert "eu" in regions
        assert "global" in regions
        assert "apac" in regions

    def test_returns_list(self):
        regions = get_bedrock_cross_region_inference_regions()
        assert isinstance(regions, list)

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestGetBedrockBaseModel:
    """Tests for get_bedrock_base_model function."""

    def test_strips_bedrock_prefix(self):
        assert get_bedrock_base_model("bedrock/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_converse_prefix(self):
        assert get_bedrock_base_model("bedrock/converse/claude-3-sonnet") == "claude-3-sonnet"

    def test_strips_us_region_prefix(self):
        # us.anthropic.model -> anthropic.model
        assert (
            get_bedrock_base_model("us.anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )

    def test_strips_eu_region_prefix(self):
        assert (
            get_bedrock_base_model("eu.anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )

    def test_extracts_from_arn(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-model"
        assert get_bedrock_base_model(arn) == "my-model"

    def test_model_without_prefix_unchanged(self):
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
        assert get_bedrock_base_model(model) == model

    def test_combined_bedrock_and_region_prefix(self):
        # bedrock/us.anthropic.model -> anthropic.model
        assert (
            get_bedrock_base_model("bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0")
            == "anthropic.claude-3-sonnet-20240229-v1:0"
        )

    @pytest.mark.parametrize(
        "input_model,expected",
        [
            (
                "anthropic.claude-haiku-4-5-20251001-v1:0:51k",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            (
                "anthropic.claude-haiku-4-5-20251001-v1:0:18k",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            (
                "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0:51k",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            (
                "us.anthropic.claude-haiku-4-5-20251001-v1:0:51k",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
        ],
    )
    def test_strips_throughput_suffix(self, input_model, expected):
        """Test that throughput tier suffixes like :51k are stripped. Issue #19113."""
        assert get_bedrock_base_model(input_model) == expected

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestBedrockModelInfoWrappers:
    """Tests that BedrockModelInfo methods correctly wrap standalone functions."""

    def test_get_base_model_matches_standalone(self):
        test_cases = [
            "bedrock/claude-3-sonnet",
            "us.anthropic.claude-3-sonnet-20240229-v1:0",
            "arn:aws:bedrock:us-east-1:123:model/my-model",
        ]
        for model in test_cases:
            assert BedrockModelInfo.get_base_model(model) == get_bedrock_base_model(model)

    def test_extract_model_name_from_arn_matches_standalone(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/my-model"
        assert BedrockModelInfo.extract_model_name_from_arn(arn) == extract_model_name_from_bedrock_arn(arn)

    def test_get_non_litellm_routing_model_name_matches_standalone(self):
        model = "bedrock/converse/claude-3"
        assert BedrockModelInfo.get_non_litellm_routing_model_name(model) == strip_bedrock_routing_prefix(model)

@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
class TestBedrockTokenCounter:
    """Tests for BedrockTokenCounter class."""

    def test_should_use_token_counting_api_for_bedrock(self):
        counter = BedrockTokenCounter()
        assert counter.should_use_token_counting_api("bedrock") is True

    def test_should_not_use_token_counting_api_for_other_providers(self):
        counter = BedrockTokenCounter()
        assert counter.should_use_token_counting_api("openai") is False
        assert counter.should_use_token_counting_api("anthropic") is False
        assert counter.should_use_token_counting_api(None) is False

    def test_get_token_counter_returns_bedrock_token_counter(self):
        model_info = BedrockModelInfo()
        token_counter = model_info.get_token_counter()
        assert isinstance(token_counter, BedrockTokenCounter)

    @pytest.mark.asyncio
    async def test_count_tokens_returns_none_for_empty_messages(self):
        counter = BedrockTokenCounter()
        result = await counter.count_tokens(
            model_to_use="anthropic.claude-3-sonnet",
            messages=None,
            contents=None,
        )
        assert result is None

        result = await counter.count_tokens(
            model_to_use="anthropic.claude-3-sonnet",
            messages=[],
            contents=None,
        )
        assert result is None

@pytest.fixture
def _pr4_bedrock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "pr4-test-aws-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "pr4-test-aws-secret-key")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")

@pytest.fixture(scope="module")
def _use_local_model_cost_map() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        importlib.reload(bedrock_govcloud_model_cost_map)
        importlib.reload(litellm)
        yield

@pytest.mark.usefixtures(
    "_pr4_bedrock_env",
    "_use_local_model_cost_map",
    "_vcr_outcome_gate",
    "setup_and_teardown",
)
class TestBedrockGovCloudSupport:
    """Test suite for GovCloud model support in Bedrock"""

    def test_govcloud_regions_in_config(self):
        """Test that GovCloud regions are included in the configuration"""
        config = AmazonBedrockGlobalConfig()
        us_regions = config.get_us_regions()

        assert "us-gov-east-1" in us_regions
        assert "us-gov-west-1" in us_regions

        all_regions = config.get_all_regions()
        assert "us-gov-east-1" in all_regions
        assert "us-gov-west-1" in all_regions

    def test_govcloud_model_routing(self):
        """Test that GovCloud models are routed correctly"""
        # Test Claude model routing
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-east-1/anthropic.claude-haiku-4-5-20251001-v1:0")
        assert route == "converse"

        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-west-1/anthropic.claude-3-haiku-20240307-v1:0")
        assert route == "converse"

        # Test Llama model routing
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-east-1/meta.llama3-8b-instruct-v1:0")
        assert route == "converse"

        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-west-1/meta.llama3-70b-instruct-v1:0")
        assert route == "converse"

        # Test Titan model routing (should use invoke)
        route = BedrockModelInfo.get_bedrock_route("bedrock/us-gov-east-1/amazon.titan-text-lite-v1")
        assert route == "invoke"

    def test_base_model_extraction(self):
        """Test that base model names are correctly extracted from GovCloud models"""
        # Test GovCloud model extraction
        base_model = BedrockModelInfo.get_base_model("bedrock/us-gov-east-1/anthropic.claude-haiku-4-5-20251001-v1:0")
        assert base_model == "anthropic.claude-haiku-4-5-20251001-v1:0"

        base_model = BedrockModelInfo.get_base_model("bedrock/us-gov-west-1/meta.llama3-8b-instruct-v1:0")
        assert base_model == "meta.llama3-8b-instruct-v1:0"

    @patch("litellm.llms.bedrock.common_utils.init_bedrock_client")
    def test_govcloud_client_initialization(self, mock_init_client):
        """Test that Bedrock client can be initialized with GovCloud regions"""
        mock_client = Mock()
        mock_init_client.return_value = mock_client

        # Test that init_bedrock_client accepts GovCloud regions
        from litellm.llms.bedrock.common_utils import init_bedrock_client

        # This should not raise an error
        client = init_bedrock_client(
            region_name="us-gov-east-1",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_region_name="us-gov-east-1",
            aws_bedrock_runtime_endpoint=None,
            aws_session_name=None,
            aws_profile_name=None,
            aws_role_name=None,
            aws_web_identity_token=None,
            extra_headers=None,
            timeout=None,
        )

        assert mock_init_client.called

    def test_govcloud_model_in_bedrock_models_list(self):
        """Test that GovCloud models are NOT included in bedrock_models list (they are pricing-only)"""
        # Regional models including GovCloud should be excluded from bedrock_models list
        # They are only in model_cost for pricing purposes
        assert not any("us-gov-east-1" in model for model in litellm.bedrock_models)
        assert not any("us-gov-west-1" in model for model in litellm.bedrock_models)

    @patch("litellm.completion")
    def test_govcloud_completion_cost_calculation(self, mock_completion):
        """Test that completion requests use correct pricing for GovCloud models"""
        from litellm import Choices, Message, ModelResponse, completion_cost
        from litellm.utils import Usage

        # Mock completion response for base model
        # Use us.* inference profile ID to match us.* pricing ($1.10/$5.50 per MTok)
        base_model_response = ModelResponse(
            id="test-base",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello", role="assistant"),
                )
            ],
            created=1234567890,
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        base_model_response._hidden_params = {
            "custom_llm_provider": "bedrock",
            "region_name": "us-east-1",
        }

        # Mock completion response for gov model
        # GovCloud responses use base anthropic.* model ID; pricing is looked up
        # via bedrock/us-gov-east-1/anthropic.* entries in model_cost
        gov_model_response = ModelResponse(
            id="test-gov",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello", role="assistant"),
                )
            ],
            created=1234567890,
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        gov_model_response._hidden_params = {
            "custom_llm_provider": "bedrock",
            "region_name": "us-gov-east-1",
        }

        # Mock completion response for gov-west model
        gov_west_model_response = ModelResponse(
            id="test-gov-west",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello", role="assistant"),
                )
            ],
            created=1234567890,
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        gov_west_model_response._hidden_params = {
            "custom_llm_provider": "bedrock",
            "region_name": "us-gov-west-1",
        }

        # Test messages
        messages = [{"role": "user", "content": "Hello, how are you?"}]

        # Calculate costs using the standard Bedrock format with region parameter
        # Base model uses us.* inference profile — no region_name needed since
        # the response model already contains the us.* prefix for pricing lookup.
        base_cost = completion_cost(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            completion_response=base_model_response,
            messages=messages,
        )

        # GovCloud models use region_name to look up bedrock/us-gov-*/anthropic.* pricing
        gov_east_cost = completion_cost(
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            completion_response=gov_model_response,
            messages=messages,
            region_name="us-gov-east-1",
        )

        gov_west_cost = completion_cost(
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            completion_response=gov_west_model_response,
            messages=messages,
            region_name="us-gov-west-1",
        )

        # Expected costs based on pricing:
        # Base model (us.*): 10 * 1.1e-06 + 5 * 5.5e-06 = 1.1e-05 + 2.75e-05 = 3.85e-05
        # Gov models: 10 * 1.2e-06 + 5 * 6e-06 = 1.2e-05 + 3e-05 = 4.2e-05
        expected_base_cost = 10 * 1.1e-06 + 5 * 5.5e-06
        expected_gov_cost = 10 * 1.2e-06 + 5 * 6e-06

        # Verify costs are calculated correctly
        assert abs(base_cost - expected_base_cost) < 1e-10, (
            f"Base cost mismatch: got {base_cost}, expected {expected_base_cost}"
        )
        assert abs(gov_east_cost - expected_gov_cost) < 1e-10, (
            f"Gov East cost mismatch: got {gov_east_cost}, expected {expected_gov_cost}"
        )
        assert abs(gov_west_cost - expected_gov_cost) < 1e-10, (
            f"Gov West cost mismatch: got {gov_west_cost}, expected {expected_gov_cost}"
        )

        # Verify GovCloud costs are approximately 20% higher than base cost
        assert abs(gov_east_cost / base_cost - 1.2) < 0.15, (
            f"Gov East cost should be ~20% higher than base: got {gov_east_cost}, base {base_cost}"
        )
        assert abs(gov_west_cost / base_cost - 1.2) < 0.15, (
            f"Gov West cost should be ~20% higher than base: got {gov_west_cost}, base {base_cost}"
        )

        # Test with different token counts
        large_response = ModelResponse(
            id="test-large",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="A longer response", role="assistant"),
                )
            ],
            created=1234567890,
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        large_response._hidden_params = {
            "custom_llm_provider": "bedrock",
            "region_name": "us-east-1",
        }

        large_base_cost = completion_cost(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            completion_response=large_response,
            messages=messages,
        )

        # Create large response for gov model
        large_gov_response = ModelResponse(
            id="test-large-gov",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="A longer response", role="assistant"),
                )
            ],
            created=1234567890,
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        large_gov_response._hidden_params = {
            "custom_llm_provider": "bedrock",
            "region_name": "us-gov-east-1",
        }

        large_gov_cost = completion_cost(
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            completion_response=large_gov_response,
            messages=messages,
            region_name="us-gov-east-1",
        )

        # Expected costs for larger response:
        # Base model (us.*): 100 * 1.1e-06 + 50 * 5.5e-06 = 1.1e-04 + 2.75e-04 = 3.85e-04
        # Gov model: 100 * 1.2e-06 + 50 * 6e-06 = 1.2e-04 + 3e-04 = 4.2e-04
        expected_large_base_cost = 100 * 1.1e-06 + 50 * 5.5e-06
        expected_large_gov_cost = 100 * 1.2e-06 + 50 * 6e-06

        assert abs(large_base_cost - expected_large_base_cost) < 1e-10, (
            f"Large base cost mismatch: got {large_base_cost}, expected {expected_large_base_cost}"
        )
        assert abs(large_gov_cost - expected_large_gov_cost) < 1e-10, (
            f"Large gov cost mismatch: got {large_gov_cost}, expected {expected_large_gov_cost}"
        )
        assert abs(large_gov_cost / large_base_cost - 1.2) < 0.15, (
            f"Large gov cost should be ~20% higher than base: got {large_gov_cost}, base {large_base_cost}"
        )

    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_govcloud_completion_with_cost_tracking(self, mock_post):
        """Test that completion requests with cost tracking use correct pricing for GovCloud models"""
        import json
        from unittest.mock import Mock

        # Mock the HTTP client's post method to return responses
        def mock_post_side_effect(url, headers=None, data=None, **kwargs):
            # Extract region from the URL to determine which response to return
            region = "us-east-1"  # default
            if "us-gov-east-1" in url:
                region = "us-gov-east-1"
            elif "us-gov-west-1" in url:
                region = "us-gov-west-1"

            # Create mock response based on region
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}

            # Create a realistic Bedrock converse response structure
            bedrock_response = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"Hello from {region}"}],
                    }
                },
                "usage": {"inputTokens": 15, "outputTokens": 8, "totalTokens": 23},
                "stopReason": "end_turn",
            }

            mock_response.json.return_value = bedrock_response
            mock_response.text = json.dumps(bedrock_response)
            mock_response.raise_for_status = Mock()  # Don't raise exceptions

            return mock_response

        mock_post.side_effect = mock_post_side_effect

        # Test base model completion
        base_result = completion(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            aws_region_name="us-east-1",
        )

        # Test gov-east model completion
        # GovCloud users specify the base anthropic.* model ID with the gov region
        gov_east_result = completion(
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            aws_region_name="us-gov-east-1",
        )

        # Test gov-west model completion
        gov_west_result = completion(
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
            aws_region_name="us-gov-west-1",
        )

        # Verify the mock was called correctly
        assert mock_post.call_count == 3

        # Verify usage information is present
        from litellm.types.utils import ModelResponse

        assert isinstance(base_result, ModelResponse)
        assert isinstance(gov_east_result, ModelResponse)
        assert isinstance(gov_west_result, ModelResponse)

        base_result_typed: ModelResponse = base_result
        gov_east_result_typed: ModelResponse = gov_east_result
        gov_west_result_typed: ModelResponse = gov_west_result

        # Verify usage information is present
        assert hasattr(base_result_typed, "usage") and base_result_typed.usage.prompt_tokens == 15
        assert hasattr(base_result_typed, "usage") and base_result_typed.usage.completion_tokens == 8
        assert hasattr(gov_east_result_typed, "usage") and gov_east_result_typed.usage.prompt_tokens == 15
        assert hasattr(gov_east_result_typed, "usage") and gov_east_result_typed.usage.completion_tokens == 8
        assert hasattr(gov_west_result_typed, "usage") and gov_west_result_typed.usage.prompt_tokens == 15
        assert hasattr(gov_west_result_typed, "usage") and gov_west_result_typed.usage.completion_tokens == 8

        # Verify cost calculation uses correct pricing for each region
        # Get costs directly from the completion response _hidden_params
        base_cost = base_result_typed._hidden_params.get("response_cost", 0.0)
        gov_east_cost = gov_east_result_typed._hidden_params.get("response_cost", 0.0)
        gov_west_cost = gov_west_result_typed._hidden_params.get("response_cost", 0.0)

        print(f"Base cost: {base_cost}")
        print(f"Gov East cost: {gov_east_cost}")
        print(f"Gov West cost: {gov_west_cost}")

        # Expected costs based on pricing:
        # Base model (us.*): 15 * 1.1e-06 + 8 * 5.5e-06 = 1.65e-05 + 4.4e-05 = 6.05e-05
        # Gov models: 15 * 1.2e-06 + 8 * 6e-06 = 1.8e-05 + 4.8e-05 = 6.6e-05
        expected_base_cost = 15 * 1.1e-06 + 8 * 5.5e-06
        expected_gov_cost = 15 * 1.2e-06 + 8 * 6e-06

        # Verify costs are calculated correctly
        assert abs(base_cost - expected_base_cost) < 1e-10, (
            f"Base cost mismatch: got {base_cost}, expected {expected_base_cost}"
        )
        assert abs(gov_east_cost - expected_gov_cost) < 1e-10, (
            f"Gov East cost mismatch: got {gov_east_cost}, expected {expected_gov_cost}"
        )
        assert abs(gov_west_cost - expected_gov_cost) < 1e-10, (
            f"Gov West cost mismatch: got {gov_west_cost}, expected {expected_gov_cost}"
        )

        # Verify GovCloud costs are approximately 20% higher than base cost
        assert abs(gov_east_cost / base_cost - 1.2) < 0.15, (
            f"Gov East cost should be ~20% higher than base: got {gov_east_cost}, base {base_cost}"
        )
        assert abs(gov_west_cost / base_cost - 1.2) < 0.15, (
            f"Gov West cost should be ~20% higher than base: got {gov_west_cost}, base {base_cost}"
        )

        # Print cost information for verification
        print(f"Base model cost: ${base_cost:.6f}")
        print(f"GovCloud East cost: ${gov_east_cost:.6f}")
        print(f"GovCloud West cost: ${gov_west_cost:.6f}")
        print(f"GovCloud cost increase: {((gov_east_cost / base_cost) - 1) * 100:.1f}%")

    def test_govcloud_cost_per_token_with_region(self):
        """Test that cost_per_token function correctly uses region-based pricing for GovCloud models"""
        from litellm import cost_per_token
        from litellm.utils import Usage

        # Test usage object
        usage = Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)

        # Commercial list pricing uses the us.* inference profile id; GovCloud keys use anthropic.* + region
        haiku_us_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        haiku_anthropic_id = "anthropic.claude-haiku-4-5-20251001-v1:0"
        # Test base model with standard region
        base_prompt_cost, base_completion_cost = cost_per_token(
            model=haiku_us_id,
            prompt_tokens=20,
            completion_tokens=10,
            custom_llm_provider="bedrock",
            region_name="us-east-1",
        )

        # Test gov models with gov regions
        gov_east_prompt_cost, gov_east_completion_cost = cost_per_token(
            model=haiku_anthropic_id,
            prompt_tokens=20,
            completion_tokens=10,
            custom_llm_provider="bedrock",
            region_name="us-gov-east-1",
        )

        gov_west_prompt_cost, gov_west_completion_cost = cost_per_token(
            model=haiku_anthropic_id,
            prompt_tokens=20,
            completion_tokens=10,
            custom_llm_provider="bedrock",
            region_name="us-gov-west-1",
        )

        # Expected costs:
        # Base model (us.*): 20 * 1.1e-06 + 10 * 5.5e-06 = 2.2e-05 + 5.5e-05 = 7.7e-05
        # Gov models: 20 * 1.2e-06 + 10 * 6e-06 = 2.4e-05 + 6e-05 = 8.4e-05
        expected_base_prompt_cost = 20 * 1.1e-06
        expected_base_completion_cost = 10 * 5.5e-06
        expected_gov_prompt_cost = 20 * 1.2e-06
        expected_gov_completion_cost = 10 * 6e-06

        # Verify costs are calculated correctly
        assert abs(base_prompt_cost - expected_base_prompt_cost) < 1e-10, (
            f"Base prompt cost mismatch: got {base_prompt_cost}, expected {expected_base_prompt_cost}"
        )
        assert abs(base_completion_cost - expected_base_completion_cost) < 1e-10, (
            f"Base completion cost mismatch: got {base_completion_cost}, expected {expected_base_completion_cost}"
        )

        assert abs(gov_east_prompt_cost - expected_gov_prompt_cost) < 1e-10, (
            f"Gov East prompt cost mismatch: got {gov_east_prompt_cost}, expected {expected_gov_prompt_cost}"
        )
        assert abs(gov_east_completion_cost - expected_gov_completion_cost) < 1e-10, (
            f"Gov East completion cost mismatch: got {gov_east_completion_cost}, expected {expected_gov_completion_cost}"
        )

        assert abs(gov_west_prompt_cost - expected_gov_prompt_cost) < 1e-10, (
            f"Gov West prompt cost mismatch: got {gov_west_prompt_cost}, expected {expected_gov_prompt_cost}"
        )
        assert abs(gov_west_completion_cost - expected_gov_completion_cost) < 1e-10, (
            f"Gov West completion cost mismatch: got {gov_west_completion_cost}, expected {expected_gov_completion_cost}"
        )

        # Verify GovCloud costs are approximately 20% higher than base costs
        # (uses 1e-8 tolerance because GovCloud prices are independently rounded, not exact * 1.2)
        assert abs(gov_east_prompt_cost / base_prompt_cost - 1.2) < 0.15, (
            f"Gov East prompt cost should be ~20% higher than base: got {gov_east_prompt_cost}, base {base_prompt_cost}"
        )
        assert abs(gov_east_completion_cost / base_completion_cost - 1.2) < 0.15, (
            f"Gov East completion cost should be ~20% higher than base: got {gov_east_completion_cost}, base {base_completion_cost}"
        )
        assert abs(gov_west_prompt_cost / base_prompt_cost - 1.2) < 0.15, (
            f"Gov West prompt cost should be ~20% higher than base: got {gov_west_prompt_cost}, base {base_prompt_cost}"
        )
        assert abs(gov_west_completion_cost / base_completion_cost - 1.2) < 0.15, (
            f"Gov West completion cost should be ~20% higher than base: got {gov_west_completion_cost}, base {base_completion_cost}"
        )

        # Test total costs
        base_total_cost = base_prompt_cost + base_completion_cost
        gov_east_total_cost = gov_east_prompt_cost + gov_east_completion_cost
        gov_west_total_cost = gov_west_prompt_cost + gov_west_completion_cost

        expected_base_total = expected_base_prompt_cost + expected_base_completion_cost
        expected_gov_total = expected_gov_prompt_cost + expected_gov_completion_cost

        assert abs(base_total_cost - expected_base_total) < 1e-10, (
            f"Base total cost mismatch: got {base_total_cost}, expected {expected_base_total}"
        )
        assert abs(gov_east_total_cost - expected_gov_total) < 1e-10, (
            f"Gov East total cost mismatch: got {gov_east_total_cost}, expected {expected_gov_total}"
        )
        assert abs(gov_west_total_cost - expected_gov_total) < 1e-10, (
            f"Gov West total cost mismatch: got {gov_west_total_cost}, expected {expected_gov_total}"
        )
        assert abs(gov_east_total_cost / base_total_cost - 1.2) < 0.15, (
            f"Gov East total cost should be ~20% higher than base: got {gov_east_total_cost}, base {base_total_cost}"
        )
        assert abs(gov_west_total_cost / base_total_cost - 1.2) < 0.15, (
            f"Gov West total cost should be ~20% higher than base: got {gov_west_total_cost}, base {base_total_cost}"
        )

    @pytest.mark.parametrize(
        "model_name",
        [
            "bedrock/us-gov-east-1/anthropic.claude-haiku-4-5-20251001-v1:0",
            "bedrock/us-gov-west-1/anthropic.claude-3-haiku-20240307-v1:0",
            "bedrock/us-gov-east-1/meta.llama3-8b-instruct-v1:0",
            "bedrock/us-gov-west-1/meta.llama3-70b-instruct-v1:0",
        ],
    )
    def test_govcloud_converse_models(self, model_name):
        """Test that GovCloud Claude and Llama models support Converse API"""
        route = BedrockModelInfo.get_bedrock_route(model_name)
        assert route == "converse"

    @pytest.mark.parametrize(
        "model_name",
        [
            "bedrock/us-gov-east-1/amazon.titan-text-lite-v1",
            "bedrock/us-gov-west-1/amazon.titan-text-express-v1",
            "bedrock/us-gov-east-1/amazon.titan-text-premier-v1:0",
        ],
    )
    def test_govcloud_invoke_models(self, model_name):
        """Test that GovCloud Titan models use Invoke API"""
        route = BedrockModelInfo.get_bedrock_route(model_name)
        assert route == "invoke"
