"""
Pure-logic contract tests for litellm/proxy/video_endpoints/utils.py

Five helpers the video proxy endpoints lean on:
  - extract_model_from_target_model_names: first model from a comma string / list
  - video_reference_to_id: normalize a video reference (dict / bare id / JSON string) to an id
  - get_custom_provider_from_data: provider precedence (top-level > extra_body)
  - encode_character_id_in_response: re-encode a response id in place
  - encode_video_id_in_response: re-stamp a VideoObject id with the resolved model_id

Every test asserts the exact result (or identity), so a mutation that flips a
branch, drops a strip/filter, or changes precedence fails. The only collaborator
is encode_character_id_with_provider, which runs for real; encoding assertions
are checked by the genuine decode round-trip.
"""

from typing import Final
from unittest.mock import MagicMock, patch

import pytest


from litellm.router import Router
from litellm.proxy._types import (
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamTableCachedObj,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from litellm.types.videos.main import DecodedVideoId
from litellm.proxy.video_endpoints.utils import (
    authorized_video_model,
    encode_character_id_in_response,
    encode_video_id_in_response,
    extract_model_from_target_model_names,
    fallback_requested_model,
    get_custom_provider_from_data,
    routable_model_name,
    stamp_listed_video_ids,
    VideoListTarget,
    video_list_target,
    video_model_group,
    video_reference_to_id,
)
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import (
    decode_character_id_with_provider,
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)

# =========================================================================== #
# extract_model_from_target_model_names
# =========================================================================== #


@pytest.mark.parametrize(
    "value,expected",
    [
        ("m1,m2,m3", "m1"),
        ("  a , b ", "a"),  # leading/trailing whitespace stripped
        (",, m1 ,,", "m1"),  # empty tokens filtered out
        ("solo", "solo"),  # single token, no comma
        ("", None),  # empty string -> no tokens
        (" , , ", None),  # only separators/whitespace -> no tokens
        (["x", "y"], "x"),  # list -> first element
        ([], None),  # empty list
    ],
)
def test_extract_model__str_and_list(value, expected):
    assert extract_model_from_target_model_names(value) == expected


@pytest.mark.parametrize("value", [None, 123, {"a": 1}, 4.5])
def test_extract_model__non_str_non_list_is_none(value):
    assert extract_model_from_target_model_names(value) is None


# =========================================================================== #
# video_reference_to_id
# =========================================================================== #


@pytest.mark.parametrize(
    "video_ref,expected",
    [
        ({"id": "video_123"}, "video_123"),  # dict reference -> its id
        ({"id": ""}, ""),  # dict with empty id
        ({}, ""),  # dict missing id -> default empty
        ({"other": "x"}, ""),  # dict without id key
        ("video_123", "video_123"),  # bare id string (not valid JSON) -> itself
        ('{"id": "video_9"}', "video_9"),  # JSON-encoded dict -> its id
        ('{"other": 1}', ""),  # JSON-encoded dict without id -> empty
        ("[1, 2]", "[1, 2]"),  # JSON parses to non-dict -> original string
        (None, ""),  # non-str, non-dict
        (123, ""),  # non-str, non-dict
        (["video_123"], ""),  # list is neither dict nor str
    ],
)
def test_video_reference_to_id(video_ref, expected):
    assert video_reference_to_id(video_ref) == expected


# =========================================================================== #
# get_custom_provider_from_data
# =========================================================================== #


def test_provider__top_level_wins_over_extra_body():
    data = {
        "custom_llm_provider": "azure",
        "extra_body": {"custom_llm_provider": "openai"},
    }
    assert get_custom_provider_from_data(data) == "azure"


@pytest.mark.parametrize("falsy", ["", None])
def test_provider__falsy_top_level_falls_through_to_extra_body(falsy):
    data = {
        "custom_llm_provider": falsy,
        "extra_body": {"custom_llm_provider": "vertex_ai"},
    }
    assert get_custom_provider_from_data(data) == "vertex_ai"


def test_provider__from_extra_body_dict():
    assert get_custom_provider_from_data({"extra_body": {"custom_llm_provider": "bedrock"}}) == "bedrock"


def test_provider__from_extra_body_json_string():
    data = {"extra_body": '{"custom_llm_provider": "gemini"}'}
    assert get_custom_provider_from_data(data) == "gemini"


def test_provider__invalid_json_string_is_none():
    assert get_custom_provider_from_data({"extra_body": "not-json{"}) is None


def test_provider__json_string_parsing_to_non_dict_is_none():
    # parses to a list, not a dict -> no provider extracted.
    assert get_custom_provider_from_data({"extra_body": "[1, 2]"}) is None


def test_provider__extra_body_provider_not_a_string_is_none():
    assert get_custom_provider_from_data({"extra_body": {"custom_llm_provider": 123}}) is None


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"extra_body": {}},
        {"extra_body": 5},  # non-dict, non-str
        {"extra_body": {"other": "x"}},  # dict without provider key
    ],
)
def test_provider__no_provider_anywhere_is_none(data):
    assert get_custom_provider_from_data(data) is None


# =========================================================================== #
# encode_character_id_in_response
# =========================================================================== #


class _Resp:
    """Minimal response object exposing an `id` attribute."""


def test_encode__dict_with_id_mutates_in_place_and_preserves_other_keys():
    response = {"id": "char_raw", "object": "character", "name": "hero"}

    out = encode_character_id_in_response(response, "azure", "model-1")

    assert out is response  # same dict, mutated in place
    assert out["object"] == "character" and out["name"] == "hero"
    assert out["id"] == encode_character_id_with_provider("char_raw", "azure", "model-1")
    decoded = decode_character_id_with_provider(out["id"])
    assert decoded["custom_llm_provider"] == "azure"
    assert decoded["model_id"] == "model-1"
    assert decoded["character_id"] == "char_raw"


@pytest.mark.parametrize("response", [{}, {"id": ""}, {"id": None}])
def test_encode__dict_without_usable_id_unchanged(response):
    snapshot = dict(response)
    out = encode_character_id_in_response(response, "azure", "model-1")
    assert out == snapshot


def test_encode__object_with_str_id():
    resp = _Resp()
    resp.id = "char_raw"

    out = encode_character_id_in_response(resp, "openai", None)

    assert out is resp
    assert resp.id == encode_character_id_with_provider("char_raw", "openai", None)
    decoded = decode_character_id_with_provider(resp.id)
    assert decoded["custom_llm_provider"] == "openai"
    assert decoded["character_id"] == "char_raw"


@pytest.mark.parametrize("bad_id", [None, 123, ""])
def test_encode__object_non_str_or_empty_id_unchanged(bad_id):
    resp = _Resp()
    resp.id = bad_id

    out = encode_character_id_in_response(resp, "azure", "model-1")

    assert out is resp
    assert resp.id == bad_id  # untouched


def test_encode__object_without_id_attr_returned_unchanged():
    resp = _Resp()
    out = encode_character_id_in_response(resp, "azure", "model-1")
    assert out is resp
    assert not hasattr(resp, "id")


# =========================================================================== #
# encode_video_id_in_response
# =========================================================================== #


def _video(video_id: str, hidden_params: dict[str, object] | None = None) -> VideoObject:
    video: Final[VideoObject] = VideoObject(id=video_id, object="video", status="queued")
    video.hidden_params = dict(hidden_params or {})
    return video


def test_encode_video__fills_empty_model_id_and_deployment_from_hidden_params() -> None:
    original_id: Final = encode_video_id_with_provider("video_raw", "openai", "")
    video: Final = _video(original_id, hidden_params={"model_id": "deployment-1"})

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id=None)

    assert isinstance(out, VideoObject)
    assert decode_video_id_with_provider(out.id) == {
        "custom_llm_provider": "openai",
        "model_id": "deployment-1",
        "video_id": "video_raw",
        "deployment_id": "deployment-1",
    }
    assert out is not video
    assert video.id == original_id
    assert out.hidden_params == {"model_id": "deployment-1"}
    assert out.status == "queued"


def test_encode_video__asked_for_model_wins_over_hidden_deployment_id() -> None:
    video: Final = _video(
        encode_video_id_with_provider("video_raw", "openai", ""), hidden_params={"model_id": "deployment-1"}
    )

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id="sora-alias")

    assert isinstance(out, VideoObject)
    decoded: Final = decode_video_id_with_provider(out.id)
    assert decoded.get("model_id") == "sora-alias"
    assert decoded.get("deployment_id") == "deployment-1"


def test_encode_video__stamped_id_keeps_model_and_served_group_and_gains_routed_deployment() -> None:
    original_id: Final = encode_video_id_with_provider("task_1", "volcengine", "seedance-group", "seedance-east")
    video: Final = _video(original_id, hidden_params={"model_id": "deployment-1"})

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id="other")

    assert isinstance(out, VideoObject)
    assert decode_video_id_with_provider(out.id) == {
        "custom_llm_provider": "volcengine",
        "model_id": "seedance-group",
        "video_id": "task_1",
        "served_model": "seedance-east",
        "deployment_id": "deployment-1",
    }


def test_encode_video__routed_deployment_replaces_the_one_in_the_id() -> None:
    original_id: Final = encode_video_id_with_provider("video_raw", "openai", "sora", deployment_id="deployment-old")
    video: Final = _video(original_id, hidden_params={"model_id": "deployment-new"})

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id=None)

    assert isinstance(out, VideoObject)
    assert decode_video_id_with_provider(out.id).get("deployment_id") == "deployment-new"


@pytest.mark.parametrize(
    ("polled_deployment", "answering_deployment", "returned_deployment"),
    [
        (None, "deployment-routed", "deployment-routed"),
        ("deployment-gone", "deployment-routed", "deployment-routed"),
        ("deployment-creator", None, "deployment-creator"),
    ],
)
def test_encode_video__a_poll_takes_the_deployment_that_answered_else_keeps_the_one_its_id_named(
    polled_deployment: str | None,
    answering_deployment: str | None,
    returned_deployment: str,
) -> None:
    polled_id: Final = encode_video_id_with_provider("video_raw", "openai", "sora", deployment_id=polled_deployment)
    video: Final = _video(
        encode_video_id_with_provider("video_raw", "openai", ""),
        hidden_params={} if answering_deployment is None else {"model_id": answering_deployment},
    )

    out: Final = encode_video_id_in_response(
        video,
        fallback_provider=None,
        fallback_model_id="sora",
        polled_video=decode_video_id_with_provider(polled_id),
    )

    assert isinstance(out, VideoObject)
    assert out.id == encode_video_id_with_provider("video_raw", "openai", "sora", deployment_id=returned_deployment)


def test_encode_video__a_poll_keeps_the_group_a_fallback_create_was_serving() -> None:
    polled_id: Final = encode_video_id_with_provider(
        "video_raw", "azure", "sora-2", deployment_id="dep-backup", requested_model="sora-pair"
    )
    video: Final = _video(encode_video_id_with_provider("video_raw", "azure", "sora-2"))

    out: Final = encode_video_id_in_response(
        video, fallback_provider=None, fallback_model_id=None, polled_video=decode_video_id_with_provider(polled_id)
    )

    assert isinstance(out, VideoObject)
    assert out.id == polled_id


def test_encode_video__a_remix_of_a_fallback_video_keeps_its_requested_group() -> None:
    video: Final = _video(
        encode_video_id_with_provider("video_new", "azure", "sora-2"), hidden_params={"model_id": "dep-backup"}
    )

    out: Final = encode_video_id_in_response(
        video, fallback_provider=None, fallback_model_id=None, requested_model="sora-pair"
    )

    assert isinstance(out, VideoObject)
    assert decode_video_id_with_provider(out.id).get("requested_model") == "sora-pair"


def test_encode_video__id_already_naming_the_routed_deployment_is_unchanged() -> None:
    original_id: Final = encode_video_id_with_provider("video_raw", "openai", "sora", deployment_id="deployment-1")
    video: Final = _video(original_id, hidden_params={"model_id": "deployment-1"})

    assert encode_video_id_in_response(video, fallback_provider=None, fallback_model_id="other") is video


def test_encode_video__unrouted_response_keeps_the_deployment_in_its_id() -> None:
    original_id: Final = encode_video_id_with_provider("video_raw", "openai", "", deployment_id="deployment-1")

    out: Final = encode_video_id_in_response(_video(original_id), fallback_provider=None, fallback_model_id="sora")

    assert isinstance(out, VideoObject)
    decoded: Final = decode_video_id_with_provider(out.id)
    assert decoded.get("model_id") == "sora"
    assert decoded.get("deployment_id") == "deployment-1"


def test_encode_video__name_with_id_separator_is_never_written_into_the_id() -> None:
    video: Final = _video(
        encode_video_id_with_provider("video_raw", "openai", ""), hidden_params={"model_id": "deployment-1"}
    )

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id="sora;model_id:evil")

    assert isinstance(out, VideoObject)
    assert decode_video_id_with_provider(out.id) == {
        "custom_llm_provider": "openai",
        "model_id": "deployment-1",
        "video_id": "video_raw",
        "deployment_id": "deployment-1",
    }


def test_encode_video__falls_back_to_given_model_id() -> None:
    video: Final = _video(encode_video_id_with_provider("video_raw", "openai", ""))

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id="sora-2")

    assert isinstance(out, VideoObject)
    decoded: Final = decode_video_id_with_provider(out.id)
    assert decoded.get("model_id") == "sora-2"
    assert decoded.get("video_id") == "video_raw"


def test_encode_video__keeps_existing_model_id_over_fallback() -> None:
    original_id: Final = encode_video_id_with_provider("video_raw", "azure", "deployment-1")

    out: Final = encode_video_id_in_response(_video(original_id), fallback_provider="openai", fallback_model_id="other")

    assert isinstance(out, VideoObject)
    assert out.id == original_id


def test_encode_video__plain_id_encoded_with_fallback_provider() -> None:
    video: Final = _video("video_raw", hidden_params={"model_id": "deployment-1"})

    out: Final = encode_video_id_in_response(video, fallback_provider="openai", fallback_model_id=None)

    assert isinstance(out, VideoObject)
    assert decode_video_id_with_provider(out.id) == {
        "custom_llm_provider": "openai",
        "model_id": "deployment-1",
        "video_id": "video_raw",
        "deployment_id": "deployment-1",
    }


def test_encode_video__no_model_id_anywhere_unchanged() -> None:
    video: Final = _video(encode_video_id_with_provider("video_raw", "openai", ""))

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id=None)

    assert out is video


def test_encode_video__no_provider_anywhere_unchanged() -> None:
    video: Final = _video("video_raw")

    out: Final = encode_video_id_in_response(video, fallback_provider=None, fallback_model_id="sora-2")

    assert out is video


@pytest.mark.parametrize("response", [b"video-bytes", {"id": "video_raw"}, None])
def test_encode_video__non_video_object_passthrough(response: object) -> None:
    assert encode_video_id_in_response(response, "openai", "sora-2") is response


def _router_with_groups(*names: str) -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": "openai/sora-2", "api_key": "group-key"}} for name in names
        ]
    )


def test_routable_model_name__keeps_a_name_the_router_resolves() -> None:
    assert routable_model_name("sora-alias", _router_with_groups("sora-alias"), "openai") == "sora-alias"


def test_routable_model_name__drops_a_name_the_router_cannot_route() -> None:
    assert routable_model_name("team-public-name", _router_with_groups("sora-alias"), "openai") is None


def test_routable_model_name__keeps_the_name_without_a_router() -> None:
    assert routable_model_name("sora-2", None, None) == "sora-2"


@pytest.mark.parametrize("model", [None, "", 7, "sora;model_id:evil"])
def test_routable_model_name__never_returns_a_name_unsafe_for_the_id(model: object) -> None:
    assert routable_model_name(model, _router_with_groups("sora;model_id:evil"), "openai") is None


def _two_groups_sharing_sora() -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "azure/sora-2", "api_base": f"https://{name}.example.com", "api_key": name},
                "model_info": {"id": f"dep-{name}"},
            }
            for name in ("sora-team", "sora-shared")
        ]
    )


def _decoded(model_id: str, deployment_id: str | None) -> DecodedVideoId:
    base: Final = DecodedVideoId(custom_llm_provider="azure", model_id=model_id, video_id="video_raw")
    return base if deployment_id is None else {**base, "deployment_id": deployment_id}


@pytest.mark.parametrize("deployment_id", ["dep-sora-team", "dep-sora-shared"])
def test_video_model_group__shared_provider_model_resolves_to_the_creating_deployments_group(
    deployment_id: str,
) -> None:
    router: Final = _two_groups_sharing_sora()

    assert video_model_group(_decoded("sora-2", deployment_id), router) == deployment_id.removeprefix("dep-")


@pytest.mark.parametrize("deployment_id", [None, "dep-gone"])
def test_video_model_group__without_a_known_deployment_matches_plain_resolution(deployment_id: str | None) -> None:
    router: Final = _two_groups_sharing_sora()

    assert video_model_group(_decoded("sora-2", deployment_id), router) == router.resolve_model_name_from_model_id(
        "sora-2", "azure"
    )


def test_video_model_group__keeps_a_group_named_in_the_id_over_a_deployment_of_another_group() -> None:
    assert video_model_group(_decoded("sora-team", "dep-sora-shared"), _two_groups_sharing_sora()) == "sora-team"


@pytest.mark.parametrize("deployment_id", ["dep-sora-team", "dep-sora-shared"])
def test_video_model_group__keeps_an_alias_of_the_creating_deployments_group(deployment_id: str) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": f"openai/sora-{name}", "api_key": name},
                "model_info": {"id": f"dep-{name}"},
            }
            for name in ("sora-team", "sora-shared")
        ],
        model_group_alias={"sora-alias": deployment_id.removeprefix("dep-")},
    )
    provider_model: Final = f"sora-{deployment_id.removeprefix('dep-')}"

    assert video_model_group(_decoded(provider_model, deployment_id), router) == "sora-alias"


def test_video_model_group__names_a_team_model_by_its_public_name() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "model_name_team-1_internal",
                "litellm_params": {"model": "azure/sora-2", "api_key": "team-key"},
                "model_info": {"id": "dep-team", "team_id": "team-1", "team_public_model_name": "team-sora"},
            }
        ]
    )

    assert video_model_group(_decoded("sora-2", "dep-team"), router) == "team-sora"


def _fallback_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {"model": "azure/sora-2", "api_base": f"https://{name}.example.com", "api_key": name},
                "model_info": {"id": f"dep-{name}"},
            }
            for name in ("sora-pair", "sora-backup", "sora-last")
        ],
        model_group_alias={"sora-alias": "sora-pair"},
        fallbacks=[{"sora-pair": ["sora-backup"]}, {"sora-backup": ["sora-last"]}],
    )


def _wildcard_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*", "api_key": "wildcard-key"},
                "model_info": {"id": "dep-wildcard"},
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enforce_fallback_model_access", "allowed_model"),
    [(False, "sora-pair"), (True, "sora-backup")],
)
async def test_authorized_video_model_keeps_fallback_routing_after_the_key_check(
    monkeypatch: pytest.MonkeyPatch, enforce_fallback_model_access: bool, allowed_model: str
) -> None:
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {"enforce_fallback_model_access": enforce_fallback_model_access},
    )
    decoded: Final = decode_video_id_with_provider(
        encode_video_id_with_provider(
            "video_raw", "azure", "sora-2", deployment_id="dep-sora-backup", requested_model="sora-pair"
        )
    )
    token: Final = UserAPIKeyAuth(api_key="hashed", models=[allowed_model])

    assert await authorized_video_model(decoded, _fallback_router(), token) == "sora-backup"


@pytest.mark.asyncio
async def test_authorized_video_model_without_a_router_has_no_routing_model() -> None:
    assert await authorized_video_model(_decoded("sora-2", "dep-sora-backup"), None, UserAPIKeyAuth()) is None


@pytest.mark.parametrize(
    ("requested_model", "answering_deployment", "expected"),
    [
        ("sora-pair", "dep-sora-backup", "sora-pair"),
        ("sora-pair", "dep-sora-last", "sora-pair"),
        ("sora-backup", "dep-sora-pair", None),
        ("sora-pair", "dep-sora-pair", None),
        ("sora-alias", "dep-sora-pair", None),
        ("sora-pair", None, None),
        (None, "dep-sora-backup", None),
    ],
    ids=[
        "fallback-served",
        "chained-fallback-served",
        "served-by-a-group-it-never-falls-back-to",
        "served-by-own-group",
        "served-by-alias-target",
        "unrouted",
        "no-requested-model",
    ],
)
def test_fallback_requested_model__names_the_asked_group_only_when_another_group_served_the_create(
    requested_model: str | None, answering_deployment: str | None, expected: str | None
) -> None:
    video: Final = _video(
        encode_video_id_with_provider("video_raw", "azure", "sora-2"),
        hidden_params={} if answering_deployment is None else {"model_id": answering_deployment},
    )

    assert fallback_requested_model(requested_model, video, _fallback_router()) == expected


def test_fallback_requested_model__is_none_for_a_create_a_provider_wildcard_served() -> None:
    video: Final = _video(
        encode_video_id_with_provider("operations/op_1", "gemini", "veo-3.1-fast-generate-preview"),
        hidden_params={"model_id": "dep-wildcard"},
    )

    assert fallback_requested_model("gemini/veo-3.1-fast-generate-preview", video, _wildcard_router()) is None


def test_video_model_group__is_none_without_a_router() -> None:
    assert video_model_group(_decoded("sora-2", "dep-sora-shared"), None) is None


def _video_list_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "seedance-video",
                "litellm_params": {"model": "volcengine/doubao-seedance-2-0-mini-260615", "api_key": "ark-key"},
                "model_info": {"id": "dep-seedance"},
            },
            {
                "model_name": "sora-video",
                "litellm_params": {"model": "openai/sora-2", "api_key": "openai-key"},
                "model_info": {"id": "dep-sora"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("custom_llm_provider", "deployment_id"), [("volcengine", "dep-seedance"), ("openai", "dep-sora")])
async def test_video_list_target_routes_a_model_less_list_to_the_providers_deployment(
    custom_llm_provider: str, deployment_id: str
) -> None:
    target: Final = await video_list_target(
        None, custom_llm_provider, _video_list_router(), UserAPIKeyAuth(api_key="hashed")
    )

    assert target == VideoListTarget(model=deployment_id, deployment_id=deployment_id)


@pytest.mark.asyncio
async def test_video_list_target_never_lends_a_deployment_the_key_cannot_call() -> None:
    restricted: Final = UserAPIKeyAuth(api_key="hashed", models=["sora-video"])

    assert await video_list_target(None, "volcengine", _video_list_router(), restricted) == VideoListTarget()
    assert await video_list_target(None, "openai", _video_list_router(), restricted) == VideoListTarget(
        model="dep-sora", deployment_id="dep-sora"
    )


def _team_video_list_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "team-seedance",
                "litellm_params": {"model": "volcengine/doubao-seedance-2-0-mini-260615", "api_key": "team-ark-key"},
                "model_info": {"id": "team-a-seedance", "team_id": "team-a", "team_public_model_name": "team-seedance"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("router", "team_id", "custom_llm_provider", "project_models", "expected"),
    [
        (_team_video_list_router(), "team-a", "volcengine", ["sora-video"], None),
        (_team_video_list_router(), "team-a", "volcengine", ["team-seedance"], "team-a-seedance"),
        (_video_list_router(), None, "volcengine", ["sora-video"], None),
        (_video_list_router(), None, "openai", ["sora-video"], "dep-sora"),
    ],
)
async def test_video_list_target_lends_a_key_without_a_model_list_only_what_its_project_allows(
    router: Router,
    team_id: str | None,
    custom_llm_provider: str,
    project_models: list[str],
    expected: str | None,
) -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=AuthObject.team("team-a"), value=LiteLLM_TeamTableCachedObj(team_id="team-a", models=["team-seedance"])
    )
    await cache.async_set_cache(
        key=AuthObject.project("proj-1"),
        value=LiteLLM_ProjectTableCachedObj(project_id="proj-1", models=project_models),
    )
    project_key: Final = UserAPIKeyAuth(
        api_key="hashed",
        team_id=team_id,
        team_models=["team-seedance"] if team_id else [],
        models=[],
        project_id="proj-1",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),  # test-quality-ok: proxy_server global, no seam
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),  # test-quality-ok: proxy_server global, no seam
    ):
        target: Final = await video_list_target(None, custom_llm_provider, router, project_key)

    assert target == VideoListTarget(model=expected, deployment_id=expected)


def _mixed_video_list_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "mixed-video",
                "litellm_params": {"model": "openai/sora-2", "api_key": "openai-key"},
                "model_info": {"id": "mixed-sora"},
            },
            {
                "model_name": "mixed-video",
                "litellm_params": {"model": "volcengine/doubao-seedance-2-0-mini-260615", "api_key": "ark-key"},
                "model_info": {"id": "mixed-seedance"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_model", "custom_llm_provider", "deployment_id"),
    [
        (None, "volcengine", "mixed-seedance"),
        (None, "openai", "mixed-sora"),
        ("mixed-video", "volcengine", "mixed-seedance"),
        ("mixed-video", "openai", "mixed-sora"),
    ],
)
async def test_video_list_target_finds_the_providers_deployment_anywhere_in_a_mixed_group(
    requested_model: str | None, custom_llm_provider: str, deployment_id: str
) -> None:
    key: Final = UserAPIKeyAuth(api_key="hashed", models=["mixed-video"])

    target: Final = await video_list_target(requested_model, custom_llm_provider, _mixed_video_list_router(), key)

    assert target == VideoListTarget(model=deployment_id, deployment_id=deployment_id)


def _two_seedance_groups_router(group_alias: dict[str, str]) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "seedance-a",
                "litellm_params": {"model": "volcengine/dreamina-seedance-2-5-260628", "api_key": "ark-key-a"},
                "model_info": {"id": "dep-a"},
            },
            {
                "model_name": "seedance-b",
                "litellm_params": {"model": "volcengine/dreamina-seedance-2-5-260628", "api_key": "ark-key-b"},
                "model_info": {"id": "dep-b"},
            },
        ],
        model_group_alias=group_alias,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_llm_provider", [None, "volcengine"])
@pytest.mark.parametrize(
    ("key", "router_settings", "group_alias"),
    [
        (UserAPIKeyAuth(api_key="hashed", aliases={"seedance-a": "seedance-b"}), None, {}),
        (UserAPIKeyAuth(api_key="hashed", team_model_aliases={"seedance-a": "seedance-b"}), None, {}),
        (UserAPIKeyAuth(api_key="hashed"), {"model_group_alias": {"seedance-a": "seedance-b"}}, {}),
        (UserAPIKeyAuth(api_key="hashed"), None, {"seedance-a": "seedance-b"}),
    ],
    ids=["key-alias", "team-alias", "key-or-team-router-settings", "router-group-alias"],
)
async def test_video_list_target_pins_the_group_the_callers_aliases_send_a_named_model_to(
    key: UserAPIKeyAuth,
    router_settings: dict[str, object] | None,
    group_alias: dict[str, str],
    custom_llm_provider: str | None,
) -> None:
    target: Final = await video_list_target(
        "seedance-a", custom_llm_provider, _two_seedance_groups_router(group_alias), key, router_settings
    )

    assert target == VideoListTarget(model="dep-b", deployment_id="dep-b")


@pytest.mark.asyncio
async def test_video_list_target_falls_back_to_a_config_deployment_the_project_allows_over_the_teams() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "team-seedance",
                "litellm_params": {"model": "volcengine/doubao-seedance-2-0-mini-260615", "api_key": "team-ark-key"},
                "model_info": {"id": "team-a-seedance", "team_id": "team-a", "team_public_model_name": "team-seedance"},
            },
            {
                "model_name": "seedance-video",
                "litellm_params": {"model": "volcengine/doubao-seedance-2-0-mini-260615", "api_key": "ark-key"},
                "model_info": {"id": "dep-seedance"},
            },
        ]
    )
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(key=AuthObject.team("team-a"), value=LiteLLM_TeamTableCachedObj(team_id="team-a"))
    await cache.async_set_cache(
        key=AuthObject.project("proj-1"),
        value=LiteLLM_ProjectTableCachedObj(project_id="proj-1", models=["seedance-video"]),
    )
    project_key: Final = UserAPIKeyAuth(api_key="hashed", team_id="team-a", models=[], project_id="proj-1")

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),  # test-quality-ok: proxy_server global, no seam
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),  # test-quality-ok: proxy_server global, no seam
    ):
        target: Final = await video_list_target(None, "volcengine", router, project_key)

    assert target == VideoListTarget(model="dep-seedance", deployment_id="dep-seedance")


def test_stamp_listed_video_ids_records_the_deployment_that_served_the_list() -> None:
    first_id: Final = encode_video_id_with_provider("cgt-1", "volcengine", "doubao-seedance-2-0-mini-260615")
    last_id: Final = encode_video_id_with_provider("cgt-2", "volcengine", "doubao-seedance-2-0-mini-260615")
    kept_id: Final = encode_video_id_with_provider("cgt-3", "volcengine", "doubao-seedance-2-0-mini-260615", deployment_id="dep-old")
    listed: Final = {
        "object": "list",
        "data": [{"id": first_id, "status": "succeeded"}, {"id": "cgt-raw"}, {"id": kept_id}, {"id": last_id}],
        "first_id": first_id,
        "last_id": last_id,
        "has_more": False,
    }

    stamped: Final = stamp_listed_video_ids(listed, "dep-b")

    assert [decode_video_id_with_provider(video["id"]) for video in stamped["data"]] == [
        decode_video_id_with_provider(
            encode_video_id_with_provider("cgt-1", "volcengine", "doubao-seedance-2-0-mini-260615", deployment_id="dep-b")
        ),
        decode_video_id_with_provider("cgt-raw"),
        decode_video_id_with_provider(kept_id),
        decode_video_id_with_provider(
            encode_video_id_with_provider("cgt-2", "volcengine", "doubao-seedance-2-0-mini-260615", deployment_id="dep-b")
        ),
    ]
    assert (stamped["first_id"], stamped["last_id"]) == (stamped["data"][0]["id"], stamped["data"][3]["id"])
    assert (stamped["object"], stamped["data"][0]["status"], stamped["has_more"]) == ("list", "succeeded", False)
    assert stamp_listed_video_ids(listed, None) == listed


@pytest.mark.asyncio
async def test_a_listed_video_resolves_to_the_group_that_listed_it_when_groups_share_an_upstream_model() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": f"seedance-{account}",
                "litellm_params": {"model": "volcengine/doubao-seedance-2-0-mini-260615", "api_key": f"ark-{account}"},
                "model_info": {"id": f"dep-{account}"},
            }
            for account in ("a", "b")
        ]
    )
    key_for_b: Final = UserAPIKeyAuth(api_key="hashed", models=["seedance-b"])
    upstream_listed_id: Final = encode_video_id_with_provider("cgt-b-1", "volcengine", "doubao-seedance-2-0-mini-260615")

    target: Final = await video_list_target(None, "volcengine", router, key_for_b)
    stamped: Final = stamp_listed_video_ids({"object": "list", "data": [{"id": upstream_listed_id}]}, target.deployment_id)
    listed_id: Final = stamped["data"][0]["id"]

    assert await authorized_video_model(decode_video_id_with_provider(listed_id), router, key_for_b) == "seedance-b"


@pytest.mark.asyncio
async def test_video_list_target_without_a_provider_or_model_leaves_routing_alone() -> None:
    assert await video_list_target(
        None, None, _video_list_router(), UserAPIKeyAuth(api_key="hashed")
    ) == VideoListTarget()


@pytest.mark.asyncio
async def test_video_list_target_routes_a_named_model_the_key_may_call() -> None:
    allowed: Final = UserAPIKeyAuth(api_key="hashed", models=["seedance-video"])

    assert await video_list_target("seedance-video", None, _video_list_router(), allowed) == VideoListTarget(
        model="dep-seedance", deployment_id="dep-seedance"
    )
    assert await video_list_target("seedance-video", "openai", _video_list_router(), allowed) == VideoListTarget(
        model="seedance-video"
    )


@pytest.mark.asyncio
async def test_video_list_target_refuses_a_named_model_the_key_cannot_call() -> None:
    restricted: Final = UserAPIKeyAuth(api_key="hashed", models=["sora-video"])

    with pytest.raises(ProxyException) as refused:
        await video_list_target("seedance-video", None, _video_list_router(), restricted)

    assert str(refused.value.code) == "403"
