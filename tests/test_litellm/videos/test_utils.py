"""
Pure-logic contract tests for litellm/videos/main.py's request utils
(litellm/videos/utils.py: VideoGenerationRequestUtils).

These lock the exact param-shaping behavior so a mutation that drops a filter,
flips a precedence, or stops removing a key fails loudly. The only seam is the
provider config's map_openai_params (a provider boundary); filter_out_litellm_params
runs for real, so the "litellm-internal params get stripped" assertions reflect
production. Every test asserts the exact resulting dict, never "ran without error".
"""

import os
import sys
from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.videos.utils import VideoGenerationRequestUtils

get_requested = (
    VideoGenerationRequestUtils.get_requested_video_generation_optional_param
)
get_optional = VideoGenerationRequestUtils.get_optional_params_video_generation


# =========================================================================== #
# get_requested_video_generation_optional_param
#
# Receives the caller's full local_vars; must return only the API-bound optional
# params. filter_out_litellm_params strips known internal keys for real; the
# values used below were chosen against the live set: seconds/size/user/foo_param/
# vertex_project/extra/a/b survive, api_key/metadata/litellm_* are stripped.
# =========================================================================== #


def test_requested__drops_none_and_excluded_keys():
    result = get_requested(
        {
            "seconds": "8",
            "size": None,  # None -> dropped
            "prompt": "a sunset",  # excluded
            "model": "sora-2",  # excluded
            "user": "u1",
        }
    )
    assert result == {"seconds": "8", "user": "u1"}


def test_requested__strips_litellm_internal_params():
    result = get_requested(
        {
            "seconds": "8",
            "api_key": "sk-secret",
            "metadata": {"x": 1},
            "litellm_call_id": "id-123",
        }
    )
    assert result == {"seconds": "8"}


def test_requested__timeout_always_removed():
    # timeout is NOT a litellm-internal param, so only the explicit pop removes it.
    result = get_requested({"seconds": "8", "timeout": 30})
    assert result == {"seconds": "8"}


def test_requested__nested_kwargs_merge_and_override_base():
    result = get_requested(
        {"seconds": "8", "kwargs": {"size": "720x1280", "seconds": "override"}}
    )
    # nested kwargs win over the top-level base params on collision.
    assert result == {"seconds": "override", "size": "720x1280"}


def test_requested__non_dict_kwargs_treated_as_empty():
    result = get_requested({"seconds": "8", "kwargs": "not-a-dict"})
    assert result == {"seconds": "8"}


def test_requested__none_input_returns_empty():
    assert get_requested(None) == {}


def test_requested__top_level_extra_body_spread_and_preserved():
    result = get_requested(
        {"seconds": "8", "extra_body": {"vertex_project": "proj", "foo_param": "bar"}}
    )
    # extra_body keys are both spread at top level AND kept under "extra_body".
    assert result == {
        "seconds": "8",
        "vertex_project": "proj",
        "foo_param": "bar",
        "extra_body": {"vertex_project": "proj", "foo_param": "bar"},
    }


def test_requested__extra_body_kwargs_overrides_top_level():
    result = get_requested(
        {
            "extra_body": {"a": "top", "b": "top_b"},
            "kwargs": {"extra_body": {"a": "kw"}},
        }
    )
    # kwargs' extra_body wins over the top-level extra_body on collision; the
    # non-colliding top-level key survives.
    assert result == {
        "a": "kw",
        "b": "top_b",
        "extra_body": {"a": "kw", "b": "top_b"},
    }


def test_requested__extra_body_strips_litellm_internal_params():
    result = get_requested({"extra_body": {"api_key": "sk", "foo_param": "bar"}})
    # api_key filtered out of extra_body; only foo_param remains (and is spread).
    assert result == {"foo_param": "bar", "extra_body": {"foo_param": "bar"}}


def test_requested__empty_extra_body_not_added():
    result = get_requested({"seconds": "8", "extra_body": {}})
    assert result == {"seconds": "8"}
    assert "extra_body" not in result


# =========================================================================== #
# get_optional_params_video_generation
#
# Delegates mapping to the provider config (the seam) then folds extra_body in.
# =========================================================================== #


def _config(map_return):
    config = MagicMock()
    config.map_openai_params.return_value = map_return
    return config


def test_optional__delegates_to_map_openai_params_with_drop_params():
    config = _config({"seconds": "8"})
    optional_params = {"seconds": "8"}

    result = get_optional(
        model="sora-2",
        video_generation_provider_config=config,
        video_generation_optional_params=optional_params,
    )

    assert result == {"seconds": "8"}
    config.map_openai_params.assert_called_once_with(
        video_create_optional_params=optional_params,
        model="sora-2",
        drop_params=litellm.drop_params,
    )


def test_optional__extra_body_overrides_mapped_and_is_removed():
    # mapped output carries a leftover extra_body that must be popped; the input
    # extra_body overrides a colliding mapped key and is spread in.
    config = _config({"seconds": "8", "size": "mapped", "extra_body": {"leftover": 1}})

    result = get_optional(
        model="sora-2",
        video_generation_provider_config=config,
        video_generation_optional_params={
            "extra_body": {"size": "override", "extra": "x"}
        },
    )

    assert result == {"seconds": "8", "size": "override", "extra": "x"}
    assert "extra_body" not in result


def test_optional__no_extra_body_returns_mapped_unchanged():
    config = _config({"seconds": "8"})

    result = get_optional(
        model="sora-2",
        video_generation_provider_config=config,
        video_generation_optional_params={"seconds": "8"},
    )

    assert result == {"seconds": "8"}


def test_optional__non_dict_extra_body_ignored():
    config = _config({"seconds": "8"})

    result = get_optional(
        model="sora-2",
        video_generation_provider_config=config,
        video_generation_optional_params={"seconds": "8", "extra_body": None},
    )

    assert result == {"seconds": "8"}
