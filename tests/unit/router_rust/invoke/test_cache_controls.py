import pytest

from litellm.router_rust.invoke.cache_controls import (
    ModelCacheControls,
    model_cache_controls,
    with_model_cache_controls,
)
from litellm.types.utils import is_litellm_owned_kwarg

_OFF = {"no-cache": True, "no-store": True}


@pytest.mark.parametrize(
    ("litellm_params", "request_cache", "expected"),
    [
        pytest.param({}, {"ttl": 5}, ModelCacheControls(None), id="no model rule keeps the request as sent"),
        pytest.param({"cache": _OFF}, None, ModelCacheControls(None), id="no request controls keeps the model rule"),
        pytest.param(
            {"cache": _OFF},
            {"ttl": 600},
            ModelCacheControls({"no-cache": True, "no-store": True, "ttl": 600}),
            id="a request ttl leaves the model's no-store in place",
        ),
        pytest.param(
            {"cache": {"ttl": 60}},
            {"ttl": 5},
            ModelCacheControls({"ttl": 5}),
            id="without a list the request wins the control it sends",
        ),
        pytest.param(
            {"cache": _OFF, "allowed_cache_controls": []},
            {"ttl": 600, "no-cache": False, "no-store": False},
            ModelCacheControls(_OFF, ("ttl", "no-cache", "no-store")),
            id="an empty list locks every control",
        ),
        pytest.param(
            {"cache": {"ttl": 60}, "allowed_cache_controls": ["no-cache", "no-store"]},
            {"ttl": 5, "no-cache": True},
            ModelCacheControls({"ttl": 60, "no-cache": True}, ("ttl",)),
            id="a listed control is kept and an unlisted one is ignored",
        ),
        pytest.param(
            {"cache": {"ttl": 60}, "allowed_cache_controls": ["ttl"]},
            {"ttl": 5},
            ModelCacheControls({"ttl": 5}),
            id="a listed ttl overrides the model's",
        ),
        pytest.param(
            {"allowed_cache_controls": []},
            {"s-maxage": 5, "namespace": "team-a"},
            ModelCacheControls({"namespace": "team-a"}, ("s-maxage",)),
            id="a lock without a model rule still ignores controls and keeps other cache fields",
        ),
        pytest.param(
            {"cache": {"s-max-age": 3600}},
            {"s-maxage": 1},
            ModelCacheControls({"s-maxage": 1}),
            id="a request s-maxage replaces the model's s-max-age spelling",
        ),
        pytest.param(
            {"cache": {"use-cache": False}, "allowed_cache_controls": []},
            {"use-cache": True},
            ModelCacheControls({"use-cache": False}, ("use-cache",)),
            id="a locked model ignores a request opting in to the cache",
        ),
        pytest.param(
            {"cache": {"use-cache": False}, "allowed_cache_controls": ["use-cache"]},
            {"use-cache": True},
            ModelCacheControls({"use-cache": True}),
            id="a listed use-cache lets a request opt in",
        ),
        pytest.param(
            {"cache": {"s-maxage": 1}, "allowed_cache_controls": []},
            {"s-max-age": 3600},
            ModelCacheControls({"s-maxage": 1}, ("s-max-age",)),
            id="the s-max-age spelling is locked with s-maxage",
        ),
        pytest.param(
            {"cache": {"s-maxage": 1}, "allowed_cache_controls": ["s-maxage"]},
            {"s-max-age": 3600},
            ModelCacheControls({"s-maxage": 3600}),
            id="the s-max-age spelling is allowed with s-maxage",
        ),
        pytest.param(
            {"cache": {"ttl": 60}, "allowed_cache_controls": []},
            None,
            ModelCacheControls({"ttl": 60}),
            id="a locked model with no request controls reports nothing ignored",
        ),
    ],
)
def test_model_cache_controls_merge_per_control(
    litellm_params: dict[str, object], request_cache: object, expected: ModelCacheControls
) -> None:
    assert model_cache_controls(litellm_params, request_cache) == expected


def test_allowed_cache_controls_is_not_sent_to_the_provider() -> None:
    assert is_litellm_owned_kwarg("allowed_cache_controls")


def test_a_null_list_leaves_the_model_cache_controls_and_the_request_as_they_are() -> None:
    request = {"model": "a"}

    params, merged, ignored = with_model_cache_controls(
        {"model": "openai/a", "cache": {"no-store": True}, "allowed_cache_controls": None}, request
    )

    assert params == {"model": "openai/a", "cache": {"no-store": True}}
    assert merged is request
    assert ignored == ()


def test_either_spelling_of_s_maxage_in_a_lock_and_a_fallback_list_is_the_same_control() -> None:
    from litellm.router_rust.invoke.cache_controls import with_cache_lock

    locked = with_cache_lock(
        {"model": "openai/b", "allowed_cache_controls": ["s-maxage", "ttl"]},
        {"cache": {"ttl": 60}, "allowed_cache_controls": ["s-max-age", "no-cache"]},
    )

    assert locked == {"model": "openai/b", "cache": {"ttl": 60}, "allowed_cache_controls": ["s-maxage"]}


@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
def test_a_control_the_virtual_key_enforces_survives_a_model_list_that_leaves_it_out(bucket: str) -> None:
    request = {
        "cache": {"no-store": True, "use-cache": True, "ttl": 5},
        bucket: {"user_api_key_metadata": {"cache": {"no-store": True}}},
    }

    _, merged, ignored = with_model_cache_controls(
        {"cache": {"ttl": 60}, "allowed_cache_controls": ["use-cache"]}, request
    )

    assert merged["cache"] == {"ttl": 60, "no-store": True, "use-cache": True}
    assert ignored == ("ttl",)


def test_a_fallback_with_its_own_list_keeps_the_strictest_of_its_lock_and_the_requested_models() -> None:
    from litellm.router_rust.invoke.cache_controls import with_cache_lock

    locked = with_cache_lock(
        {"model": "openai/b", "cache": {"no-store": True, "ttl": 60}, "allowed_cache_controls": []},
        {"cache": {"no-store": False, "ttl": 3600, "s-maxage": None}, "allowed_cache_controls": []},
    )

    assert locked == {
        "model": "openai/b",
        "cache": {"no-store": True, "ttl": 60, "s-maxage": None},
        "allowed_cache_controls": [],
    }


def test_the_strictest_lock_keeps_a_lifetime_that_another_lock_leaves_null() -> None:
    from litellm.router_rust.invoke.cache_controls import strictest_cache_lock

    locks = [
        {"cache": {"ttl": None}, "allowed_cache_controls": []},
        {"cache": {"ttl": 60}, "allowed_cache_controls": []},
    ]

    assert strictest_cache_lock(locks) == {"cache": {"ttl": 60}, "allowed_cache_controls": []}
    assert strictest_cache_lock(locks[::-1]) == {"cache": {"ttl": 60}, "allowed_cache_controls": []}


def test_fallback_params_cannot_pose_as_the_virtual_keys_cache_controls() -> None:
    forged = {"user_api_key_metadata": {"cache": {"no-store": False}}}
    attempt = {"cache": {"no-store": False}, "litellm_metadata": forged, "metadata": forged}

    _, merged, ignored = with_model_cache_controls(
        {"cache": {"no-store": True}, "allowed_cache_controls": []}, attempt, authenticated={"metadata": {}}
    )

    assert merged["cache"] == {"no-store": True}
    assert ignored == ("no-store",)


def test_a_key_control_the_attempt_changed_is_no_longer_the_keys() -> None:
    key = {"metadata": {"user_api_key_metadata": {"cache": {"no-store": True}}}}

    _, merged, ignored = with_model_cache_controls(
        {"cache": {"ttl": 60}, "allowed_cache_controls": []}, {**key, "cache": {"no-store": False}}, authenticated=key
    )

    assert merged["cache"] == {"ttl": 60}
    assert ignored == ("no-store",)


def test_a_number_standing_in_for_the_keys_boolean_is_not_the_keys_control() -> None:
    key = {"metadata": {"user_api_key_metadata": {"cache": {"no-store": True}}}}

    _, merged, ignored = with_model_cache_controls(
        {"cache": {"no-store": True}, "allowed_cache_controls": []},
        {**key, "cache": {"no-store": 1}},
        authenticated=key,
    )

    assert merged["cache"]["no-store"] is True
    assert ignored == ("no-store",)
