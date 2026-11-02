"""
Settings and configs both cores must treat as ``PythonCore`` does (oddities included): count and
timeout settings kept as given, ``default_fallbacks=[]``, malformed model lists that fail at
construction, and deprecated registry members called with a ``None`` name.
"""

import warnings
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.router import Router
from litellm.router_core import RouterCore
from tests.unit.test_router.core.retries_health_helpers import MakeCore
from tests.unit.test_router.core.scripted_host import deployment


def as_router(core: RouterCore) -> Router:
    router: Final = Router.__new__(Router)
    router._attach(core)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]  # the facade's own attach, as from_config calls it
    return router


def read(router: Router, name: str) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return getattr(router, name)  # pyright: ignore[reportAny]  # a deprecated attribute, as given


# --- settings kept as given -------------------------------------------------------------------------


def test_retry_after_and_an_int_timeout_stay_ints(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")], timeout=1))
    assert (repr(read(router, "retry_after")), repr(read(router, "timeout"))) == ("0", "1")
    settings: Final = router.get_settings()
    assert (repr(settings["retry_after"]), repr(settings["timeout"])) == ("0", "1")


def test_update_settings_casts_timeout_and_retry_after_to_int(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    router.update_settings(timeout=3.7, retry_after=2.2)
    assert (repr(read(router, "timeout")), repr(read(router, "retry_after"))) == ("3", "2")


@pytest.mark.parametrize("name", ["num_retries", "allowed_fails", "max_fallbacks"])
def test_a_count_setting_that_is_no_int_is_kept_as_given(make_core: MakeCore, name: str) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")], **{name: "1e-05"}))
    assert read(router, name) == "1e-05"


# --- default_fallbacks ------------------------------------------------------------------------------


def test_an_empty_default_fallbacks_adds_a_star_entry_with_no_models(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")], default_fallbacks=[]))
    assert read(router, "fallbacks") == [{"*": None}]


# --- malformed model lists --------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", [None, 5])
def test_a_model_name_that_is_no_string_fails_at_construction(make_core: MakeCore, model_name: object) -> None:
    entry: Final = {"model_name": model_name, "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"}}
    with pytest.raises(TypeError, match="expected str instance"):
        make_core(model_list=[entry])
    with pytest.raises(ValidationError, match="model_name"):
        make_core(model_list=[{**entry, "model_info": {"id": "x"}}])


def test_an_alias_entry_without_hidden_fails_at_construction(make_core: MakeCore) -> None:
    with pytest.raises(KeyError, match="hidden"):
        make_core(model_list=[deployment("gpt", "a")], model_group_alias={"al": {"model": "gpt"}})


# --- deprecated registry members with a None name -----------------------------------------------------


@pytest.mark.parametrize(
    ("member", "arguments", "expected"),
    [
        ("get_model_group", (None,), None),
        ("resolve_served_model_name", (None, {}), None),
        ("get_model_from_alias", (None,), None),
        ("get_routable_upstream_model", (None,), None),
        ("get_candidate_model_ids_for_route", (None,), frozenset()),
    ],
)
def test_a_deprecated_member_called_with_no_name_returns_the_old_empty_result(
    make_core: MakeCore, member: str, arguments: tuple[object, ...], expected: object
) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert getattr(router, member)(*arguments) == expected  # pyright: ignore[reportAny]  # the deprecated member


@pytest.mark.asyncio
async def test_remaining_usage_of_no_group_is_empty(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a")]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert await router.get_remaining_model_group_usage(None) == {}  # pyright: ignore[reportArgumentType]  # a None group, as the fuzzer passes it
