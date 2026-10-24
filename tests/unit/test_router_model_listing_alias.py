"""
A model_group_alias listed by /v1/models reports the token limits of the group it
routes to. Before this, the listing lookup read only deployment names, so an alias
row carried no max_input_tokens while /model/info showed them, and Claude Code's
picker view could not mark a 1M-context alias with [1m].
"""

from typing import Final

from litellm import Router
from litellm.proxy.utils import create_model_info_response


def _router(model_group_alias: dict) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "long-context",
                "litellm_params": {"model": "openai/org/long-context-model", "api_key": "k"},
                "model_info": {"id": "lc", "max_input_tokens": 1_000_000, "max_output_tokens": 128_000},
            },
            {
                "model_name": "short",
                "litellm_params": {"model": "openai/org/short-model", "api_key": "k"},
                "model_info": {"id": "s", "max_input_tokens": 8192, "max_output_tokens": 4096},
            },
        ],
        model_group_alias=model_group_alias,
    )


def test_alias_reports_its_target_groups_limits() -> None:
    router: Final = _router({"big": "long-context", "item": {"model": "long-context", "hidden": False}})
    for alias in ("big", "item"):
        listing = router.get_model_listing_info(alias)
        assert listing is not None
        assert (listing.max_input_tokens, listing.max_output_tokens) == (1_000_000, 128_000)
        row = create_model_info_response(model_id=alias, provider="openai", llm_router=router)
        assert (row["id"], row.get("max_input_tokens"), row.get("max_output_tokens")) == (alias, 1_000_000, 128_000)


def test_a_deployment_name_wins_over_an_alias_of_the_same_name() -> None:
    router: Final = _router({"short": "long-context"})
    listing: Final = router.get_model_listing_info("short")
    assert listing is not None
    assert listing.max_input_tokens == 8192


def test_alias_to_an_unknown_group_and_unknown_names_stay_none() -> None:
    router: Final = _router({"dangling": "no-such-group"})
    assert router.get_model_listing_info("dangling") is None
    assert router.get_model_listing_info("not-listed") is None
