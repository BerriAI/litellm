from collections.abc import Mapping
from typing import Final, Literal

import httpx
import pytest
import respx

import litellm
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
_API_BASE: Final = "https://api.openai.com/v1"
_ITEMS_URL: Final = f"{_API_BASE}/responses/resp_sdk/input_items"
_EMPTY_PAGE: Final = {"object": "list", "data": [], "has_more": False}


def _list_items(
    extra_query: Mapping[str, object], drop_params: bool, order: Literal["asc", "desc"] = "desc"
) -> tuple[tuple[tuple[str, str], ...], int]:
    with respx.mock() as mock_router:
        route: Final = mock_router.get(_ITEMS_URL).mock(return_value=httpx.Response(200, json=_EMPTY_PAGE))
        litellm.list_input_items(
            response_id="resp_sdk",
            custom_llm_provider="openai",
            api_key="sk-test",
            api_base=_API_BASE,
            order=order,
            extra_query=extra_query,
            drop_params=drop_params,
        )
    return tuple(sorted(route.calls.last.request.url.params.multi_items())), route.call_count


def test_sdk_extra_query_takes_typed_values_and_forwards_them() -> None:
    forwarded: Final = _list_items({"limit": 5, "order": "desc", "include": ("message.output_text.logprobs",)}, False)

    assert forwarded == (
        (("include[]", "message.output_text.logprobs"), ("limit", "5"), ("order", "desc")),
        1,
    )


def test_sdk_extra_query_under_drop_params_falls_back_to_the_callers_own_arguments() -> None:
    forwarded: Final = _list_items(
        {"after": ["item_a", "item_b"], "order": "sideways", "include": [1], "limit": 3},
        True,
        order="asc",
    )

    assert forwarded == ((("limit", "3"), ("order", "asc")), 1)


@pytest.mark.parametrize(
    ("extra_query", "param"),
    (
        ({"include": ["message.output_text.logprobs", 1]}, "include"),
        ({"include[]": {"name": "x"}}, "include"),
        ({"limit": True}, "limit"),
        ({"limit": -1}, "limit"),
    ),
)
def test_sdk_extra_query_with_a_malformed_value_is_a_400_before_any_upstream_call(
    extra_query: Mapping[str, object], param: str
) -> None:
    with respx.mock(assert_all_called=False) as mock_router:
        route: Final = mock_router.get(_ITEMS_URL).mock(return_value=httpx.Response(200, json=_EMPTY_PAGE))
        with pytest.raises(litellm.BadRequestError) as exc_info:
            litellm.list_input_items(
                response_id="resp_sdk",
                custom_llm_provider="openai",
                api_key="sk-test",
                api_base=_API_BASE,
                extra_query=extra_query,
            )

    assert exc_info.value.status_code == 400
    assert f"Invalid '{param}'" in str(exc_info.value)
    assert route.call_count == 0
