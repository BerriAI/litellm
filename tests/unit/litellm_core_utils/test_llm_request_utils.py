import datetime
import time
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.llm_request_utils import (
    ensure_extra_body_is_safe,
    flatten_form_field_values,
    pick_cheapest_chat_models_from_llm_provider,
    rank_chat_models_from_llm_provider,
    serialize_multipart_form_fields,
)


def _multipart_field_names(data: dict) -> list[str]:
    request = httpx.Request(
        "POST",
        "http://backend/v1/images/edits",
        data=data,
        files=[("image[]", ("in.png", b"stub", "image/png"))],
    )
    request.read()
    body = request.content.decode("utf-8", "replace")
    prefix = 'Content-Disposition: form-data; name="'
    return [line[len(prefix) : line.index('"', len(prefix))] for line in body.splitlines() if line.startswith(prefix)]


def test_serialize_multipart_form_fields_flattens_like_the_openai_sdk():
    fields = serialize_multipart_form_fields(
        {
            "model": "sora-2",
            "prompt": "a cat surfing",
            "hd": True,
            "watermark": False,
            "seconds": 4,
            "size": None,
            "metadata": {"trace": {"id": "t1"}},
            "characters": [{"id": "char_1", "name": "Mia"}, "solo"],
        }
    )

    assert fields == (
        ("model", (None, "sora-2")),
        ("prompt", (None, "a cat surfing")),
        ("hd", (None, "true")),
        ("watermark", (None, "false")),
        ("seconds", (None, "4")),
        ("metadata[trace][id]", (None, "t1")),
        ("characters[][id]", (None, "char_1")),
        ("characters[][name]", (None, "Mia")),
        ("characters[]", (None, "solo")),
    )


def test_serialize_multipart_form_fields_drops_empty_strings():
    assert serialize_multipart_form_fields({"prompt": "", "model": "sora-2"}) == (("model", (None, "sora-2")),)


def test_serialize_multipart_form_fields_empty_body():
    assert serialize_multipart_form_fields({}) == ()


def test_flatten_form_field_values_flattens_nested_and_drops_empty():
    assert flatten_form_field_values(
        {
            "seed": 42,
            "hd": True,
            "size": None,
            "prompt": "",
            "generation_config": {"steps": 30, "guidance": True},
        }
    ) == (
        ("seed", "42"),
        ("hd", "true"),
        ("generation_config[steps]", "30"),
        ("generation_config[guidance]", "true"),
    )


def test_flatten_form_field_values_later_source_wins_on_collision():
    assert flatten_form_field_values({"seed": 1}, None, {"seed": 2}) == (
        ("seed", "1"),
        ("seed", "2"),
    )
    assert dict(flatten_form_field_values({"seed": 1}, {"seed": 2}))["seed"] == "2"


def test_flatten_form_field_values_keeps_scalar_lists_as_repeated_fields():
    assert flatten_form_field_values({"loras": ["a", "b", "c"], "generation_config": {"tags": [1, 2]}, "seed": 42}) == (
        ("loras", ("a", "b", "c")),
        ("generation_config[tags]", ("1", "2")),
        ("seed", "42"),
    )


def test_flatten_form_field_values_scalar_list_survives_update_into_multipart():
    request_params: dict = {"model": "my-edit-model"}
    request_params.update(flatten_form_field_values({"loras": ["style_a", "style_b"]}))

    names = _multipart_field_names(request_params)

    assert names.count("loras") == 2
    assert names.count("model") == 1


def test_flatten_form_field_values_rejects_over_deep_nesting():
    nested: object = "leaf"
    for _ in range(102):
        nested = {"k": nested}
    assert isinstance(nested, dict)
    with pytest.raises(ValueError, match="max depth"):
        flatten_form_field_values(nested)


def test_ensure_extra_body_is_safe_converts_prompt_without_filtering_metadata_keys():
    from typing import cast

    class Prompt:
        pass

    prompt = Prompt()
    metadata: dict[object, object] = {"prompt": prompt, 1: "retained"}
    extra_body = cast(dict[str, object], {"metadata": metadata})

    result = ensure_extra_body_is_safe(extra_body)

    assert result is extra_body
    assert metadata == {"prompt": prompt.__dict__, 1: "retained"}


def test_ensure_extra_body_is_safe_returns_non_dict_unchanged():
    from collections import UserDict
    from typing import cast

    extra_body = UserDict({"metadata": {"prompt": object()}})

    result = ensure_extra_body_is_safe(cast(dict[str, object], extra_body))

    assert result is extra_body


PICKER_PROVIDER: Final = "deepseek"


@pytest.fixture
def deepseek_rows_whose_cheapest_ids_route_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: Final = {
        "deepseek/fake-pricey-chat": {"input_cost_per_token": 3e-12, "output_cost_per_token": 3e-12, "mode": "chat"},
        "deepseek/fake-cheapest-embedding": {"input_cost_per_token": 1e-13, "mode": "embedding"},
        "openai/fake-free-chat": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "mode": "chat"},
        "deepseek/fake-cheap-chat": {"input_cost_per_token": 1e-12, "output_cost_per_token": 1e-12, "mode": "chat"},
        "deepseek/fake-retired-chat": {
            "input_cost_per_token": 1e-14,
            "output_cost_per_token": 1e-14,
            "mode": "chat",
            "deprecation_date": "2000-01-01",
        },
        "deepseek/fake-retiring-chat": {
            "input_cost_per_token": 2e-12,
            "output_cost_per_token": 2e-12,
            "mode": "chat",
            "deprecation_date": "2999-01-01",
        },
        "deepseek/fake-misdated-chat": {
            "input_cost_per_token": 2.5e-12,
            "output_cost_per_token": 2.5e-12,
            "mode": "chat",
            "deprecation_date": "next spring",
        },
        "fake-free-chat-without-provider": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "mode": "chat"},
    }
    for name, row in rows.items():
        monkeypatch.setitem(litellm.model_cost, name, {**row, "litellm_provider": PICKER_PROVIDER})
    monkeypatch.setitem(litellm.models_by_provider, PICKER_PROVIDER, tuple(rows))


def test_pick_cheapest_chat_models_prefers_provider_routed_ids_skips_ids_routed_elsewhere_and_keeps_retired_ones(
    deepseek_rows_whose_cheapest_ids_route_elsewhere: None,
) -> None:
    assert pick_cheapest_chat_models_from_llm_provider(PICKER_PROVIDER, n=10) == [
        "deepseek/fake-retired-chat",
        "deepseek/fake-cheap-chat",
        "deepseek/fake-retiring-chat",
        "deepseek/fake-misdated-chat",
        "deepseek/fake-pricey-chat",
        "fake-free-chat-without-provider",
    ]
    assert pick_cheapest_chat_models_from_llm_provider(PICKER_PROVIDER, n=1) == ["deepseek/fake-retired-chat"]


def test_rank_chat_models_flags_only_the_rows_whose_cost_map_deprecation_date_has_passed(
    deepseek_rows_whose_cheapest_ids_route_elsewhere: None,
) -> None:
    assert {
        ranked_model.model: ranked_model.is_past_deprecation_date
        for ranked_model in rank_chat_models_from_llm_provider(PICKER_PROVIDER)
    } == {
        "deepseek/fake-retired-chat": True,
        "deepseek/fake-cheap-chat": False,
        "deepseek/fake-retiring-chat": False,
        "deepseek/fake-misdated-chat": False,
        "deepseek/fake-pricey-chat": False,
        "fake-free-chat-without-provider": False,
    }


@pytest.mark.parametrize(
    "listed_order",
    [("deepseek/fake-tie-b", "deepseek/fake-tie-a"), ("deepseek/fake-tie-a", "deepseek/fake-tie-b")],
)
def test_pick_cheapest_chat_models_keeps_the_provider_listing_order_for_equal_price_models(
    monkeypatch: pytest.MonkeyPatch, listed_order: tuple[str, str]
) -> None:
    for name in listed_order:
        monkeypatch.setitem(
            litellm.model_cost,
            name,
            {
                "input_cost_per_token": 1e-12,
                "output_cost_per_token": 1e-12,
                "mode": "chat",
                "litellm_provider": PICKER_PROVIDER,
            },
        )
    monkeypatch.setitem(litellm.models_by_provider, PICKER_PROVIDER, listed_order)

    assert pick_cheapest_chat_models_from_llm_provider(PICKER_PROVIDER, n=2) == list(listed_order)


def test_rank_chat_models_flags_a_retired_model_the_provider_lists_without_its_cost_map_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: Final = {
        "vertex_ai/fake-retired-vertex-chat": {"input_cost_per_token": 1e-14, "deprecation_date": "2000-01-01"},
        "vertex_ai/fake-live-vertex-chat": {"input_cost_per_token": 1e-12, "deprecation_date": "2999-01-01"},
    }
    for name, row in rows.items():
        monkeypatch.setitem(litellm.model_cost, name, {**row, "mode": "chat", "litellm_provider": "vertex_ai"})
    monkeypatch.setitem(
        litellm.models_by_provider, "vertex_ai", tuple(name.removeprefix("vertex_ai/") for name in rows)
    )

    assert [
        (ranked_model.model, ranked_model.is_past_deprecation_date)
        for ranked_model in rank_chat_models_from_llm_provider("vertex_ai")
    ] == [("fake-retired-vertex-chat", True), ("fake-live-vertex-chat", False)]


@pytest.mark.parametrize("host_zone", ["Etc/GMT+12", "Etc/GMT-14"])
def test_rank_chat_models_flags_retirement_by_the_utc_date_whatever_the_host_zone(
    monkeypatch: pytest.MonkeyPatch, host_zone: str
) -> None:
    utc_today: Final = datetime.datetime.now(datetime.timezone.utc).date()
    retirement_dates: Final = {
        "deepseek/fake-retires-today-chat": utc_today,
        "deepseek/fake-retires-tomorrow-chat": utc_today + datetime.timedelta(days=1),
    }
    for name, retirement_date in retirement_dates.items():
        monkeypatch.setitem(
            litellm.model_cost,
            name,
            {
                "input_cost_per_token": 1e-12,
                "output_cost_per_token": 1e-12,
                "mode": "chat",
                "litellm_provider": PICKER_PROVIDER,
                "deprecation_date": retirement_date.isoformat(),
            },
        )
    monkeypatch.setitem(litellm.models_by_provider, PICKER_PROVIDER, tuple(retirement_dates))
    monkeypatch.setenv("TZ", host_zone)
    time.tzset()
    try:
        flags: Final = {
            ranked_model.model: ranked_model.is_past_deprecation_date
            for ranked_model in rank_chat_models_from_llm_provider(PICKER_PROVIDER)
        }
    finally:
        monkeypatch.undo()
        time.tzset()

    assert flags == {"deepseek/fake-retires-today-chat": True, "deepseek/fake-retires-tomorrow-chat": False}
