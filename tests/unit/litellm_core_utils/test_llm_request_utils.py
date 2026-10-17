import httpx
import pytest

from litellm.litellm_core_utils.llm_request_utils import (
    ensure_extra_body_is_safe,
    flatten_form_field_values,
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
