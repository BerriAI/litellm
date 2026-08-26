import httpx

from litellm.litellm_core_utils.llm_request_utils import (
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
    assert flatten_form_field_values(
        {"loras": ["a", "b", "c"], "generation_config": {"tags": [1, 2]}, "seed": 42}
    ) == (
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
