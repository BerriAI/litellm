from litellm.litellm_core_utils.llm_request_utils import (
    flatten_form_field_values,
    serialize_multipart_form_fields,
)


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
