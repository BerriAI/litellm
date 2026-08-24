from litellm.litellm_core_utils.llm_request_utils import serialize_multipart_form_fields


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
