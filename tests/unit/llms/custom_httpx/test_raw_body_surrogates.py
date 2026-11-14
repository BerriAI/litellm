import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import _prepare_request_data_and_content


@pytest.mark.parametrize("argument", ["data", "content"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("hi\ud800!", "hi\ufffd!"), ("\udc00", "\ufffd"), ("\ud83d\ude00", "😀"), ("漢字😀", "漢字😀")],
)
def test_raw_text_can_be_encoded_as_utf8(argument, raw, expected):
    data, content = _prepare_request_data_and_content(**{argument: raw})
    request = httpx.Request("POST", "http://localhost/", data=data, content=content)
    assert request.content == expected.encode()


def test_valid_text_and_bytes_keep_identity():
    text = "valid 😀"
    payload = b"\xed\xa0\x80"
    assert _prepare_request_data_and_content(data=text) == (None, text)
    assert _prepare_request_data_and_content(content=text)[1] is text
    assert _prepare_request_data_and_content(data=payload)[1] is payload


def test_explicit_content_wins_without_mutating_form_fields():
    fields = {"field": "value"}
    data, content = _prepare_request_data_and_content(data=fields, content="\ud800")
    assert data is fields
    assert content == "\ufffd"
    assert _prepare_request_data_and_content(data="ignored\ud800", content=b"chosen") == (None, b"chosen")
