"""ClinePass must not be reachable on endpoints it does not implement.

ClinePass implements chat completions only. Before this guard existed, listing
it in `litellm.openai_compatible_providers` made the speech, transcription and
image-generation branches in litellm match it. Those branches send the request
to the provider's own `api_base` but read the credential from `OPENAI_API_KEY`,
so a `litellm.speech(model="clinepass/...")` call POSTed the caller's OpenAI key
to the Cline host -- for an endpoint that does not exist there.

Asserting "not in the list" is a structural check and lives with the other
registry tests. These tests assert the behaviour instead: that no HTTP request
leaves the process at all, and that the OpenAI credential is never transmitted.
"""

import contextlib

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

SENTINEL_OPENAI_KEY = "sk-sentinel-openai-key-must-never-be-transmitted"


@pytest.fixture
def no_request_allowed(monkeypatch):
    """Fail loudly if anything attempts an outbound request, and record it.

    Patched at the httpx transport layer rather than at litellm's handlers, so a
    future dispatch path that bypasses HTTPHandler is still caught.
    """
    attempted: list[tuple[str, str]] = []

    def record_and_block(self, request, *args, **kwargs):
        attempted.append((str(request.url), request.headers.get("authorization", "")))
        raise AssertionError(f"outbound request attempted to {request.url}")

    monkeypatch.setattr(httpx.Client, "send", record_and_block, raising=True)
    monkeypatch.setattr(httpx.AsyncClient, "send", record_and_block, raising=True)
    for handler in (HTTPHandler, AsyncHTTPHandler):
        monkeypatch.setattr(handler, "post", record_and_block, raising=True)

    monkeypatch.setenv("OPENAI_API_KEY", SENTINEL_OPENAI_KEY)
    monkeypatch.setenv("CLINEPASS_API_KEY", "cp-test-key")
    return attempted


def test_speech_makes_no_outbound_request(no_request_allowed):
    # Which exception litellm raises for an unsupported endpoint is its business
    # and may change; that nothing is transmitted is the contract under test. The
    # fixture's AssertionError means the network WAS reached, so it must escape
    # rather than be swallowed as "some exception happened".
    try:
        litellm.speech(model="clinepass/deepseek-v4-flash", input="hi", voice="alloy")
    except AssertionError:
        raise
    except Exception:  # noqa: S110 - deliberate; see above
        pass

    assert no_request_allowed == []


def test_transcription_makes_no_outbound_request(no_request_allowed, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00\x00")

    with open(audio, "rb") as handle:
        try:
            litellm.transcription(model="clinepass/deepseek-v4-flash", file=handle)
        except AssertionError:
            raise
        except Exception:  # noqa: S110 - deliberate; see test_speech_makes_no_outbound_request
            pass

    assert no_request_allowed == []


def test_image_generation_makes_no_outbound_request(no_request_allowed):
    """Image generation must not reach the Cline host.

    Note it does not raise either: litellm returns an empty `ImageResponse` for
    any provider with no image support. That is pre-existing upstream behaviour,
    not a ClinePass quirk -- `mistral`, which has the same shape ClinePass now
    has (own module, absent from `openai_compatible_providers`, explicitly
    registered for exception mapping), returns the same empty response with zero
    outbound requests. So this test asserts the property that is ours to keep:
    nothing is transmitted.
    """
    litellm.image_generation(model="clinepass/deepseek-v4-flash", prompt="a cat")

    assert no_request_allowed == []


def test_openai_credential_is_never_transmitted(no_request_allowed):
    """The point of the P1: whatever happens, the OpenAI key must not go out."""
    for call in (
        lambda: litellm.speech(model="clinepass/deepseek-v4-flash", input="hi", voice="alloy"),
        lambda: litellm.transcription(model="clinepass/deepseek-v4-flash", file=None),
        lambda: litellm.image_generation(model="clinepass/deepseek-v4-flash", prompt="a cat"),
    ):
        # Whether each endpoint raises or returns an empty response is upstream's
        # business; that no credential leaves the process is ours.
        with contextlib.suppress(Exception):
            call()

    leaked = [url for url, auth in no_request_allowed if SENTINEL_OPENAI_KEY in auth]
    assert leaked == [], f"OPENAI_API_KEY was transmitted to {leaked}"


def test_unknown_kwargs_are_flattened_not_wrapped_in_extra_body():
    """`extra_body` is an OpenAI-SDK concept the SDK unwraps client-side.

    ClinePass dispatches through `BaseLLMHTTPHandler`, which serialises optional
    params directly into the JSON body, so an `extra_body` key would be sent to
    the API verbatim and a genuine vendor kwarg would arrive nested one level
    too deep.
    """
    params = litellm.utils.get_optional_params(
        model="cline-pass/deepseek-v4-flash",
        custom_llm_provider="clinepass",
        temperature=0.5,
        some_vendor_knob=7,
    )

    assert "extra_body" not in params
    assert params["some_vendor_knob"] == 7
