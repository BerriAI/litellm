from typing import Final

import pytest
import respx
from httpx import Response

import litellm
from litellm import atext_completion, text_completion

COMPLETIONS_URL: Final = (
    "https://example-resource.openai.azure.com/openai/deployments/gpt-35-turbo-instruct/completions"
)
AZURE_TEXT_CALL: Final = {
    "model": "azure_text/gpt-35-turbo-instruct",
    "prompt": "hello",
    "api_base": "https://example-resource.openai.azure.com",
    "api_version": "2024-02-01",
    "api_key": "azure-test-key",
}


@pytest.fixture
def rejected_completions_endpoint():
    return respx.post(url__startswith=COMPLETIONS_URL).mock(
        return_value=Response(
            400,
            json={"error": {"message": "bad prompt", "code": "invalid_prompt"}},
            headers={"x-request-id": "req-1"},
        )
    )


@respx.mock
def test_completion_surfaces_the_status_and_headers_of_a_rejected_request(rejected_completions_endpoint):
    with pytest.raises(litellm.BadRequestError) as rejected:
        text_completion(**AZURE_TEXT_CALL)

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
async def test_acompletion_surfaces_the_status_and_headers_of_a_rejected_request(
    rejected_completions_endpoint, monkeypatch, stream: bool
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with pytest.raises(litellm.BadRequestError) as rejected:
        await atext_completion(**AZURE_TEXT_CALL, stream=stream)

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"
