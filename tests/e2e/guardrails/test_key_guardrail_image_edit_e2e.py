from __future__ import annotations

import base64
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import Success, UnknownApiError
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from guardrails_client import GuardrailsClient, poll_until_blocked
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

CHAT_MODEL: Final = "gemini-2.5-flash"
IMAGE_BACKEND: Final = "openai/gpt-image-2.5-flare"
SOURCE_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo3"
    "3UrYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AYHLAMpT0sIcNbcEAAAAAElFTkSuQmCC"
)


def _edit_prompt_with(banned_keyword: str) -> str:
    return f"Turn this into a watercolor painting of a lighthouse. {banned_keyword}"


def _create_image_model(client: GuardrailsClient, resources: ResourceManager) -> str:
    model_name = f"e2e-guard-image-edit-{unique_marker()}"
    model_id = client.proxy.create_model(
        model_name,
        LiteLLMParamsBody(model=IMAGE_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
        provider_live=True,
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_name


class TestKeyAttachedGuardrailOnImageEdits:
    @pytest.mark.covers(
        "guardrail.litellm_content_filter.pre_call.blocks_image_edit",
        exercised_on=["images_edits"],
    )
    @meta(
        Subject(
            domain=Domain.GUARDRAILS,
            route=Route.IMAGES,
            providers=(Provider.GEMINI, Provider.OPENAI,),
            models=(CHAT_MODEL, IMAGE_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_key_attached_content_filter_blocks_banned_image_edit_prompt(
        self, client: GuardrailsClient, resources: ResourceManager
    ) -> None:
        banned = unique_marker()
        guardrail_name = f"e2e-image-edit-filter-{banned}"
        guardrail_id = client.create_content_filter_guardrail(guardrail_name, banned, default_on=False)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))
        key = client.create_key_with_guardrails(resources, [guardrail_name])
        model = _create_image_model(client, resources)

        synced = poll_until_blocked(lambda: client.chat(key, CHAT_MODEL, _edit_prompt_with(banned)))
        assert isinstance(synced, UnknownApiError) and synced.status_code == 400, (
            f"key guardrail {guardrail_name!r} never synced to the proxy on /chat/completions: {synced}"
        )

        result = client.edit_image(key, model, _edit_prompt_with(banned), SOURCE_PNG)
        match result:
            case UnknownApiError(status_code=status, body=body):
                assert status == 400, f"expected a 400 guardrail block, got {status}: {body[:300]}"
                assert "content blocked" in body.lower() or banned in body, (
                    f"block response missing content-filter reason: {body[:300]}"
                )
            case Success():
                pytest.fail(
                    f"key-attached guardrail {guardrail_name!r} was skipped on /v1/images/edits: "
                    "the banned prompt reached the provider and an edited image came back"
                )
            case _:
                pytest.fail(f"unexpected /v1/images/edits outcome for a banned prompt: {result}")
