import asyncio
import json
import re

import httpx
import pytest
import respx

import litellm
from litellm.integrations.argilla import ArgillaLogger

ARGILLA_BASE_URL = "https://argilla.example.test"
USER_MESSAGE = {"role": "user", "content": "where is my order?"}


async def _cancel_background_tasks() -> None:
    background_tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in background_tasks:
        task.cancel()
    await asyncio.gather(*background_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transformation_object", "standard_logging_object", "expected_fields"),
    [
        pytest.param(
            {"prompt": "messages", "answer": "response"},
            {"messages": [USER_MESSAGE], "response": "it ships tomorrow"},
            {"prompt": [USER_MESSAGE], "answer": "it ships tomorrow"},
            id="message-list-and-text-response",
        ),
        pytest.param(
            {"answer": "response", "prompt": "messages"},
            {"messages": USER_MESSAGE, "response": {"choices": [{"message": {"content": "it ships tomorrow"}}]}},
            {"answer": "it ships tomorrow", "prompt": [USER_MESSAGE]},
            id="single-message-and-chat-completion-response",
        ),
        pytest.param(
            {"prompt": "messages", "context": "messages", "answer": "response", "label": "response"},
            {"messages": [USER_MESSAGE, USER_MESSAGE], "response": {"choices": [{"message": {}}]}},
            {
                "prompt": [USER_MESSAGE, USER_MESSAGE],
                "context": [USER_MESSAGE, USER_MESSAGE],
                "answer": "",
                "label": "",
            },
            id="payload-field-reused-by-several-argilla-fields",
        ),
    ],
)
async def test_argilla_logger_sends_the_record_shaped_by_the_configured_field_mapping(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    transformation_object: dict[str, str],
    standard_logging_object: dict[str, object],
    expected_fields: dict[str, object],
) -> None:
    monkeypatch.setattr(litellm, "argilla_transformation_object", transformation_object)
    respx_mock.get(f"{ARGILLA_BASE_URL}/api/v1/me/datasets", params={"name": "support-chats"}).mock(
        return_value=httpx.Response(200, json={"items": [{"id": "dataset-403"}]})
    )
    bulk_route = respx_mock.post(f"{ARGILLA_BASE_URL}/api/v1/datasets/dataset-403/records/bulk").mock(
        return_value=httpx.Response(200, json={})
    )
    logger = ArgillaLogger(
        argilla_api_key="argilla-key",
        argilla_dataset_name="support-chats",
        argilla_base_url=ARGILLA_BASE_URL,
        batch_size=1,
    )
    try:
        logger.log_success_event({"standard_logging_object": standard_logging_object}, None, None, None)
    finally:
        await _cancel_background_tasks()

    sent_request = bulk_route.calls.last.request
    assert json.loads(sent_request.content) == [{"fields": expected_fields}]
    assert sent_request.headers["X-Argilla-Api-Key"] == "argilla-key"
    assert logger.argilla_transformation_object is transformation_object


@pytest.mark.parametrize(
    ("transformation_object", "message"),
    [
        pytest.param(
            None,
            "'litellm.argilla_transformation_object' is required, to log your payload to Argilla.",
            id="mapping-not-configured",
        ),
        pytest.param(
            ["messages", "response"],
            "'argilla_transformation_object' must be a dictionary, to log your payload to Argilla.",
            id="mapping-is-a-list",
        ),
        pytest.param(
            {"prompt": "messages", "spend": "response_cost"},
            "All values in argilla_transformation_object must be a key in SUPPORTED_PAYLOAD_FIELDS, "
            "response_cost is not a valid key.",
            id="mapping-names-an-unsupported-payload-field",
        ),
    ],
)
def test_argilla_logger_refuses_to_start_without_a_supported_field_mapping(
    monkeypatch: pytest.MonkeyPatch, transformation_object: object, message: str
) -> None:
    monkeypatch.setattr(litellm, "argilla_transformation_object", transformation_object)

    with pytest.raises(Exception, match=re.escape(message)):
        ArgillaLogger(argilla_api_key="argilla-key")
