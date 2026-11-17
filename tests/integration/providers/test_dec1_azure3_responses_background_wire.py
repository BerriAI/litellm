"""Azure Responses background mode: queue, poll, finish and cancel through the proxy's managed response ids.

`background: true` is an OpenAI Responses request field litellm forwards unchanged
(`litellm/llms/openai/responses/transformation.py:89`). Azure answers at once with `status: queued`, and the caller
polls `GET /v1/responses/{id}` and may `POST /v1/responses/{id}/cancel`. The proxy replaces Azure's id with a
managed id, `resp_` + base64 of `litellm:custom_llm_provider:azure;model_id:<deployment id>;response_id:<raw>`
(`ResponsesAPIRequestUtils._build_responses_api_response_id`), and decodes it to route follow-ups back to the same
Azure deployment, where `AzureOpenAIResponsesAPIConfig._construct_url_for_response_id_in_path` puts the raw id in
the path before the query.

With no `api_version` the base is `/openai/v1/responses?api-version=preview` (round 1 pinned that for create);
follow-ups must keep that query. A Rust port that changes id encoding breaks every in-flight background job.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Final
from urllib.parse import urlsplit

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure3_support import decoded_response_id, full_peer, responses_object
from integration.providers.dec1_azure_support import API_KEY, GPT5_DEPLOYMENT, body_of, json_reply, marker, query_of

MODEL: Final = f"azure/{GPT5_DEPLOYMENT}"


def _lifecycle(raw_id: str, poll_statuses: list[str]) -> Callable[[Request], Reply]:
    """POST create answers queued; each GET answers the next poll status (the last repeats); cancel answers cancelled."""

    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "POST" and path.endswith("/cancel"):
            return json_reply(responses_object(raw_id, "cancelled"))
        if request.method == "POST":
            return json_reply(responses_object(raw_id, "queued"))
        if request.method == "GET" and path.endswith(f"/{raw_id}"):
            status = poll_statuses.pop(0) if len(poll_statuses) > 1 else poll_statuses[0]
            return json_reply(
                responses_object(raw_id, status, text="background answer" if status == "completed" else None)
            )
        if request.method == "DELETE":
            return json_reply({"id": raw_id, "object": "response", "deleted": True})
        return json_reply({"object": "list", "data": []})

    return respond


def _start(gateway: Gateway, respond: Callable[[Request], Reply]):  # noqa: ANN202
    return full_peer(gateway, respond, model=MODEL, api_key=API_KEY)


def _create(rig: object) -> dict[str, object]:
    response = rig.gateway.request(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        "POST",
        "/v1/responses",
        {"model": rig.model, "input": marker(), "background": True},  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )
    assert response.status_code == 200, response.text  # pyright: ignore[reportUnknownMemberType]
    return response.json()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _provider_calls(rig: object) -> list[Request]:
    return [r for r in rig.wire.drain() if "/responses" in urlsplit(r.target).path]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]


def test_background_true_is_forwarded_to_azure(gateway: Gateway) -> None:
    """The caller creates a response with `background: true` on an Azure deployment with no api_version.

    Azure must get one `POST /openai/v1/responses?api-version=preview` whose body has `background: true`.
    """
    raw: Final = f"resp_{uuid.uuid4().hex}"
    with _start(gateway, _lifecycle(raw, ["queued"])) as rig:
        _create(rig)
        (create,) = _provider_calls(rig)
        assert (urlsplit(create.target).path, query_of(create)) == ("/openai/v1/responses", {"api-version": "preview"})
        assert body_of(create)["background"] is True


def test_the_queued_answer_carries_a_managed_id_that_names_azure_and_the_raw_id(gateway: Gateway) -> None:
    """Azure answers `status: queued` with id `resp_<raw>`.

    The caller must get `status == "queued"` and an `id` that is not Azure's raw id but decodes to
    `litellm:custom_llm_provider:azure;model_id:<id>;response_id:<raw>`.
    """
    raw: Final = f"resp_{uuid.uuid4().hex}"
    with _start(gateway, _lifecycle(raw, ["queued"])) as rig:
        created: Final = _create(rig)
        assert created["status"] == "queued"
        assert created["id"] != raw
        decoded: Final = decoded_response_id(str(created["id"]))
        assert decoded.startswith("litellm:custom_llm_provider:azure;model_id:") and decoded.endswith(
            f";response_id:{raw}"
        )


def test_polling_the_managed_id_reaches_the_same_deployment_with_the_raw_id_and_query(gateway: Gateway) -> None:
    """The caller polls `GET /v1/responses/<managed id>` once while Azure says `in_progress`.

    Azure must get `GET /openai/v1/responses/<raw>?api-version=preview` with the deployment key in `api-key`, and
    the caller must see `status == "in_progress"`.
    """
    raw: Final = f"resp_{uuid.uuid4().hex}"
    with _start(gateway, _lifecycle(raw, ["in_progress"])) as rig:
        created: Final = _create(rig)
        polled: Final = rig.gateway.request("GET", f"/v1/responses/{created['id']}")
        assert polled.status_code == 200, polled.text
        assert polled.json()["status"] == "in_progress"
        get: Final = [r for r in _provider_calls(rig) if r.method == "GET"]
        assert [(urlsplit(r.target).path, query_of(r)) for r in get] == [
            (f"/openai/v1/responses/{raw}", {"api-version": "preview"})
        ]
        assert get[0].headers.get("api-key") == API_KEY


def test_a_completed_poll_returns_the_output_text_and_usage(gateway: Gateway) -> None:
    """Azure answers the first poll `in_progress` and the second `completed` with text `background answer`.

    The second poll must give the caller `status == "completed"`, `output[0].content[0].text ==
    "background answer"` and `usage.total_tokens == 18`.
    """
    raw: Final = f"resp_{uuid.uuid4().hex}"
    with _start(gateway, _lifecycle(raw, ["in_progress", "completed"])) as rig:
        created: Final = _create(rig)
        rig.gateway.request("GET", f"/v1/responses/{created['id']}")
        final: Final = rig.gateway.request("GET", f"/v1/responses/{created['id']}").json()
        assert final["status"] == "completed"
        assert final["output"][0]["content"][0]["text"] == "background answer"
        assert final["usage"]["total_tokens"] == 18


def test_cancelling_a_background_response_posts_cancel_with_the_raw_id(gateway: Gateway) -> None:
    """The caller cancels with `POST /v1/responses/<managed id>/cancel`.

    Azure must get `POST /openai/v1/responses/<raw>/cancel?api-version=preview` and the caller must see
    `status == "cancelled"`.
    """
    raw: Final = f"resp_{uuid.uuid4().hex}"
    with _start(gateway, _lifecycle(raw, ["queued"])) as rig:
        created: Final = _create(rig)
        cancelled: Final = rig.gateway.request("POST", f"/v1/responses/{created['id']}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        cancel: Final = [r for r in _provider_calls(rig) if urlsplit(r.target).path.endswith("/cancel")]
        assert [(urlsplit(r.target).path, query_of(r)) for r in cancel] == [
            (f"/openai/v1/responses/{raw}/cancel", {"api-version": "preview"})
        ]


def test_a_dated_deployment_polls_with_the_responses_api_version(gateway: Gateway) -> None:
    """The deployment sets `api_version: 2025-04-01-preview` (valid for Responses, so kept on the classic route).

    The create and the poll must both use `/openai/responses...` with query `{"api-version": "2025-04-01-preview"}`,
    the poll path being `/openai/responses/<raw>`.
    """
    raw: Final = f"resp_{uuid.uuid4().hex}"
    with full_peer(
        gateway, _lifecycle(raw, ["queued"]), model=MODEL, api_key=API_KEY, api_version="2025-04-01-preview"
    ) as rig:
        created: Final = _create(rig)
        rig.gateway.request("GET", f"/v1/responses/{created['id']}")
        assert [(r.method, urlsplit(r.target).path, query_of(r)) for r in _provider_calls(rig)] == [
            ("POST", "/openai/responses", {"api-version": "2025-04-01-preview"}),
            ("GET", f"/openai/responses/{raw}", {"api-version": "2025-04-01-preview"}),
        ]


def test_deleting_a_background_response_reaches_azure_with_the_raw_id(gateway: Gateway) -> None:
    """The caller deletes with `DELETE /v1/responses/<managed id>`.

    Azure must get `DELETE /openai/v1/responses/<raw>?api-version=preview` and the caller must get 200 with
    `deleted: true`.
    """
    raw: Final = f"resp_{uuid.uuid4().hex}"
    with _start(gateway, _lifecycle(raw, ["queued"])) as rig:
        created: Final = _create(rig)
        deleted: Final = rig.gateway.request("DELETE", f"/v1/responses/{created['id']}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["deleted"] is True
        delete: Final = [r for r in _provider_calls(rig) if r.method == "DELETE"]
        assert [(urlsplit(r.target).path, query_of(r)) for r in delete] == [
            (f"/openai/v1/responses/{raw}", {"api-version": "preview"})
        ]
