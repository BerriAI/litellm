"""Tests for the post_call logging_obj stash + sibling fallback bridge."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_post_call_recovers_app_id_via_logging_obj_stash(
    make_guardrail, make_request_data, make_logging_obj
):
    """Reproduces the framework gap: request body metadata is dropped before
    post_call. The logging_obj stash from the prior ``input_type="request"``
    call must be used to resolve app_id."""
    guardrail, client = make_guardrail(allow_request_metadata_override=True)
    guardrail._client_cache["default-api-key"] = client
    request_obj = Mock()
    request_obj.action = "NO_ACTION"
    request_obj.detections = []
    request_obj.correlation_id = None
    client.evaluate_prompt.return_value = request_obj
    response_obj = Mock()
    response_obj.action = "NO_ACTION"
    response_obj.detections = []
    response_obj.correlation_id = None
    client.evaluate_response.return_value = response_obj

    logging_obj = make_logging_obj()

    # Step 1: simulate pre_call / during_call with full request body
    # metadata — this is where the stash happens.
    await guardrail.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data=make_request_data(
            metadata={"alice_wonderfence_app_id": "tenant-X"}
        ),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Step 2: simulate post_call as the framework actually invokes it —
    # the request body's metadata is gone (only litellm_metadata.user_api_key_*
    # would normally be present, neither populated here). Without the
    # bridge this raises; with it, we recover from logging_obj.
    out = await guardrail.apply_guardrail(
        inputs={"texts": ["llm response"]},
        request_data={"model": "gpt-4", "metadata": {}},
        input_type="response",
        logging_obj=logging_obj,
    )
    assert out["texts"] == ["llm response"]
    assert client.evaluate_response.call_args.kwargs["app_id"] == "tenant-X"


@pytest.mark.asyncio
async def test_post_call_prefers_request_data_over_stash(
    make_guardrail, make_request_data, make_logging_obj
):
    """If post_call's request_data still resolves (e.g. app_id from key/team
    metadata), use it — don't fall back to the stash."""
    guardrail, client = make_guardrail(allow_request_metadata_override=True)
    guardrail._client_cache["default-api-key"] = client
    request_obj = Mock()
    request_obj.action = "NO_ACTION"
    request_obj.detections = []
    request_obj.correlation_id = None
    client.evaluate_prompt.return_value = request_obj
    response_obj = Mock()
    response_obj.action = "NO_ACTION"
    response_obj.detections = []
    response_obj.correlation_id = None
    client.evaluate_response.return_value = response_obj

    logging_obj = make_logging_obj()

    # Stash a different app_id during the request phase.
    await guardrail.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=make_request_data(
            metadata={"alice_wonderfence_app_id": "stashed-app"}
        ),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Post_call request_data resolves via key metadata to a DIFFERENT app_id.
    # The resolver path must win over the stash.
    await guardrail.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data={
            "model": "gpt-4",
            "metadata": {
                "user_api_key_metadata": {"alice_wonderfence_app_id": "key-app"}
            },
        },
        input_type="response",
        logging_obj=logging_obj,
    )
    assert client.evaluate_response.call_args.kwargs["app_id"] == "key-app"


@pytest.mark.asyncio
async def test_post_call_without_prior_stash_raises(make_guardrail, make_logging_obj):
    """If neither request_data nor logging_obj has the app_id (e.g. mode is
    post_call only and app_id was supplied only in the request body), the
    error path must still fire — not silently allow."""
    guardrail, client = make_guardrail()
    guardrail._client_cache["default-api-key"] = client

    logging_obj = make_logging_obj()  # empty model_call_details

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["resp"]},
            request_data={"model": "gpt-4", "metadata": {}},
            input_type="response",
            logging_obj=logging_obj,
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]


@pytest.mark.asyncio
async def test_post_call_does_not_borrow_sibling_stash(
    make_guardrail, make_request_data, make_logging_obj
):
    """A stricter instance must NOT inherit a sibling's stashed credentials.

    Exploit being closed: a permissive writer (allow_request_metadata_override
    =True) stashes caller-supplied request-body credentials; a stricter reader
    (allow_request_metadata_override=False) that has no own stash must fail
    closed in post_call rather than scan with the writer's caller-controlled
    creds."""
    g_writer, c_writer = make_guardrail(
        guardrail_name="writer",
        allow_request_metadata_override=True,
    )
    g_writer._client_cache["default-api-key"] = c_writer
    g_reader, c_reader = make_guardrail(
        guardrail_name="reader",
        allow_request_metadata_override=False,
    )
    g_reader._client_cache["default-api-key"] = c_reader
    for c in (c_writer, c_reader):
        result = Mock()
        result.action = "NO_ACTION"
        result.detections = []
        result.correlation_id = None
        c.evaluate_prompt.return_value = result
        c.evaluate_response.return_value = result

    logging_obj = make_logging_obj()

    # Writer stashes caller-supplied request-body app_id (override allowed).
    await g_writer.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=make_request_data(
            metadata={"alice_wonderfence_app_id": "caller-supplied-app"}
        ),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Reader's post_call: no own stash, request_data resolves nothing, and it
    # must NOT borrow the writer's stash -> fail closed.
    with pytest.raises(HTTPException) as exc:
        await g_reader.apply_guardrail(
            inputs={"texts": ["resp"]},
            request_data={"model": "gpt-4", "metadata": {}},
            input_type="response",
            logging_obj=logging_obj,
        )
    assert exc.value.status_code == 500
    assert "alice_wonderfence_app_id" in exc.value.detail["exception"]
    c_reader.evaluate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_stash_keyed_per_guardrail_name(
    make_guardrail, make_request_data, make_logging_obj
):
    """Two alice_wonderfence instances on the same logging_obj must not
    overwrite each other's stash — they're keyed by guardrail_name."""
    g1, c1 = make_guardrail(
        guardrail_name="alice-a",
        allow_request_metadata_override=True,
    )
    g1._client_cache["default-api-key"] = c1
    g2, c2 = make_guardrail(
        guardrail_name="alice-b",
        allow_request_metadata_override=True,
    )
    g2._client_cache["default-api-key"] = c2
    for c in (c1, c2):
        result = Mock()
        result.action = "NO_ACTION"
        result.detections = []
        result.correlation_id = None
        c.evaluate_prompt.return_value = result
        c.evaluate_response.return_value = result

    logging_obj = make_logging_obj()

    # Both instances stash under the SAME logging_obj using DIFFERENT
    # request app_ids.
    await g1.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=make_request_data(metadata={"alice_wonderfence_app_id": "app-a"}),
        input_type="request",
        logging_obj=logging_obj,
    )
    await g2.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=make_request_data(metadata={"alice_wonderfence_app_id": "app-b"}),
        input_type="request",
        logging_obj=logging_obj,
    )

    # Each must recover its own value on post_call.
    await g1.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data={"model": "gpt-4", "metadata": {}},
        input_type="response",
        logging_obj=logging_obj,
    )
    await g2.apply_guardrail(
        inputs={"texts": ["resp"]},
        request_data={"model": "gpt-4", "metadata": {}},
        input_type="response",
        logging_obj=logging_obj,
    )
    assert c1.evaluate_response.call_args.kwargs["app_id"] == "app-a"
    assert c2.evaluate_response.call_args.kwargs["app_id"] == "app-b"


def test_recover_resolved_with_no_logging_obj_returns_none():
    """``recover_resolved`` must short-circuit on None logging_obj."""
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.credentials import (
        recover_resolved,
    )

    assert recover_resolved(None, "any-name") is None
