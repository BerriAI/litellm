"""Tests for LiteLLMSendMessageResponse JSON-RPC normalization."""

from litellm.types.agents import LiteLLMSendMessageResponse


def test_from_dict_backfills_id_on_agent_error_response():
    agent_error = {
        "jsonrpc": "2.0",
        "error": {"code": -32054, "message": "Session not found"},
    }

    response = LiteLLMSendMessageResponse.from_dict(
        agent_error, request_id="r1"
    )

    assert response.id == "r1"
    assert response.error == {"code": -32054, "message": "Session not found"}
    assert response.result is None


def test_from_dict_preserves_existing_id():
    payload = {
        "id": "upstream-id",
        "jsonrpc": "2.0",
        "error": {"code": -32001, "message": "Task not found"},
    }

    response = LiteLLMSendMessageResponse.from_dict(
        payload, request_id="r1"
    )

    assert response.id == "upstream-id"


def test_from_dict_preserves_integer_id_echoed_by_upstream():
    """JSON-RPC 2.0 types ``id`` as string|integer|null, and pydantic v2 does not
    coerce int to str, so a str-only annotation rejects an upstream agent that
    echoes an integer id. The value AND the type must survive."""
    payload = {
        "id": 42,
        "jsonrpc": "2.0",
        "result": {"kind": "task"},
    }

    response = LiteLLMSendMessageResponse.from_dict(payload, request_id="r1")

    assert response.id == 42
    assert isinstance(response.id, int)


def test_from_dict_preserves_falsy_integer_id():
    """``0`` is a legal JSON-RPC id and is falsy, so it must not be mistaken for an
    absent id and backfilled from the request id."""
    payload = {"id": 0, "jsonrpc": "2.0", "result": {}}

    response = LiteLLMSendMessageResponse.from_dict(payload, request_id="r1")

    assert response.id == 0


def test_backfilled_id_keeps_the_request_id_type():
    """The proxy's A2A endpoint reads the caller's ``id`` straight off the request
    body, so it can be an integer. JSON-RPC requires the response id to equal the
    request id, so backfilling an omitted id must not stringify it: a caller that
    sent ``7`` cannot correlate a response carrying ``"7"``. One test, both
    directions, so neither can regress unnoticed."""
    agent_error = {
        "jsonrpc": "2.0",
        "error": {"code": -32054, "message": "Session not found"},
    }

    from_int = LiteLLMSendMessageResponse.from_dict(agent_error, request_id=7)
    from_str = LiteLLMSendMessageResponse.from_dict(agent_error, request_id="7")

    assert from_int.id == 7
    assert isinstance(from_int.id, int)
    assert from_str.id == "7"
    assert isinstance(from_str.id, str)


def test_from_dict_accepts_null_id_when_the_error_cannot_be_correlated():
    """JSON-RPC 2.0 section 5 requires ``id`` to be null on an error that cannot be
    matched to a request, which is exactly the case where the caller supplied no id
    for the backfill to use. Rejecting it turned an agent's error into a proxy 500."""
    response = LiteLLMSendMessageResponse.from_dict(
        {"jsonrpc": "2.0", "error": {"code": -32054, "message": "x"}}
    )

    assert response.id is None
    assert response.error == {"code": -32054, "message": "x"}


def test_from_dict_accepts_null_id_echoed_by_upstream():
    """An agent may answer an uncorrelatable request with an explicit ``"id": null``.
    That is a well-formed response, not a validation failure."""
    response = LiteLLMSendMessageResponse.from_dict(
        {"id": None, "jsonrpc": "2.0", "error": {"code": -32600, "message": "bad"}}
    )

    assert response.id is None


def test_id_accepts_every_member_of_the_json_rpc_union_and_nothing_else():
    """One test pinning the whole ``string | integer | null`` union the spec defines,
    so widening the annotation cannot silently become "accept anything"."""
    for accepted in ("s1", 42, 0, None):
        assert LiteLLMSendMessageResponse(id=accepted).id == accepted

    # ``True``/``False`` are in here because bool subclasses int: a non-strict integer
    # half would accept them and relay them as 1/0. Direct construction bypasses
    # normalization, so the model has to hold this line on its own.
    for rejected in (True, False, 1.5, ["a"], {"a": 1}):
        try:
            LiteLLMSendMessageResponse(id=rejected)
        except Exception:
            continue
        raise AssertionError(f"id={rejected!r} is outside the JSON-RPC union and must be rejected")


def test_boolean_id_is_never_relayed_as_an_integer():
    """``bool`` subclasses ``int``, so widening the annotation to accept integers also
    made pydantic coerce a boolean id to 1 or 0. That is worse than rejecting it: an id
    of ``1`` collides with a real integer id another in-flight request may be using.
    Both directions in one test, since either alone leaves the other free to regress."""
    agent_error = {"jsonrpc": "2.0", "error": {"code": -32054, "message": "x"}}

    echoed = LiteLLMSendMessageResponse.from_dict({"id": True, "jsonrpc": "2.0", "result": {}})
    backfilled = LiteLLMSendMessageResponse.from_dict(agent_error, request_id=True)

    assert echoed.id == "True"
    assert backfilled.id == "True"
    assert not isinstance(echoed.id, int)
    assert not isinstance(backfilled.id, int)
