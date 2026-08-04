"""Tests for the transport's failure classification.

No `e2e` marker: this is harness coverage, so it runs without a proxy. It pins two
contracts the suite's triage depends on.

Edge-proxy statuses (502/503/504) classify as InfraUnavailable, separately from
litellm's own errors, because the two mean different things: an ALB answering 504
says nothing about the behavior under test, while a litellm 500 is a real defect.
Before the split every non-2xx collapsed into UnknownApiError, so an unhealthy
environment and a product regression were indistinguishable in the report.

Failures carry the gateway's x-litellm-call-id. The transport already read the
header; it was dropped on the failure path, leaving a red test on the cluster with
no token to search the gateway's logs or traces by.

Real `requests.Response` objects are assembled here rather than any kind of double:
the thing under test is exactly how a real response is interpreted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import requests
from _pytest.outcomes import Failed
from pydantic import BaseModel

from e2e_http import (
    CALL_ID_FIELD,
    EDGE_PROXY_STATUS,
    INFRA_FAILURE_PREFIX,
    InfraUnavailable,
    RateLimitedError,
    StreamingResponse,
    Success,
    UnauthorizedError,
    UnknownApiError,
    ValidationError,
    classify_response,
    call_id_suffix,
    is_transient_failure,
    require_successful_call,
    unwrap,
)


class _Body(BaseModel):
    id: str


#: What an ALB or nginx serves when it cannot reach the gateway (verbatim shape from
#: the committed stage run logs).
EDGE_PROXY_HTML = (
    b"<html>\r\n<head><title>504 Gateway Time-out</title></head>\r\n"
    b"<body>\r\n<center><h1>504 Gateway Time-out</h1></center>\r\n</body>\r\n</html>"
)


def _response(
    status_code: int,
    *,
    body: bytes = b"{}",
    call_id: str | None = None,
    content_type: str | None = None,
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = body  # pyright: ignore[reportPrivateUsage]  # only supported way to build a Response offline
    if call_id is not None:
        resp.headers["x-litellm-call-id"] = call_id
    if content_type is not None:
        resp.headers["content-type"] = content_type
    return resp


@pytest.mark.parametrize("status_code", sorted(EDGE_PROXY_STATUS))
def test_edge_proxy_html_page_is_infra_not_unknown(status_code: int) -> None:
    """An edge-proxy status with an HTML error page and no call id is the ALB or
    nginx answering on its own, so it must not be reported as an API error from the
    code under test."""
    result = classify_response(
        _response(status_code, body=EDGE_PROXY_HTML, content_type="text/html"), _Body
    )
    assert isinstance(result, InfraUnavailable), (
        f"HTTP {status_code} with an HTML body and no call id is an edge-proxy "
        f"failure and must classify as InfraUnavailable so a red run can be triaged "
        f"as an unhealthy environment; got {result!r}"
    )
    assert result.status_code == status_code


@pytest.mark.parametrize(
    ("status_code", "body", "why"),
    [
        (
            503,
            b'{"error":{"message":"litellm.ServiceUnavailableError: Model \'gpt-5.5\' is '
            b'currently paused and cannot accept requests."}}',
            "/model/block is a product contract, and a paused model answers 503",
        ),
        (
            503,
            b'{"detail":"Database not available"}',
            "an unreachable DB is a real failure the suite must surface",
        ),
        (
            503,
            b'{"error":{"message":"Guardrail blocked on error"}}',
            "a guardrail configured to fail closed answers 503 by design",
        ),
        (
            504,
            b'{"detail":"MCP handler did not respond in time"}',
            "an MCP handler timing out is a product defect, not an ALB timeout",
        ),
        (
            502,
            b'{"error":{"message":"litellm.BadGatewayError: provider returned 502"}}',
            "litellm maps an upstream provider error to its own 502",
        ),
    ],
)
def test_litellm_own_5xx_is_never_blamed_on_the_environment(
    status_code: int, body: bytes, why: str
) -> None:
    """The regression this class of bug produces is the worst one available: a real
    product failure silently reclassified as environment noise and dropped from
    triage. litellm emits 503 and 504 from its own handlers, so status code alone
    must never move blame off the product."""
    result = classify_response(
        _response(status_code, body=body, content_type="application/json"), _Body
    )
    assert isinstance(result, UnknownApiError), (
        f"HTTP {status_code} answered by litellm itself ({why}) must stay a product "
        f"failure; calling it infra would discard a real regression. got {result!r}"
    )


def test_call_id_alone_proves_litellm_answered() -> None:
    """Only the gateway mints x-litellm-call-id, so its presence is proof the request
    reached litellm - even on an edge-proxy status with an HTML-looking body."""
    result = classify_response(
        _response(504, body=EDGE_PROXY_HTML, call_id="call-real-1"), _Body
    )
    assert isinstance(result, UnknownApiError), (
        "a response carrying x-litellm-call-id came from the gateway; it cannot be "
        f"an edge proxy answering on its own. got {result!r}"
    )


def test_ambiguous_failure_stays_a_product_failure() -> None:
    """Asymmetric costs: a false 'product bug' costs an investigation, a false 'bad
    environment' silently discards a regression. Ambiguity must resolve toward the
    product."""
    result = classify_response(_response(503, body=b""), _Body)
    assert isinstance(result, UnknownApiError), (
        f"an empty-bodied 503 is not proof of an edge proxy; got {result!r}"
    )


def test_litellm_500_stays_unknown_api_error() -> None:
    """The blame boundary: litellm returns 500 for its own unhandled errors, and
    those are real defects. Folding 500 into the infra bucket would hide them."""
    result = classify_response(
        _response(500, body=b'{"error":{"message":"reload degraded"}}'), _Body
    )
    assert isinstance(result, UnknownApiError), (
        "a litellm 500 is a product failure, not environment noise; classifying it "
        f"as infra would suppress a real regression. got {result!r}"
    )
    assert result.status_code == 500


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (200, Success),
        (401, UnauthorizedError),
        (429, RateLimitedError),
        (400, UnknownApiError),
        (404, UnknownApiError),
    ],
)
def test_non_edge_statuses_keep_their_existing_classification(
    status_code: int, expected: type[BaseModel]
) -> None:
    """The new variant must not perturb the statuses tests already match on."""
    result = classify_response(_response(status_code, body=b'{"id":"x"}'), _Body)
    assert isinstance(result, expected), f"HTTP {status_code} -> {result!r}"


def test_unwrap_infra_failure_names_the_environment_not_the_code() -> None:
    """The assertion text is what an engineer reads first, and what the dashboard
    can split on, so it must carry the greppable marker."""
    result = classify_response(_response(503, body=b"<html>no healthy upstream</html>"), _Body)
    with pytest.raises(AssertionError) as excinfo:
        _ = unwrap(result)
    message = str(excinfo.value)
    assert INFRA_FAILURE_PREFIX in message, (
        "an infra failure must be greppable by a stable marker so unhealthy runs can "
        f"be separated from regressions without parsing tracebacks; got: {message}"
    )
    assert "503" in message


def test_infra_failure_message_does_not_claim_a_code_regression() -> None:
    result = classify_response(_response(502, body=b"<html>bad gateway</html>"), _Body)
    with pytest.raises(AssertionError) as excinfo:
        _ = unwrap(result)
    assert "not a behavior regression" in str(excinfo.value)


@pytest.mark.parametrize("status_code", [400, 429, 500, 502])
def test_failure_carries_call_id_for_gateway_correlation(status_code: int) -> None:
    """Every failure the gateway answered must expose its call id, so a cluster
    failure is a lookup in the gateway's logs rather than a guess."""
    result = classify_response(
        _response(status_code, body=b'{"error":"x"}', call_id="call-abc-123"), _Body
    )
    assert getattr(result, "call_id", None) == "call-abc-123", (
        f"HTTP {status_code} dropped x-litellm-call-id; without it a red test on the "
        "cluster cannot be traced to the request that produced it"
    )


def test_call_id_reaches_the_assertion_message() -> None:
    """Carrying the id on the model is only useful if it survives into the text
    that JUnit records as the failure."""
    result = classify_response(_response(500, body=b'{"error":"x"}', call_id="call-xyz-789"), _Body)
    with pytest.raises(AssertionError) as excinfo:
        _ = unwrap(result)
    assert "call-xyz-789" in str(excinfo.value)


def test_validation_error_keeps_call_id() -> None:
    """A 200 whose body does not match the model is still a failure worth tracing."""
    result = classify_response(_response(200, body=b'{"wrong":"shape"}', call_id="call-v-1"), _Body)
    assert isinstance(result, ValidationError)
    assert result.call_id == "call-v-1"


@pytest.mark.parametrize(
    ("status_code", "body", "content_type"),
    [
        (504, EDGE_PROXY_HTML, "text/html"),
        (502, EDGE_PROXY_HTML, "text/html"),
        (503, EDGE_PROXY_HTML, "text/html"),
        (500, b'{"error":"boom"}', "application/json"),
        (408, b'{"error":"timeout"}', "application/json"),
        (429, b'{"error":"slow down"}', "application/json"),
    ],
)
def test_transient_failures_stay_retryable(
    status_code: int, body: bytes, content_type: str
) -> None:
    """Retry-worthiness must follow the variant, not a status list each caller keeps
    its own copy of. Adding InfraUnavailable to the union silently emptied the
    batches retry budget for 502/503/504 - exactly the statuses those retries exist
    for - because the match arm discriminated on UnknownApiError by class."""
    result = classify_response(
        _response(status_code, body=body, content_type=content_type), _Body
    )
    assert is_transient_failure(result), (
        f"HTTP {status_code} is transient and must remain retryable regardless of "
        f"which Result variant it maps to; got {result!r}"
    )


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_deterministic_failures_are_not_retryable(status_code: int) -> None:
    """Retrying a 4xx just burns wall clock and provider budget."""
    result = classify_response(
        _response(status_code, body=b'{"error":"nope"}', content_type="application/json"),
        _Body,
    )
    assert not is_transient_failure(result), (
        f"HTTP {status_code} is a deterministic rejection and must not be retried; "
        f"got {result!r}"
    )


def test_unauthorized_carries_call_id() -> None:
    """A 401 on the cluster is as worth tracing as any other failure."""
    result = classify_response(_response(401, call_id="call-401-1"), _Body)
    assert isinstance(result, UnauthorizedError)
    assert result.call_id == "call-401-1"


def test_this_module_makes_no_network_calls() -> None:
    """Guards the allowlist entry in check_e2e_no_raw_requests.py: `requests` is
    imported here only to build Response objects offline. If someone later adds a
    real call, the exemption stops being honest and this fails."""
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = tuple(
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    )
    # requests.Response() is an in-memory constructor, not a request; the verbs and
    # Session are what would actually put bytes on the wire.
    sending = frozenset(
        {
            "requests.get",
            "requests.post",
            "requests.put",
            "requests.patch",
            "requests.delete",
            "requests.head",
            "requests.options",
            "requests.request",
            "requests.Session",
        }
    )
    offenders = tuple(name for name in called if name in sending)
    assert offenders == (), (
        "this module is exempt from the raw-HTTP ban only because it sends nothing; "
        f"remove the exemption or the call(s): {offenders}"
    )


def test_call_id_suffix_is_empty_when_absent() -> None:
    """No id (the usual case for an edge proxy that never reached litellm) must not
    produce a dangling `call_id=None` in the message."""
    assert call_id_suffix(None) == ""
    assert call_id_suffix("") == ""
    assert call_id_suffix("abc") == f" {CALL_ID_FIELD}=abc"


def test_require_successful_call_splits_infra_on_the_streaming_path() -> None:
    """Streaming/native calls skip classify_response, so the split holds here too:
    this is the path the raw ALB HTML error pages arrive on."""
    streamed = StreamingResponse(
        status_code=504,
        body="<html><title>504 Gateway Time-out</title></html>",
        call_id=None,
    )
    # pytest.fail raises Failed, which derives from BaseException, not Exception.
    with pytest.raises(Failed) as excinfo:
        require_successful_call(streamed)
    assert INFRA_FAILURE_PREFIX in str(excinfo.value)


def test_require_successful_call_keeps_product_failures_distinct() -> None:
    streamed = StreamingResponse(
        status_code=400,
        body='{"error":{"message":"bad request"}}',
        call_id="call-s-2",
    )
    with pytest.raises(Failed) as excinfo:
        require_successful_call(streamed)
    message = str(excinfo.value)
    assert INFRA_FAILURE_PREFIX not in message, (
        "a 400 is the product rejecting the request; labelling it infra would hide a "
        f"real behavior failure. got: {message}"
    )
    assert "call-s-2" in message
