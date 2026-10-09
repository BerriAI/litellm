"""Retry breadcrumbs must stay small: they are stored on the request record."""

from litellm.router import RETRY_BREADCRUMB_EXCEPTION_LIMIT, Router


def _record_for(error: Exception) -> dict:
    kwargs = {"model": "gpt-4o", "metadata": {}}
    Router.log_retry(object(), kwargs, error)
    return kwargs["metadata"]["previous_models"][-1]


def test_large_exception_message_is_truncated_in_the_breadcrumb():
    """Litellm exceptions embed the upstream body, so str(e) can be hundreds of KB.

    The release-gate test test_failing_requests_do_not_grow_rss_or_stored_request
    saw the proxy RSS grow 86.8 MB over 300 failing requests against a 48 MB budget.
    """
    error = Exception("Error code: 500 - " + "x" * 200_000)

    record = _record_for(error)

    assert len(record["exception_string"]) < RETRY_BREADCRUMB_EXCEPTION_LIMIT + 100
    assert record["exception_string"].startswith("Error code: 500 - ")
    assert "truncated" in record["exception_string"]
    assert record["exception_type"] == "Exception"


def test_short_exception_message_is_kept_whole():
    record = _record_for(ValueError("bad request"))

    assert record["exception_string"] == "bad request"
    assert record["exception_type"] == "ValueError"


def test_breadcrumb_metadata_stays_bounded_across_requests():
    error = Exception("Error code: 500 - " + "x" * 200_000)

    total = 0
    for _ in range(50):
        kwargs = {"model": "gpt-4o", "metadata": {}}
        for _ in range(6):
            Router.log_retry(object(), kwargs, error)
        total += len(repr(kwargs["metadata"]))

    assert total < 500_000
