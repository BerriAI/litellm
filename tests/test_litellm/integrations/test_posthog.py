

def test_distinct_id_prefers_session_id_over_trace_id():
    """PostHog identifies a person, so it needs session grouping rather than the trace."""
    from litellm.integrations.posthog import PostHogLogger

    logger = PostHogLogger.__new__(PostHogLogger)
    payload = {"trace_id": "per-trace", "session_id": "per-session"}

    assert logger._get_distinct_id(standard_logging_object=payload, kwargs={}) == "per-session"

    legacy = {"trace_id": "only-trace"}
    assert logger._get_distinct_id(standard_logging_object=legacy, kwargs={}) == "only-trace"
