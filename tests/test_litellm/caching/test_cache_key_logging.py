import logging

from litellm.caching.caching import Cache


def test_cache_key_logging_does_not_emit_raw_request_payload(caplog):
    cache = Cache()
    payload_marker = "RAW_PAYLOAD_MARKER_29414"
    tool_marker = "RAW_TOOL_MARKER_29414"
    schema_marker = "RAW_SCHEMA_MARKER_29414"

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        cache_key = cache.get_cache_key(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize {payload_marker}.",
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_customer",
                        "description": tool_marker,
                        "parameters": {
                            "type": "object",
                            "properties": {"customer_id": {"type": "string"}},
                        },
                    },
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "summary",
                    "schema": {
                        "type": "object",
                        "description": schema_marker,
                    },
                },
            },
        )

    log_output = "\n".join(
        record.getMessage() for record in caplog.records if record.name == "LiteLLM"
    )

    assert payload_marker not in log_output
    assert tool_marker not in log_output
    assert schema_marker not in log_output
    assert cache_key in log_output
