from datetime import datetime, timezone

from litellm.integrations.s3 import get_s3_object_key


def test_get_s3_object_key_sanitizes_embedded_s3_uri_in_filename():
    start_time = datetime(2026, 3, 9, 17, 40, 11, 901585, tzinfo=timezone.utc)
    s3_path = "LiteLLMAPPLogs"
    prefix = ""
    s3_file_name = (
        "time-17-40-11-901585_s3://bucket-int/litellm-bedrock-files-us.anthropic."
        "claude-sonnet-4-5-20250929-v1-0-29ea93-452e-8a2f.jsonl.json"
    )

    s3_object_key = get_s3_object_key(
        s3_path=s3_path,
        prefix=prefix,
        start_time=start_time,
        s3_file_name=s3_file_name,
    )

    assert s3_object_key.startswith("LiteLLMAPPLogs/2026-03-09/")
    assert s3_object_key.endswith(".json")

    # The only path separators allowed are the ones we generate for prefix/date.
    file_component = s3_object_key[len("LiteLLMAPPLogs/2026-03-09/") :]
    assert "/" not in file_component
    assert "s3://" not in s3_object_key

