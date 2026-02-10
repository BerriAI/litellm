import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import litellm

from litellm.integrations.s3 import S3Logger


def test_s3_prompts_only_payload():
    mock_client = MagicMock()
    mock_client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

    with patch("boto3.client", return_value=mock_client):
        litellm.s3_callback_params = {
            "s3_bucket_name": "test-bucket",
            "s3_region_name": "us-east-1",
            "s3_log_prompts_only": True,
        }

        logger = S3Logger()

        kwargs = {
            "standard_logging_object": {
                "id": "test-id",
                "metadata": {},
                "messages": [{"role": "user", "content": "hello"}],
                "response": {"id": "resp"},
            }
        }

        logger.log_event(
            kwargs=kwargs,
            response_obj={},
            start_time=datetime(2026, 2, 10, 15, 0, 0),
            end_time=datetime(2026, 2, 10, 15, 0, 1),
            print_verbose=lambda *args, **kwargs: None,
        )

    call_args = mock_client.put_object.call_args
    assert call_args is not None
    payload_str = call_args.kwargs["Body"]
    payload = json.loads(payload_str)

    assert payload == {"messages": [{"role": "user", "content": "hello"}]}

    litellm.s3_callback_params = None
