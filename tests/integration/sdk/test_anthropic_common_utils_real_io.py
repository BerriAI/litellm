import json
import subprocess
import sys
from typing import Final


_MASKED_HOSTED_WEB_CALL_PROBE: Final = """
import json

from litellm.llms.anthropic.common_utils import write_request_hosted_web_call_arguments

message = {
    "role": "assistant",
    "provider_specific_fields": {
        "web_search_calls": [
            {
                "id": "ws_srvtoolu_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "deploy token 12345", "queries": ["deploy token 12345"]},
            }
        ]
    },
}
write_request_hosted_web_call_arguments(message, {"srvtoolu_1": json.dumps({"query": "deploy token [REDACTED]"})})
print(json.dumps(message["provider_specific_fields"]["web_search_calls"]))
"""


def test_a_masked_replayed_hosted_web_call_serializes_before_its_openai_model_was_ever_built() -> None:
    result: Final = subprocess.run(
        (sys.executable, "-I", "-c", _MASKED_HOSTED_WEB_CALL_PROBE),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    hosted: Final = json.loads(result.stdout.strip().splitlines()[-1])
    assert [call["action"] for call in hosted] == [
        {"type": "search", "query": "deploy token [REDACTED]", "queries": ["deploy token [REDACTED]"]}
    ]
