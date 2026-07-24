from typing import Optional, Iterator, Dict, Any
from litellm.llms.base_llm.chat.transformation import BaseConfig


class SnowflakeBase(BaseConfig):
    def validate_environment(
        self,
        headers: dict,
        JWT: Optional[str] = None,
    ) -> dict:
        """
        Return headers to use for Snowflake completion request

        Snowflake REST API Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#api-reference
        Expected headers:
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + <JWT>,
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT"
        }
        """

        if JWT is None:
            raise ValueError("Missing Snowflake JWT key")

        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + JWT,
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            }
        )
        return headers

    def chunk_parser(self, chunk: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Parse Snowflake streaming chunks to emit OpenAI-style delta."""
        # Snowflake streaming returns events with data: {"type": "delta", ...}
        # or {"type": "tool_use", ...}
        if "type" not in chunk:
            return
        if chunk["type"] == "delta":
            yield {"choices": [{"delta": {"content": chunk.get("delta", "")}}]}
        elif chunk["type"] == "tool_use":
            # Snowflake tool_use format: {"type": "tool_use", "name": ..., "input": ...}
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": None,
                                    "function": {
                                        "name": chunk.get("name", ""),
                                        "arguments": chunk.get("input", ""),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
