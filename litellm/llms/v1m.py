import json
import time

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class V1MError(BaseLLMException):
    def __init__(self, status_code: int, message: str, headers: object = None) -> None:
        super().__init__(status_code=status_code, message=message, headers=headers)


class V1MConfig(OpenAIGPTConfig):
    DEFAULT_BASE_URL: str = "https://v1m.ir/v1/systemone"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        if not self.base_url.endswith("/systemone"):
            self.base_url = f"{self.base_url}/systemone"
        self.timeout = timeout

    def transform_request(
        self,
        model: str,
        messages: list[dict] | None = None,
        state: str | dict | None = None,
        questions: dict | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        clean_model = model.replace("v1m/", "") if model.startswith("v1m/") else model

        if state is not None and questions is not None:
            return {
                "model": clean_model or "rev-latest",
                "state": state,
                "questions": questions,
            }

        extracted_state = ""
        user_prompt = ""
        if messages:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    extracted_state = content
                elif role == "user":
                    user_prompt = content

        return {
            "model": clean_model or "rev-latest",
            "state": extracted_state or user_prompt or "General State Assessment",
            "questions": {
                "decision": {
                    "prompt": user_prompt = "Evaluate state validity.",
                    "instructions": "Determine probabilistic confidence and classification.",
                    "type": "choice",
                    "criteria": {
                        "affirmative": "Conditions met",
                        "negative": "Conditions not met",
                    },
                }
            },
        }

    def transform_response(self, raw_resp: dict, model: str) -> dict:
        evals = raw_resp.get("evaluations", raw_resp)
        return {
            "id": f"v1m-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(evals, ensure_ascii=False),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": raw_resp.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            "systemone_evaluations": evals,
        }
