import asyncio
import json
import sys
from typing import Final

import httpx
import httpx2
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, JsonValue, SecretStr
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from .agent import ToolContext, run_agent
from .models import Action, ActionResult, ChatRequest, Decision


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chat: ChatRequest
    access_token: SecretStr
    headers: dict[str, str]
    management_url: str
    inference_url: str


def emit(value: JsonValue) -> None:
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()


async def confirm(action: Action) -> bool:
    emit({"type": "approval", "action": action.model_dump()})
    decision: Final = Decision.model_validate_json(await asyncio.to_thread(sys.stdin.readline))
    return decision.id == action.id and decision.approved


async def on_result(action: Action, result: ActionResult) -> None:
    emit({"type": "result", "action": action.model_dump(), "result": result.model_dump(exclude_none=True)})


async def main() -> None:
    request: Final = WorkerRequest.model_validate_json(await asyncio.to_thread(sys.stdin.readline))
    async with (
        httpx.AsyncClient(
            base_url=request.management_url,
            headers=request.headers,
            timeout=60,
            follow_redirects=False,
            trust_env=False,
        ) as management,
        httpx2.AsyncClient(timeout=60, follow_redirects=False, trust_env=False) as inference,
        AsyncOpenAI(
            api_key=request.access_token.get_secret_value(),
            base_url=request.inference_url.rstrip("/") + "/",
            http_client=inference,
            default_headers=request.headers,
            max_retries=0,
            timeout=60,
        ) as model_client,
    ):
        context: Final = ToolContext(
            management, (request.access_token.get_secret_value(),), confirm, on_result, asyncio.Lock()
        )
        model: Final = OpenAIChatModel(
            request.chat.model,
            provider=OpenAIProvider(openai_client=model_client),
            settings=OpenAIChatModelSettings(parallel_tool_calls=False, max_tokens=2048, timeout=60),
        )
        async for text in run_agent(request.chat, context, model):
            emit({"type": "message", "text": text})
        emit({"type": "done"})


if __name__ == "__main__":
    asyncio.run(main())
