"""Layer 2: pipecat full-stack smoke through the proxy, per provider.

A realism check on top of the raw-WS suite: drive the proxy with pipecat's GA
OpenAIRealtimeLLMService (one OpenAI-protocol client, base_url pointed at the
proxy, model swapped per provider) and confirm the audio/function-call wiring
survives the round-trip. Assertions are deliberately coarse -> the raw-WS suite
is the source of truth; this only proves pipecat can talk to the proxy and that
a registered tool callback fires.

Skips unless pipecat is installed:
    poetry run pip install "pipecat-ai[openai]"

Known caveat: pipecat tool calling over the realtime service has been flaky
upstream (pipecat-ai/pipecat#2544). Treat a failure here as "investigate
pipecat first", not "litellm regression", until the raw-WS tool test also fails.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlencode

import pytest

from .conftest import skip_if_creds_missing
from .providers import PROVIDER_IDS, PROVIDERS, RealtimeProvider

pipecat = pytest.importorskip("pipecat", reason="pipecat-ai not installed")

from pipecat.adapters.schemas.function_schema import FunctionSchema  # noqa: E402
from pipecat.adapters.schemas.tools_schema import ToolsSchema  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.openai_llm_context import (  # noqa: E402
    OpenAILLMContext,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService  # noqa: E402

pytestmark = [pytest.mark.realtime_e2e, pytest.mark.asyncio]


class _Capture(FrameProcessor):
    """Records assistant text frames so the test can assert the model spoke."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSTextFrame, TranscriptionFrame)):
            self.texts.append(frame.text)
        await self.push_frame(frame, direction)


@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDER_IDS)
async def test_pipecat_tool_smoke(
    provider: RealtimeProvider, proxy_ws_url: str, proxy_api_key: str
) -> None:
    skip_if_creds_missing(provider)

    tool_called = asyncio.Event()

    async def get_weather(params) -> None:
        tool_called.set()
        await params.result_callback({"city": "Paris", "temperature_f": 72})

    base_url = f"{proxy_ws_url.rstrip('/')}/v1/realtime?{urlencode({'model': provider.model})}"
    llm = OpenAIRealtimeLLMService(
        api_key=proxy_api_key,
        base_url=base_url,
        model=provider.model,
    )
    llm.register_function("get_weather", get_weather)

    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="get_weather",
                description="Get the current temperature in Fahrenheit for a city.",
                properties={"city": {"type": "string"}},
                required=["city"],
            )
        ]
    )
    context = OpenAILLMContext(tools=tools)
    aggregator = llm.create_context_aggregator(context)

    capture = _Capture()
    pipeline = Pipeline([aggregator.user(), llm, capture, aggregator.assistant()])
    task = PipelineTask(pipeline)

    await task.queue_frames(
        [
            TranscriptionFrame(
                "What's the weather in Paris?", user_id="test", timestamp=""
            ),
            LLMRunFrame(),
        ]
    )

    runner = PipelineRunner()
    try:
        await asyncio.wait_for(runner.run(task), timeout=45)
    except asyncio.TimeoutError:
        await task.queue_frame(EndFrame())

    assert tool_called.is_set(), "pipecat did not invoke the get_weather callback"
    assert capture.texts, "pipecat produced no assistant text frames"
