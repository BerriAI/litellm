"""Live pipecat smoke for the proxy realtime websocket.

A realism layer on top of test_realtime_e2e: instead of speaking the GA protocol
by hand, drive the proxy through the shared LiteLLMRealtimeLLMService (pipecat's
GA service with proxy-specific overrides, keepalive pings disabled) with its
base_url pointed at the proxy and the model swapped per provider. It confirms the
audio and function-call wiring survives the round-trip. Assertions are coarse;
the raw-websocket suite is the source of truth.

The harness is synchronous, so each test stays a normal sync function and drives
the async pipecat pipeline with asyncio.run. Skips unless pipecat is installed:

    uv pip install "pipecat-ai[openai]"

Known caveat: pipecat tool calling over the realtime service has been flaky
upstream (pipecat-ai/pipecat#2544). A failure here with the matching raw-ws tool
test passing points at pipecat, not litellm.
"""

# pipecat is an optional, dynamically typed dependency loaded behind importorskip,
# so its symbols are Unknown to the type checker; relax those rules for this file.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUntypedBaseClass=false, reportUnknownParameterType=false, reportMissingParameterType=false

import asyncio

import pytest

from realtime_client import (
    PROVIDERS,
    RealtimeProvider,
    _ws_base_url,
    skip_if_unconfigured,
)

pytestmark = pytest.mark.e2e

pytest.importorskip("pipecat", reason="pipecat-ai not installed")

from pipecat.adapters.schemas.function_schema import FunctionSchema  # noqa: E402
from pipecat.adapters.schemas.tools_schema import ToolsSchema  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import (  # noqa: E402
    FrameDirection,
    FrameProcessor,
)
from pipecat.services.llm_service import FunctionCallParams  # noqa: E402

from pipecat_service import LiteLLMRealtimeLLMService  # noqa: E402

PROVIDER_PARAMS = [pytest.param(p, id=p.id) for p in PROVIDERS]

WEATHER_TOOL = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="get_weather",
            description="Get the current temperature in Fahrenheit for a city.",
            properties={"city": {"type": "string"}},
            required=["city"],
        )
    ]
)


class _CaptureText(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSTextFrame, TranscriptionFrame)):
            self.texts.append(frame.text)
        await self.push_frame(frame, direction)


async def _run_pipeline(key: str, model: str) -> tuple[bool, bool]:
    tool_called = asyncio.Event()

    async def get_weather(params: FunctionCallParams) -> None:
        tool_called.set()
        await params.result_callback({"city": "Paris", "temperature_f": 72})

    llm = LiteLLMRealtimeLLMService(
        api_key=key, base_url=f"{_ws_base_url()}/v1/realtime", model=model
    )
    llm.register_function("get_weather", get_weather)

    context = LLMContext(tools=WEATHER_TOOL)
    aggregator = LLMContextAggregatorPair(context)
    capture = _CaptureText()
    task = PipelineTask(
        Pipeline([aggregator.user(), llm, capture, aggregator.assistant()])
    )

    await task.queue_frames(
        [
            TranscriptionFrame(
                "What's the weather in Paris?", user_id="e2e", timestamp=""
            ),
            LLMRunFrame(),
        ]
    )
    try:
        await asyncio.wait_for(PipelineRunner().run(task), timeout=45)
    except asyncio.TimeoutError:
        await task.queue_frame(EndFrame())
    return tool_called.is_set(), bool(capture.texts)


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_pipecat_tool_smoke(
    scoped_key: str,
    configured_models: frozenset[str],
    provider: RealtimeProvider,
) -> None:
    skip_if_unconfigured(provider, configured_models)

    tool_called, produced_text = asyncio.run(_run_pipeline(scoped_key, provider.model))

    assert tool_called, "pipecat did not invoke the get_weather callback"
    assert produced_text, "pipecat produced no assistant text frames"
