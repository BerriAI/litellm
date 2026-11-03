from litellm.types.integrations.custom_logger import AgenticLoopPlan, AgenticLoopRequestPatch
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.results import response_id, with_id


def agentic_hooks(recorder: RecordingLogger) -> tuple[str, ...]:
    return tuple(name for name in recorder.names if "agentic" in name)


class PlannedAgenticLoop(RecordingLogger):
    async def async_should_run_agentic_loop(self, response: object, **kwargs: object) -> tuple[bool, dict[str, object]]:
        self._record("async_should_run_agentic_loop", response=response)
        return response_id(response) != "followup", {}

    async def async_build_agentic_loop_plan(self, response: object, **kwargs: object) -> AgenticLoopPlan:
        self._record("async_build_agentic_loop_plan", response=response)
        return AgenticLoopPlan(
            run_agentic_loop=True,
            request_patch=AgenticLoopRequestPatch.model_validate(
                {"messages": [{"role": "user", "content": "followup"}]}
            ),
        )

    async def async_post_agentic_loop_response_hook(self, response: object, **kwargs: object) -> object:
        self._record("async_post_agentic_loop_response_hook", response=response)
        return with_id(response, "agentic-replaced")

    async def async_agentic_loop_cleanup_hook(self, **kwargs: object) -> None:
        self._record("async_agentic_loop_cleanup_hook")


class LegacyAgenticLoop(RecordingLogger):
    async def async_should_run_agentic_loop(self, response: object, **kwargs: object) -> tuple[bool, dict[str, object]]:
        self._record("async_should_run_agentic_loop", response=response)
        return True, {}

    async def async_run_agentic_loop(self, response: object, **kwargs: object) -> object:
        self._record("async_run_agentic_loop", response=response)
        return with_id(response, "agentic-replaced")
