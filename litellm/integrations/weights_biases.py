from typing import Final

imported_openAIResponse = True
try:
    import io
    import logging
    import sys
    from typing import Any, TypeVar

    from wandb.sdk.data_types import trace_tree

    if sys.version_info >= (3, 8):
        from typing import Literal, Protocol
    else:
        from typing import Literal

        from typing_extensions import Protocol

    logger: Final = logging.getLogger(__name__)

    K = TypeVar("K", bound=str)
    V = TypeVar("V")

    class OpenAIResponse(Protocol[K, V]):
        # contains a (known) object attribute
        object: Literal["chat.completion", "edit", "text_completion"]

        def __getitem__(self, key: K) -> V: ...

        def get(self, key: K, default: V | None = None) -> V | None: ...  # pragma: no cover

    class OpenAIRequestResponseResolver:
        def __call__(
            self,
            request: dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree | None:
            try:
                if response["object"] == "edit":
                    return self._resolve_edit(request, response, time_elapsed)
                elif response["object"] == "text_completion":
                    return self._resolve_completion(request, response, time_elapsed)
                elif response["object"] == "chat.completion":
                    return self._resolve_chat_completion(request, response, time_elapsed)
                else:
                    logger.debug("Unknown OpenAI response object: %s", response["object"])
            except Exception as e:
                logger.warning("Failed to resolve request/response: %s", e)
            return None

        @staticmethod
        def results_to_trace_tree(
            request: dict[str, Any],
            response: OpenAIResponse,
            results: list[trace_tree.Result],
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Converts the request, response, and results into a trace tree.

            params:
                request: The request dictionary
                response: The response object
                results: A list of results object
                time_elapsed: The time elapsed in seconds
            returns:
                A wandb trace tree object.
            """
            start_time_ms: Final = int(round(response["created"] * 1000))
            end_time_ms: Final = start_time_ms + int(round(time_elapsed * 1000))
            span: Final = trace_tree.Span(
                name=f"{response.get('model', 'openai')}_{response['object']}_{response.get('created')}",
                attributes=dict(response),
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                span_kind=trace_tree.SpanKind.LLM,
                results=results,
            )
            model_obj: Final = {"request": request, "response": response, "_kind": "openai"}
            return trace_tree.WBTraceTree(root_span=span, model_dict=model_obj)

        def _resolve_edit(
            self,
            request: dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Edit`."""
            request_str: Final = f"\n\n**Instruction**: {request['instruction']}\n\n**Input**: {request['input']}\n"
            choices: Final = [f"\n\n**Edited**: {choice['text']}\n" for choice in response["choices"]]

            return self._request_response_result_to_trace(
                request=request,
                response=response,
                request_str=request_str,
                choices=choices,
                time_elapsed=time_elapsed,
            )

        def _resolve_completion(
            self,
            request: dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Completion`."""
            request_str: Final = f"\n\n**Prompt**: {request['prompt']}\n"
            choices: Final = [f"\n\n**Completion**: {choice['text']}\n" for choice in response["choices"]]

            return self._request_response_result_to_trace(
                request=request,
                response=response,
                request_str=request_str,
                choices=choices,
                time_elapsed=time_elapsed,
            )

        def _resolve_chat_completion(
            self,
            request: dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Completion`."""
            prompt: Final = io.StringIO()
            for message in request["messages"]:
                prompt.write(f"\n\n**{message['role']}**: {message['content']}\n")
            request_str: Final = prompt.getvalue()

            choices: Final = [
                f"\n\n**{choice['message']['role']}**: {choice['message']['content']}\n"
                for choice in response["choices"]
            ]

            return self._request_response_result_to_trace(
                request=request,
                response=response,
                request_str=request_str,
                choices=choices,
                time_elapsed=time_elapsed,
            )

        def _request_response_result_to_trace(
            self,
            request: dict[str, Any],
            response: OpenAIResponse,
            request_str: str,
            choices: list[str],
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Completion`."""
            results: Final = [
                trace_tree.Result(
                    inputs={"request": request_str},
                    outputs={"response": choice},
                )
                for choice in choices
            ]
            trace: Final = self.results_to_trace_tree(request, response, results, time_elapsed)
            return trace

except Exception:
    imported_openAIResponse = False


#### What this does ####
#    On success, logs events to Langfuse
import traceback


class WeightsBiasesLogger:
    # Class variables or attributes
    def __init__(self):
        try:
            pass
        except Exception:
            raise Exception("\033[91m wandb not installed, try running 'pip install wandb' to fix this error\033[0m")
        if imported_openAIResponse is False:
            raise Exception("\033[91m wandb not installed, try running 'pip install wandb' to fix this error\033[0m")
        self.resolver = OpenAIRequestResponseResolver()

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        import wandb

        try:
            print_verbose(f"W&B Logging - Enters logging function for model {kwargs}")
            run: Final = wandb.init()
            print_verbose(response_obj)

            trace: Final = self.resolver(kwargs, response_obj, (end_time - start_time).total_seconds())

            if trace is not None and run is not None:
                run.log({"trace": trace})

            if run is not None:
                run.finish()
                print_verbose(f"W&B Logging Logging - final response object: {response_obj}")
        except Exception:
            print_verbose(f"W&B Logging Layer Error - {traceback.format_exc()}")
