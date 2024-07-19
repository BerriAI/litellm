imported_openAIResponse = True
try:
    import io
    import logging
    import sys
    from typing import Any, Dict, List, Optional, TypeVar

    from wandb.sdk.data_types import trace_tree

    if sys.version_info >= (3, 8):
        from typing import Literal, Protocol
    else:
        from typing_extensions import Literal, Protocol

    logger = logging.getLogger(__name__)

    K = TypeVar("K", bound=str)
    V = TypeVar("V")

    class OpenAIResponse(Protocol[K, V]):  # type: ignore
        # contains a (known) object attribute
        object: Literal["chat.completion", "edit", "text_completion"]

        def __getitem__(self, key: K) -> V: ...  # noqa

        def get(  # noqa
            self, key: K, default: Optional[V] = None
        ) -> Optional[V]: ...  # pragma: no cover

    class OpenAIRequestResponseResolver:
        def __call__(
            self,
            request: Dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> Optional[trace_tree.WBTraceTree]:
            try:
                if response["object"] == "edit":
                    return self._resolve_edit(request, response, time_elapsed)
                elif response["object"] == "text_completion":
                    return self._resolve_completion(request, response, time_elapsed)
                elif response["object"] == "chat.completion":
                    return self._resolve_chat_completion(
                        request, response, time_elapsed
                    )
                else:
                    logger.info(f"Unknown OpenAI response object: {response['object']}")
            except Exception as e:
                logger.warning(f"Failed to resolve request/response: {e}")
            return None

        @staticmethod
        def results_to_trace_tree(
            request: Dict[str, Any],
            response: OpenAIResponse,
            results: List[trace_tree.Result],
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
            start_time_ms = int(round(response["created"] * 1000))
            end_time_ms = start_time_ms + int(round(time_elapsed * 1000))
            span = trace_tree.Span(
                name=f"{response.get('model', 'openai')}_{response['object']}_{response.get('created')}",
                attributes=dict(response),  # type: ignore
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                span_kind=trace_tree.SpanKind.LLM,
                results=results,
            )
            model_obj = {"request": request, "response": response, "_kind": "openai"}
            return trace_tree.WBTraceTree(root_span=span, model_dict=model_obj)

        def _resolve_edit(
            self,
            request: Dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Edit`."""
            request_str = (
                f"\n\n**Instruction**: {request['instruction']}\n\n"
                f"**Input**: {request['input']}\n"
            )
            choices = [
                f"\n\n**Edited**: {choice['text']}\n" for choice in response["choices"]
            ]

            return self._request_response_result_to_trace(
                request=request,
                response=response,
                request_str=request_str,
                choices=choices,
                time_elapsed=time_elapsed,
            )

        def _resolve_completion(
            self,
            request: Dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Completion`."""
            request_str = f"\n\n**Prompt**: {request['prompt']}\n"
            choices = [
                f"\n\n**Completion**: {choice['text']}\n"
                for choice in response["choices"]
            ]

            return self._request_response_result_to_trace(
                request=request,
                response=response,
                request_str=request_str,
                choices=choices,
                time_elapsed=time_elapsed,
            )

        def _resolve_chat_completion(
            self,
            request: Dict[str, Any],
            response: OpenAIResponse,
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Completion`."""
            prompt = io.StringIO()
            for message in request["messages"]:
                prompt.write(f"\n\n**{message['role']}**: {message['content']}\n")
            request_str = prompt.getvalue()

            choices = [
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
            request: Dict[str, Any],
            response: OpenAIResponse,
            request_str: str,
            choices: List[str],
            time_elapsed: float,
        ) -> trace_tree.WBTraceTree:
            """Resolves the request and response objects for `openai.Completion`."""
            results = [
                trace_tree.Result(
                    inputs={"request": request_str},
                    outputs={"response": choice},
                )
                for choice in choices
            ]
            trace = self.results_to_trace_tree(request, response, results, time_elapsed)
            return trace

except:
    imported_openAIResponse = False


#### What this does ####
#    On success, logs events to Langfuse
import os
import requests
import requests
from datetime import datetime

import traceback


class WeightsBiasesLogger:
    # Class variables or attributes
    def __init__(self):
        try:
            import wandb
        except:
            raise Exception(
                "\033[91m wandb not installed, try running 'pip install wandb' to fix this error\033[0m"
            )
        if imported_openAIResponse == False:
            raise Exception(
                "\033[91m wandb not installed, try running 'pip install wandb' to fix this error\033[0m"
            )
        self.resolver = OpenAIRequestResponseResolver()

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        import wandb

        try:
            print_verbose(f"W&B Logging - Enters logging function for model {kwargs}")
            run = wandb.init()
            print_verbose(response_obj)

            trace = self.resolver(
                kwargs, response_obj, (end_time - start_time).total_seconds()
            )

            if trace is not None:
                run.log({"trace": trace})

            run.finish()
            print_verbose(
                f"W&B Logging Logging - final response object: {response_obj}"
            )
        except:
            print_verbose(f"W&B Logging Layer Error - {traceback.format_exc()}")
            pass
