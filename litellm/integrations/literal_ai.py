#### What this does ####
# This file contains the LiteralAILogger class which is used to log steps to the LiteralAI observability platform.
import copy
import traceback


import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.redact_messages import redact_user_api_key_info


class LiteralAILogger:
    def __init__(
        self,
        literalai_api_key=None,
        literalai_api_url=None,
    ):
        try:
            from literalai import LiteralClient
        except Exception as e:
            raise Exception(
                f"\033[91mLiteralAI not installed, try running 'pip install literalai' to fix this error: {e}\n{traceback.format_exc()}\033[0m"
            )
    
        self.client = LiteralClient(api_key=literalai_api_key, url=literalai_api_url)

       
    @staticmethod
    def add_metadata_from_header(litellm_params: dict, metadata: dict) -> dict:
        """
        Adds metadata from proxy request headers to the generation if keys start with "literalai_"
        and overwrites litellm_params.metadata if already included.

        For example if you want to append your trace to an existing `thread_id` via header, send
        `headers: { ..., literalai_thread_id: thread_id }` via proxy request.
        """
        if litellm_params is None:
            return metadata

        if litellm_params.get("proxy_server_request") is None:
            return metadata

        if metadata is None:
            metadata = {}

        proxy_headers = (
            litellm_params.get("proxy_server_request", {}).get("headers", {}) or {}
        )

        for metadata_param_key in proxy_headers:
            if metadata_param_key.startswith("literalai_"):
                trace_param_key = metadata_param_key.replace("literalai_", "", 1)
                if trace_param_key in metadata:
                    verbose_logger.warning(
                        f"Overwriting LiteralAI `{trace_param_key}` from request header"
                    )
                else:
                    verbose_logger.debug(
                        f"Found LiteralAI `{trace_param_key}` in request header"
                    )
                metadata[trace_param_key] = proxy_headers.get(metadata_param_key)

        return metadata

    def create_step(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
        user_id,
        print_verbose,
        level="DEFAULT",
        status_message=None,
    ) -> dict:
        try:
            import uuid
            from literalai.observability.step import StepDict

            print_verbose(
                f"LiteralAI Logging - Enters logging function for model {kwargs}"
            )
            
            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )
            metadata = self.add_metadata_from_header(litellm_params, metadata)
            metadata["user_id"] = user_id
            clean_metadata = redact_user_api_key_info(metadata=metadata)

            settings = copy.deepcopy(kwargs.get("optional_params", {}))

            messages = kwargs.get("messages", None)
            
            prompt_id = None
            variables = None
            if "orig_messages" in clean_metadata:
                orig_messages = clean_metadata.pop("orig_messages")
                for index, message in enumerate(messages):
                    orig_message = orig_messages[index]
                    if literal_prompt := getattr(orig_message, "__literal_prompt__", None):
                        prompt_id = literal_prompt.get("prompt_id")
                        variables = literal_prompt.get("variables")
                        message["uuid"] = literal_prompt.get("uuid")
                        message["templated"] = True
            
            tools = settings.pop("tools", None)

            # only accepts str, int, bool, float for logging
            for param, value in settings.items():
                if not isinstance(value, (str, int, bool, float)):
                    try:
                        settings[param] = str(value)
                    except:
                        # if casting value to str fails don't block logging
                        pass
            step: StepDict =  {
                    "id": clean_metadata.get("step_id", str(uuid.uuid4())),
                    "name": kwargs.get("model", ""),
                    "threadId": clean_metadata.get("literalai_thread_id", None),
                    "parentId": clean_metadata.get("literalai_parent_id", None),
                    "rootRunId": clean_metadata.get("literalai_root_run_id", None),
                    "input": None,
                    "output": None,
                    "type": "undefined",
                    "tags": clean_metadata.get("tags", clean_metadata.get("literalai_tags", None)),
                    "startTime": str(start_time),
                    "endTime": str(end_time),
                    "metadata":  clean_metadata,
                    "generation": {
                        "promptId": prompt_id,
                        "variables": variables,
                        "provider": kwargs.get("custom_llm_provider", "litellm"),
                        "model": kwargs.get("model", ""),
                        "duration": (end_time - start_time).total_seconds(),
                        "settings": settings,
                        "messages": messages,
                        "tools": tools,
                    }
                }
            if response_obj is not None and response_obj.get("id", None) is not None:
                generation_id = litellm.utils.get_logging_id(start_time, response_obj)
                step["metadata"]["litellm_id"] = generation_id
                step["generation"]["inputTokenCount"] = response_obj.usage.prompt_tokens
                step["generation"]["outputTokenCount"] = response_obj.usage.completion_tokens
                step["generation"]["tokenCount"] = response_obj.usage.prompt_tokens + response_obj.usage.completion_tokens

            if (
                level == "ERROR"
                and status_message is not None
                and isinstance(status_message, str)
            ):
                step["error"] = status_message
            elif response_obj is not None and (
                kwargs.get("call_type", None) == "embedding"
                or isinstance(response_obj, litellm.EmbeddingResponse)
            ):
                step["type"] = "embedding"
                step["input"] = messages
                step["output"] = response_obj.data
            elif response_obj is not None and isinstance(
                response_obj, litellm.ModelResponse
            ):
                step["type"] = "llm"
                step["generation"]["messageCompletion"] = response_obj["choices"][0]["message"].json()
            elif response_obj is not None and isinstance(
                response_obj, litellm.TextCompletionResponse
            ):
                step["type"] = "llm"
                step["generation"]["completion"] = response_obj.choices[0].text
            elif response_obj is not None and isinstance(
                response_obj, litellm.ImageResponse
            ):
                pass
            elif response_obj is not None and isinstance(
                response_obj, litellm.TranscriptionResponse
            ):
                pass
            elif (
                kwargs.get("call_type") is not None
                and kwargs.get("call_type") == "pass_through_endpoint"
                and response_obj is not None
                and isinstance(response_obj, dict)
            ):
                pass
            self.client.api.send_steps([step])
            
        except Exception as e:
            verbose_logger.exception(
                "Literal AI Layer Error(): Exception occured - {}".format(str(e))
            )

