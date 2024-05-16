#### What this does ####
#    On success + failure, log events to lunary.ai
from datetime import datetime, timezone
import traceback
import importlib

import packaging
from litellm._logging import verbose_logger
from typing import Optional, Literal
import litellm

############################################################
def print_verbose(
    print_statement,
    logger_only: bool = False,
    log_level: Literal["DEBUG", "INFO"] = "DEBUG",
):
    try:
        if log_level == "DEBUG":
            verbose_logger.debug(print_statement)
        elif log_level == "INFO":
            verbose_logger.info(print_statement)
        if litellm.set_verbose == True and logger_only == False:
            print(print_statement)  # noqa
    except:
        pass


####### LOGGING ###################

# convert to {completion: xx, tokens: xx}
def parse_usage(usage):
    return {
        "completion": usage["completion_tokens"] if "completion_tokens" in usage else 0,
        "prompt": usage["prompt_tokens"] if "prompt_tokens" in usage else 0,
    }

def parse_tool_calls(tool_calls):
    if tool_calls is None:
        return None

    def clean_tool_call(tool_call):

        serialized = {
            "type": tool_call.type,
            "id": tool_call.id,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
        }

        return serialized
    
    return [clean_tool_call(tool_call) for tool_call in tool_calls]
    

def parse_messages(input):

    if input is None:
        return None

    def clean_message(message):
        # if is string, return as is
        if isinstance(message, str):
            return message

        if "message" in message:
            return clean_message(message["message"])

        serialized = {
            "role": message.get("role"),
            "content": message.get("content"),
        }

        # Only add tool_calls and function_call to res if they are set
        if message.get("tool_calls"):
            serialized["tool_calls"] = parse_tool_calls(message.get("tool_calls"))

        return serialized

    if isinstance(input, list):
        if len(input) == 1:
            return clean_message(input[0])
        else:
            return [clean_message(msg) for msg in input]
    else:
        return clean_message(input)


class LunaryLogger:
    # Class variables or attributes
    def __init__(self):
        try:
            import lunary

            version = importlib.metadata.version("lunary")
            # if version < 0.1.43 then raise ImportError
            if packaging.version.Version(version) < packaging.version.Version("0.1.43"):
                print_verbose(  # noqa
                    "Lunary version outdated. Required: >= 0.1.43. Upgrade via 'pip install lunary --upgrade'"
                )
                raise ImportError

            self.lunary_client = lunary
        except ImportError:
            print_verbose(  # noqa
                "Lunary not installed. Please install it using 'pip install lunary'"
            )  # noqa
            raise ImportError

    def log_event(
        self,
        kwargs,
        type,
        event,
        run_id,
        model,
        print_verbose,
        extra=None,
        input=None,
        user_id=None,
        response_obj=None,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        error=None,
    ):
        # Method definition
        try:
            print_verbose(f"Lunary Logging - Logging request for model {model}")

            litellm_params = kwargs.get("litellm_params", {})
            optional_params = kwargs.get("optional_params", {})
            metadata = litellm_params.get("metadata", {}) or {}

            if optional_params:
                # merge into extra
                extra = {**extra, **optional_params}

            tags = litellm_params.pop("tags", None) or []

            if extra:
                extra.pop("extra_body", None)
                extra.pop("user", None)
                template_id = extra.pop("extra_headers", {}).get("Template-Id", None)

            # keep only serializable types
            for param, value in extra.items():
                if not isinstance(value, (str, int, bool, float)) and param != "tools":
                    try:
                        extra[param] = str(value)
                    except:
                        pass

            if response_obj:
                usage = (
                    parse_usage(response_obj["usage"])
                    if "usage" in response_obj
                    else None
                )

                output = response_obj["choices"] if "choices" in response_obj else None

            else:
                usage = None
                output = None

            if error:
                error_obj = {"stack": error}
            else:
                error_obj = None

            self.lunary_client.track_event(
                type,
                "start",
                run_id,
                user_id=user_id,
                name=model,
                input=parse_messages(input),
                timestamp=start_time.astimezone(timezone.utc).isoformat(),
                template_id=template_id,
                metadata=metadata,
                runtime="litellm",
                tags=tags,
                params=extra,
            )

            self.lunary_client.track_event(
                type,
                event,
                run_id,
                timestamp=end_time.astimezone(timezone.utc).isoformat(),
                runtime="litellm",
                error=error_obj,
                output=parse_messages(output),
                token_usage=usage,
            )

        except:
            # traceback.print_exc()
            print_verbose(f"Lunary Logging Error - {traceback.format_exc()}")
            pass
