#### What this does ####
#    On success, logs events to Langfuse
import dotenv, os
import requests
import requests
from datetime import datetime

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
from packaging.version import Version


class LangFuseLogger:
    # Class variables or attributes
    def __init__(self):
        try:
            from langfuse import Langfuse
        except Exception as e:
            raise Exception(
                f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\033[0m"
            )
        # Instance variables
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")
        self.langfuse_debug = os.getenv("LANGFUSE_DEBUG")
        self.Langfuse = Langfuse(
            public_key=self.public_key,
            secret_key=self.secret_key,
            host=self.langfuse_host,
            release=self.langfuse_release,
            debug=True,
        )

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition

        try:
            print_verbose(
                f"Langfuse Logging - Enters logging function for model {kwargs}"
            )
            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            prompt = [kwargs.get("messages")]
            optional_params = kwargs.get("optional_params", {})

            optional_params.pop("functions", None)
            optional_params.pop("tools", None)

            # langfuse only accepts str, int, bool, float for logging
            for param, value in optional_params.items():
                if not isinstance(value, (str, int, bool, float)):
                    try:
                        optional_params[param] = str(value)
                    except:
                        # if casting value to str fails don't block logging
                        pass

            # end of processing langfuse ########################
            input = prompt
            output = response_obj["choices"][0]["message"].json()

            self._log_langfuse_v2(
                metadata,
                output,
                start_time,
                end_time,
                kwargs,
                optional_params,
                input,
                response_obj,
            ) if self._is_langfuse_v2() else self._log_langfuse_v1(
                metadata,
                output,
                start_time,
                end_time,
                kwargs,
                optional_params,
                input,
                response_obj,
            )

            self.Langfuse.flush()
            print_verbose(
                f"Langfuse Layer Logging - final response object: {response_obj}"
            )
        except:
            traceback.print_exc()
            print_verbose(f"Langfuse Layer Error - {traceback.format_exc()}")
            pass

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose
    ):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def _is_langfuse_v2(self):
        import langfuse

        return Version(langfuse.version.__version__) >= Version("2.0.0")

    def _log_langfuse_v1(
        self,
        metadata,
        output,
        start_time,
        end_time,
        kwargs,
        optional_params,
        input,
        response_obj,
    ):
        from langfuse.model import CreateTrace, CreateGeneration

        trace = self.Langfuse.trace(
            CreateTrace(
                name=metadata.get("generation_name", "litellm-completion"),
                input=input,
                output=output,
            )
        )

        trace.generation(
            CreateGeneration(
                name=metadata.get("generation_name", "litellm-completion"),
                startTime=start_time,
                endTime=end_time,
                model=kwargs["model"],
                modelParameters=optional_params,
                input=input,
                output=output,
                usage={
                    "prompt_tokens": response_obj["usage"]["prompt_tokens"],
                    "completion_tokens": response_obj["usage"]["completion_tokens"],
                },
                metadata=metadata,
            )
        )

    def _log_langfuse_v2(
        self,
        metadata,
        output,
        start_time,
        end_time,
        kwargs,
        optional_params,
        input,
        response_obj,
    ):
        trace = self.Langfuse.trace(
            name=metadata.get("generation_name", "litellm-completion"),
            input=input,
            output=output,
        )

        trace.generation(
            name=metadata.get("generation_name", "litellm-completion"),
            startTime=start_time,
            endTime=end_time,
            model=kwargs["model"],
            modelParameters=optional_params,
            input=input,
            output=output,
            usage={
                "prompt_tokens": response_obj["usage"]["prompt_tokens"],
                "completion_tokens": response_obj["usage"]["completion_tokens"],
            },
            metadata=metadata,
        )
