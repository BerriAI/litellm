#### What this does ####
#    On success, logs events to Langfuse
import dotenv, os
import requests
import requests
from datetime import datetime

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
from packaging.version import Version
from litellm._logging import verbose_logger
import litellm


class LangFuseLogger:
    # Class variables or attributes
    def __init__(self, langfuse_public_key=None, langfuse_secret=None):
        try:
            from langfuse import Langfuse
        except Exception as e:
            raise Exception(
                f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\033[0m"
            )
        # Instance variables
        self.secret_key = langfuse_secret or os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")
        self.langfuse_debug = os.getenv("LANGFUSE_DEBUG")
        self.Langfuse = Langfuse(
            public_key=self.public_key,
            secret_key=self.secret_key,
            host=self.langfuse_host,
            release=self.langfuse_release,
            debug=self.langfuse_debug,
        )

        if os.getenv("UPSTREAM_LANGFUSE_SECRET_KEY") is not None:
            self.upstream_langfuse_secret_key = os.getenv(
                "UPSTREAM_LANGFUSE_SECRET_KEY"
            )
            self.upstream_langfuse_public_key = os.getenv(
                "UPSTREAM_LANGFUSE_PUBLIC_KEY"
            )
            self.upstream_langfuse_host = os.getenv("UPSTREAM_LANGFUSE_HOST")
            self.upstream_langfuse_release = os.getenv("UPSTREAM_LANGFUSE_RELEASE")
            self.upstream_langfuse_debug = os.getenv("UPSTREAM_LANGFUSE_DEBUG")
            self.upstream_langfuse = Langfuse(
                public_key=self.upstream_langfuse_public_key,
                secret_key=self.upstream_langfuse_secret_key,
                host=self.upstream_langfuse_host,
                release=self.upstream_langfuse_release,
                debug=self.upstream_langfuse_debug,
            )
        else:
            self.upstream_langfuse = None

    def log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
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
            if kwargs.get("call_type", None) == "embedding" or isinstance(
                response_obj, litellm.EmbeddingResponse
            ):
                input = prompt
                output = response_obj["data"]
            else:
                input = prompt
                output = response_obj["choices"][0]["message"].json()
            print_verbose(f"OUTPUT IN LANGFUSE: {output}; original: {response_obj}")
            self._log_langfuse_v2(
                user_id,
                metadata,
                output,
                start_time,
                end_time,
                kwargs,
                optional_params,
                input,
                response_obj,
                print_verbose,
            ) if self._is_langfuse_v2() else self._log_langfuse_v1(
                user_id,
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
            verbose_logger.info(f"Langfuse Layer Logging - logging success")
        except:
            traceback.print_exc()
            print_verbose(f"Langfuse Layer Error - {traceback.format_exc()}")
            pass

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, user_id, print_verbose
    ):
        self.log_event(
            kwargs, response_obj, start_time, end_time, user_id, print_verbose
        )

    def _is_langfuse_v2(self):
        import langfuse

        return Version(langfuse.version.__version__) >= Version("2.0.0")

    def _log_langfuse_v1(
        self,
        user_id,
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

        print(
            "Please upgrade langfuse to v2.0.0 or higher: https://github.com/langfuse/langfuse-python/releases/tag/v2.0.1"
        )

        trace = self.Langfuse.trace(
            CreateTrace(
                name=metadata.get("generation_name", "litellm-completion"),
                input=input,
                output=output,
                userId=user_id,
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
        user_id,
        metadata,
        output,
        start_time,
        end_time,
        kwargs,
        optional_params,
        input,
        response_obj,
        print_verbose,
    ):
        import langfuse

        tags = []
        supports_tags = Version(langfuse.version.__version__) >= Version("2.6.3")
        supports_costs = Version(langfuse.version.__version__) >= Version("2.7.3")

        print_verbose(f"Langfuse Layer Logging - logging to langfuse v2 ")

        generation_name = metadata.get("generation_name", None)
        if generation_name is None:
            # just log `litellm-{call_type}` as the generation name
            generation_name = f"litellm-{kwargs.get('call_type', 'completion')}"

        trace_params = {
            "name": generation_name,
            "input": input,
            "output": output,
            "user_id": metadata.get("trace_user_id", user_id),
            "id": metadata.get("trace_id", None),
            "session_id": metadata.get("session_id", None),
        }
        cost = kwargs["response_cost"]
        print_verbose(f"trace: {cost}")
        if supports_tags:
            for key, value in metadata.items():
                tags.append(f"{key}:{value}")
            if "cache_hit" in kwargs:
                tags.append(f"cache_hit:{kwargs['cache_hit']}")
            trace_params.update({"tags": tags})

        trace = self.Langfuse.trace(**trace_params)

        # get generation_id
        generation_id = None
        if response_obj.get("id", None) is not None:
            generation_id = litellm.utils.get_logging_id(start_time, response_obj)
        trace.generation(
            name=generation_name,
            id=metadata.get("generation_id", generation_id),
            startTime=start_time,
            endTime=end_time,
            model=kwargs["model"],
            modelParameters=optional_params,
            input=input,
            output=output,
            usage={
                "prompt_tokens": response_obj["usage"]["prompt_tokens"],
                "completion_tokens": response_obj["usage"]["completion_tokens"],
                "total_cost": cost if supports_costs else None,
            },
            metadata=metadata,
        )
