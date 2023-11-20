#### What this does ####
#    On success, logs events to Langfuse
import dotenv, os
import requests
import requests
from datetime import datetime

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback

class LangFuseLogger:
    # Class variables or attributes
    def __init__(self):
        try:
            from langfuse import Langfuse
        except Exception as e:
            raise Exception(f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\033[0m")
        # Instance variables
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")
        self.langfuse_debug = os.getenv("LANGFUSE_DEBUG")
        self.Langfuse =  Langfuse(
            public_key=self.public_key,
            secret_key=self.secret_key,
            host=self.langfuse_host,
            release=self.langfuse_release,
            debug=self.langfuse_debug
        )

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        from langfuse.model import InitialGeneration, Usage
        try:
            print_verbose(
                f"Langfuse Logging - Enters logging function for model {kwargs}"
            )
            metadata = kwargs.get("metadata", {})
            prompt = [kwargs.get('messages')]
            optional_params = kwargs.get("optional_params", {})

            # langfuse only accepts str, int, bool, float for logging
            for param, value in optional_params.items():
                if not isinstance(value, (str, int, bool, float)):
                    try:
                        optional_params[param] = str(value)
                    except:
                        # if casting value to str fails don't block logging
                        pass
 
            # end of processing langfuse ########################

            self.Langfuse.generation(InitialGeneration(
                name=metadata.get("generation_name", "litellm-completion"),
                startTime=start_time,
                endTime=end_time,
                model=kwargs['model'],
                modelParameters=optional_params,
                prompt=prompt,
                completion=response_obj['choices'][0]['message'],
                usage=Usage(
                    prompt_tokens=response_obj['usage']['prompt_tokens'],
                    completion_tokens=response_obj['usage']['completion_tokens']
                ),
                metadata=metadata
            ))
            self.Langfuse.flush()
            print_verbose(
                f"Langfuse Layer Logging - final response object: {response_obj}"
            )
        except:
            # traceback.print_exc()
            print_verbose(f"Langfuse Layer Error - {traceback.format_exc()}")
            pass
