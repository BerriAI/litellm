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
        except:
            raise Exception("\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error\033[0m")
        # Instance variables
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.Langfuse =  Langfuse(
                        public_key=self.public_key,
                        secret_key=self.secret_key,
                        host=self.langfuse_host,
                    )

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        from langfuse.model import InitialGeneration, Usage
        try:
            print_verbose(
                f"Langfuse Logging - Enters logging function for model {kwargs}"
            )
            # print(response_obj)
            # print(response_obj['choices'][0]['message']['content'])
            # print(response_obj['usage']['prompt_tokens'])
            # print(response_obj['usage']['completion_tokens'])

            self.Langfuse.generation(InitialGeneration(
                name="litellm-completion",
                startTime=start_time,
                endTime=end_time,
                model=kwargs['model'],
                # modelParameters= kwargs,
                prompt=[kwargs['messages']],
                completion=response_obj['choices'][0]['message']['content'],
                usage=Usage(
                    prompt_tokens=response_obj['usage']['prompt_tokens'],
                    completion_tokens=response_obj['usage']['completion_tokens']
                ),
            ))
            self.Langfuse.flush()
            print_verbose(
                f"Langfuse Layer Logging - final response object: {response_obj}"
            )
        except:
            # traceback.print_exc()
            print_verbose(f"Langfuse Layer Error - {traceback.format_exc()}")
            pass
