#### What this does ####
#    On success, logs events to Langsmith
import dotenv, os
import requests
import requests
from datetime import datetime

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback

class LangsmithLogger:
    # Class variables or attributes
    def __init__(self):
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")


    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        # Method definition
        # inspired by Langsmith http api here: https://github.com/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/rest/rest.ipynb
        metadata = {}
        if "litellm_params" in kwargs:
            metadata = kwargs["litellm_params"].get("metadata", {})
        # set project name and run_name for langsmith logging 
        # users can pass project_name and run name to litellm.completion()
        # Example: litellm.completion(model, messages, metadata={"project_name": "my-litellm-project", "run_name": "my-langsmith-run"})
        # if not set litellm will use default project_name = litellm-completion, run_name = LLMRun
        project_name = metadata.get("project_name", "litellm-completion")
        run_name = metadata.get("run_name", "LLMRun")
        print_verbose(f"Langsmith Logging - project_name: {project_name}, run_name {run_name}")
        try:
            print_verbose(
                f"Langsmith Logging - Enters logging function for model {kwargs}"
            )
            import requests
            import datetime
            from datetime import timezone
            try:
                start_time = kwargs["start_time"].astimezone(timezone.utc).isoformat()
                end_time = kwargs["end_time"].astimezone(timezone.utc).isoformat()
            except:
                start_time = datetime.datetime.utcnow().isoformat()
                end_time = datetime.datetime.utcnow().isoformat()

            # filter out kwargs to not include any dicts, langsmith throws an erros when trying to log kwargs
            new_kwargs = {}
            for key in kwargs:
                value = kwargs[key]
                if key == "start_time" or key =="end_time":
                    pass
                elif type(value) != dict:
                    new_kwargs[key] = value

            requests.post(
                "https://api.smith.langchain.com/runs",
                json={
                    "name": run_name,
                    "run_type": "llm", # this should always be llm, since litellm always logs llm calls. Langsmith allow us to log "chain"
                    "inputs": {
                        **new_kwargs
                    },
                    "outputs": response_obj,
                    "session_name": project_name,
                    "start_time": start_time,
                    "end_time": end_time,
                },
                headers={
                    "x-api-key": self.langsmith_api_key
                }
            )
            print_verbose(
                f"Langsmith Layer Logging - final response object: {response_obj}"
            )
        except:
            # traceback.print_exc()
            print_verbose(f"Langsmith Layer Error - {traceback.format_exc()}")
            pass
