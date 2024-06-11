from typing import Optional, Union
from typing_extensions import override
import httpx
from openai import OpenAI
from ..llms.openai import OpenAIAssistantsAPI
from astra_assistants import patch
import astra_assistants



class AstraAssistantsAPI(OpenAIAssistantsAPI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def get_openai_client(
            self,
            api_key: Optional[str],
            api_base: Optional[str],
            timeout: Union[float, httpx.Timeout],
            max_retries: Optional[int],
            organization: Optional[str],
            client: Optional[OpenAI] = None,
    ) -> OpenAI:
        received_args = locals()
        if client is None:
            data = {}
            for k, v in received_args.items():
                if k == "self" or k == "client":
                    pass
                elif k == "api_base" and v is not None:
                    data["base_url"] = v
                elif v is not None:
                    data[k] = v
            openai_client = patch(OpenAI(**data))
        else:
            openai_client = client

        return openai_client
