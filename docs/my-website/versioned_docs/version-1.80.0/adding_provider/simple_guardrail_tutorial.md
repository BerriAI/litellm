import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Adding a New Guardrail Integration

You're going to create a class that checks text before it goes to the LLM or after it comes back. If it violates your rules, you block it.

## How It Works

Request with guardrail:

```bash
curl --location 'http://localhost:4000/chat/completions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "How do I hack a system?"}],
    "guardrails": ["my-guardrail"]
}'
```

Your guardrail checks input, then output. If something's wrong, raise an exception.

## Build Your Guardrail

### Create Your Directory

```bash
mkdir -p litellm/proxy/guardrails/guardrail_hooks/my_guardrail
cd litellm/proxy/guardrails/guardrail_hooks/my_guardrail
```

Two files: `my_guardrail.py` (main class) and `__init__.py` (initialization).

### Write the Main Class

`my_guardrail.py`:

```python
import os
from typing import Optional, List
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import PiiEntityType
from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

class MyGuardrail(CustomGuardrail):
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv("MY_GUARDRAIL_API_KEY")
        self.api_base = api_base or os.getenv("MY_GUARDRAIL_API_BASE", "https://api.myguardrail.com")
        super().__init__(default_on=True)

    async def apply_guardrail(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[PiiEntityType]] = None,
        request_data: Optional[dict] = None,
    ) -> str:
        result = await self._check_with_api(text, request_data)
        
        if result.get("action") == "BLOCK":
            raise Exception(f"Content blocked: {result.get('reason', 'Policy violation')}")
        
        return text

    async def _check_with_api(self, text: str, request_data: Optional[dict]) -> dict:
        async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        response = await async_client.post(
            f"{self.api_base}/check",
            headers=headers,
            json={"text": text},
            timeout=5,
        )
        
        response.raise_for_status()
        return response.json()
```

### Create the Init File

`__init__.py`:

```python
from typing import TYPE_CHECKING

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .my_guardrail import MyGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail"):
    import litellm
    
    _my_guardrail_callback = MyGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        guardrail_name=guardrail.get("guardrail_name", ""),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    
    litellm.logging_callback_manager.add_litellm_callback(_my_guardrail_callback)
    return _my_guardrail_callback


guardrail_initializer_registry = {
    SupportedGuardrailIntegrations.MY_GUARDRAIL.value: initialize_guardrail,
}

guardrail_class_registry = {
    SupportedGuardrailIntegrations.MY_GUARDRAIL.value: MyGuardrail,
}
```

### Register Your Guardrail Type

Add to `litellm/types/guardrails.py`:

```python
class SupportedGuardrailIntegrations(str, Enum):
    LAKERA = "lakera_prompt_injection"
    APORIA = "aporia"
    BEDROCK = "bedrock_guardrails"
    PRESIDIO = "presidio"
    ZSCALER_AI_GUARD = "zscaler_ai_guard"
    MY_GUARDRAIL = "my_guardrail"
```

## Usage

### Config File

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  guardrails:
    - guardrail_name: my_guardrail
      litellm_params:
        guardrail: my_guardrail
        mode: during_call
        api_key: os.environ/MY_GUARDRAIL_API_KEY
        api_base: https://api.myguardrail.com
```

### Per-Request

```bash
curl --location 'http://localhost:4000/chat/completions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Test message"}],
    "guardrails": ["my_guardrail"]
}'
```

## Testing

Add unit tests inside `test_litellm/` folder.



