# Instructor - Function Calling

Use LiteLLM Router with [jxnl's instructor library](https://github.com/jxnl/instructor) for function calling in prod. 

## Usage

```python
import litellm
from litellm import Router
import instructor
from pydantic import BaseModel

litellm.set_verbose = True # 👈 print DEBUG LOGS

client = instructor.patch(
    Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  openai model name
                "litellm_params": {  # params for litellm completion/embedding call - e.g.: https://github.com/BerriAI/litellm/blob/62a591f90c99120e1a51a8445f5c3752586868ea/litellm/router.py#L111
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ]
    )
)


class UserDetail(BaseModel):
    name: str
    age: int


user = client.chat.completions.create(
    model="gpt-3.5-turbo",
    response_model=UserDetail,
    messages=[
        {"role": "user", "content": "Extract Jason is 25 years old"},
    ],
)

assert isinstance(user, UserDetail)
assert user.name == "Jason"
assert user.age == 25

print(f"user: {user}")
```

## Async Calls

```python
import litellm
from litellm import Router
import instructor, asyncio
from pydantic import BaseModel

aclient = instructor.apatch(
    Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-2",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ],
        default_litellm_params={"acompletion": True}, # 👈 IMPORTANT - tells litellm to route to async completion function.
    )
)


class UserExtract(BaseModel):
    name: str
    age: int


async def main():
    model = await aclient.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserExtract,
        messages=[
            {"role": "user", "content": "Extract jason is 25 years old"},
        ],
    )
    print(f"model: {model}")


asyncio.run(main())
```