# Instructor

Combine LiteLLM with [jxnl's instructor library](https://github.com/jxnl/instructor) for more robust structured outputs. Outputs are automatically validated into Pydantic types and validation errors are provided back to the model to increase the chance of a successful response in the retries.

## Usage (Sync)

```python
import instructor
from litellm import completion
from pydantic import BaseModel


client = instructor.from_litellm(completion)


class User(BaseModel):
    name: str
    age: int


def extract_user(text: str):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=User,
        messages=[
            {"role": "user", "content": text},
        ],
        max_retries=3,
    )

user = extract_user("Jason is 25 years old")

assert isinstance(user, User)
assert user.name == "Jason"
assert user.age == 25
print(f"{user=}")
```

## Usage (Async)

```python
import asyncio

import instructor
from litellm import acompletion
from pydantic import BaseModel


client = instructor.from_litellm(acompletion)


class User(BaseModel):
    name: str
    age: int


async def extract(text: str) -> User:
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=User,
        messages=[
            {"role": "user", "content": text},
        ],
        max_retries=3,
    )

user = asyncio.run(extract("Alice is 30 years old"))

assert isinstance(user, User)
assert user.name == "Alice"
assert user.age == 30
print(f"{user=}")
```
