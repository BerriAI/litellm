# Route your first requests

In this tutorial you build a Router with three mock deployments and watch it
balance load, retry, cool a deployment down and fall back to another model
group. Every deployment uses `mock_response`, so nothing leaves your machine and
you need no API keys.

You need a checkout of this repo with its environment installed (`uv sync`). Run
each step from the repo root with `uv run python`, or paste the steps one after
another into a single file.

![The five tutorial steps in order: build a Router, send requests, look inside,
break a deployment, break the group](../img/route-your-first-requests-1.svg)

## 1. Build a Router

Create `tutorial.py` with a model list. Two deployments, `a` and `b`, serve the
model group `chat`. A third, `c`, serves the model group `backup`. The router
retries once inside a group, then falls back from `chat` to `backup`.

```python
import asyncio
import random
from collections import Counter

from litellm import Router
from litellm.router_api import ChangeSet
from litellm.types.router import Deployment

random.seed(0)


def mock(id: str, group: str, response: str) -> dict:
    return {
        "model_name": group,
        "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fake", "mock_response": response},
        "model_info": {"id": id},
    }


router = Router(
    model_list=[
        mock("a", "chat", "hi from a"),
        mock("b", "chat", "hi from b"),
        mock("c", "backup", "hi from backup"),
    ],
    fallbacks=[{"chat": ["backup"]}],
    num_retries=1,
)


async def ask() -> tuple[str, str]:
    response = await router.acompletion(model="chat", messages=[{"role": "user", "content": "hello"}])
    return response._hidden_params["model_id"], response.choices[0].message.content
```

`random.seed(0)` makes the router's random picks repeat, so your output matches
this page. `model_name` is the name callers send. `litellm_params` is what the
router passes to `litellm.acompletion` once it picks the deployment.
`model_info.id` names the deployment. When you leave it out, the router hashes
the group and params into an id.

`Router(...)` freezes these arguments into a `RouterConfig`, and settings stay
fixed after that.

## 2. Send requests

Add a `main` coroutine and send twenty requests.

```python
async def main() -> None:
    print("picked:", Counter([(await ask())[0] for _ in range(20)]))


asyncio.run(main())
```

Run it. You see both deployments picked, in a split close to even.

```
picked: Counter({'b': 12, 'a': 8})
```

No strategy was configured, so the router used `simple-shuffle`, the default.
Neither deployment sets `weight`, `rpm` or `tpm`, so each pick is a uniform
random choice among the healthy deployments of the group. Each attempt still
went through `litellm.acompletion`, so `mock_response` worked as it does without
a router.

## 3. Look inside

The router answers questions about itself through its entry points. Add these
lines to `main`.

```python
    registry = router.registry()
    print("groups:", sorted(registry.names(kind="group_set")))
    print("chat deployments:", [d["model_info"]["id"] for d in registry.deployments("chat")])
    print("strategy:", router.settings().routing_strategy)
    picked = await router.select("chat")
    print("select picked:", picked["model_info"]["id"])
```

```
groups: ['backup', 'chat']
chat deployments: ['a', 'b']
strategy: simple-shuffle
select picked: b
```

`registry()` holds the configured deployments, `settings()` is a frozen view of
the configuration, and `select()` runs the same pick a real request runs,
without calling the provider. [Entry points](../reference/entry-points.md) lists
everything they offer.

## 4. Break a deployment

Replace deployment `a` with one that always answers 429. Router settings are
static after construction, so changes go through `apply()` with a `ChangeSet`.
The router replaces the deployment and bumps the registry version. The exact
number depends on how many deployments were added before. What matters is that
it goes up.

```python
    broken_a = Deployment(**mock("a", "chat", "litellm.RateLimitError"))
    print("version after apply:", router.apply(ChangeSet(upsert=[broken_a])).version)
    print("picked:", Counter([(await ask())[0] for _ in range(10)]))
    print("cooldowns:", await router.health().cooldowns.async_get_active_cooldowns(
        model_ids=["a", "b", "c"], parent_otel_span=None
    ))
```

```
version after apply: 7
picked: Counter({'b': 10})
cooldowns: [('a', {'exception_received': 'litellm.RateLimitError: ...', 'status_code': '429', 'timestamp': ..., 'cooldown_time': 5})]
```

Every request succeeded. The first one that landed on `a` failed with a 429, and
two things happened.

![Sequence in which the first attempt on deployment a gets a 429, the router
sets a cooldown on a, retries on b and gets a 200, and later requests only see
b](../img/route-your-first-requests-2.svg)

First, the router retried in the same group and skipped `a`, so the caller got
`b`'s answer. Second, it recorded the failure and put `a` into cooldown, because
a 429 in a group with more than one deployment cools that deployment down at
once. For the next five seconds, the default cooldown time, the cooldown filter
removes `a` before the strategy ever sees it. [Retries, fallbacks and
cooldowns](../explanation/reliability.md) explains the rules.

## 5. Break the group

Break `b` too and send one more request.

```python
    broken_b = Deployment(**mock("b", "chat", "litellm.RateLimitError"))
    router.apply(ChangeSet(upsert=[broken_b]))
    print("answer:", await ask())
```

```
answer: ('c', 'hi from backup')
```

With both `chat` deployments failing, the retry has nowhere left to go in the
group. Once retries are spent, the router's fallback walk looks up `chat` in
`fallbacks`, finds `backup`, and runs the whole retry loop again against that
group.

![A request for chat retries a and b, both return 429, then falls back to
backup, where c returns 200 with 'hi from
backup'](../img/route-your-first-requests-3.svg)

## What you saw

A Router owns a list of deployments grouped by `model_name` into model groups.
Each request is resolved to a model group, the group's deployments are filtered,
and a strategy picks one. Failures feed back into shared state (the cooldown you
saw) and into the retry and fallback loop around the attempt. Code outside the
router reads and changes all of this through a handful of entry points.

Next, read [Router concepts](../explanation/concepts.md) for the ideas behind
each step, [The life of a request](../explanation/request-lifecycle.md) for the
full path, or [Configuration and defaults](../reference/configuration.md) for
the knobs you skipped here.
