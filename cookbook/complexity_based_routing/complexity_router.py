"""
Complexity-based routing strategy for LiteLLM.

Registers a custom routing strategy (LiteLLM's documented CustomRoutingStrategyBase
extension point) that intercepts requests to the "task/auto" model group, classifies
the request's complexity, and returns the appropriate underlying local/cloud
deployment. Every other model group name passes through untouched to the router's
original deployment-selection logic.

Wire it up in your proxy config:

    environment:
      LITELLM_WORKER_STARTUP_HOOKS: complexity_router:register
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./complexity_router.py:/app/complexity_router.py

and add a placeholder model_list entry for "task/auto" (see README.md) so the
router recognizes the group name -- the strategy below always overrides which
actual deployment gets used at request time.

Adjust TIER_MODEL_NAMES, CLASSIFIER_URL/MODEL, and the heuristic regexes to match
your own model_list entries and provider layout.
"""
import hashlib
import re

import httpx
from litellm.router import CustomRoutingStrategyBase

AUTO_MODEL_GROUP = "task/auto"

# tier -> the model_name you've defined for it in config.yaml's model_list
TIER_MODEL_NAMES = {
    "simple": "small-model",       # fast, cheap, local
    "standard": "primary-model",   # your everyday local model
    "deep": "primary-model-deep",  # same model, reasoning/thinking mode on
    "cloud": "cloud-model",        # last resort
}
DEFAULT_TIER = "standard"

SIMPLE_HINTS = re.compile(
    r"\b(look ?up|fetch|what is|status of|define|list|format|convert)\b", re.I
)
COMPLEX_HINTS = re.compile(
    r"\b(compare|analy[sz]e|strategy|decide|design|architecture|trade-?off|"
    r"evaluate|plan|prioriti[sz]e)\b",
    re.I,
)

# Classify locally via a small/cheap model, called *directly* (not through this
# same proxy) to avoid any risk of recursive routing.
CLASSIFIER_URL = "http://localhost:11435/api/chat"
CLASSIFIER_MODEL = "small-model"
CLASSIFIER_TIMEOUT = 15.0

# Coarse in-memory per-task cache, keyed on the opening user message. Classify
# once at task-start and stick with it rather than re-scoring every turn --
# re-scoring risks flip-flopping models mid-conversation. Not durable across
# proxy restarts; that's an accepted limitation of this simple version.
_cache: dict[str, str] = {}
_CACHE_MAX = 500


def _first_user_message(messages):
    for m in messages or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _cache_key(text):
    return hashlib.sha256(text[:500].encode("utf-8")).hexdigest()


def _heuristic_tier(text):
    """Cheap, zero-latency gate. Returns a tier, or None if ambiguous."""
    if COMPLEX_HINTS.search(text):
        return "deep"
    if len(text) < 200 and SIMPLE_HINTS.search(text):
        return "simple"
    if len(text) < 40:
        return "simple"
    return None


async def _llm_classify(text):
    """Ambiguous middle case: one cheap local call to score it."""
    prompt = (
        "Rate the complexity of fulfilling this request on a scale of 1-3.\n"
        "1 = simple lookup, fact, or single-step task\n"
        "2 = moderate reasoning, synthesis, or multi-step task\n"
        "3 = complex, strategic, or judgment-heavy task requiring deep reasoning\n"
        f"Request: {text[:2000]}\n"
        "Respond with only the number."
    )
    try:
        async with httpx.AsyncClient(timeout=CLASSIFIER_TIMEOUT) as client:
            resp = await client.post(
                CLASSIFIER_URL,
                json={
                    "model": CLASSIFIER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                },
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            match = re.search(r"[123]", content)
            score = int(match.group()) if match else 2
            return {1: "simple", 2: "standard", 3: "deep"}[score]
    except Exception:
        return DEFAULT_TIER


async def _classify(messages):
    text = _first_user_message(messages)
    if not text:
        return DEFAULT_TIER

    key = _cache_key(text)
    if key in _cache:
        return _cache[key]

    tier = _heuristic_tier(text)
    if tier is None:
        tier = await _llm_classify(text)

    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[key] = tier
    return tier


def _find_deployment(router, model_name):
    for deployment in router.model_list:
        if deployment.get("model_name") == model_name:
            return deployment
    return None


class ComplexityRoutingStrategy(CustomRoutingStrategyBase):
    def __init__(self, router, default_sync, default_async):
        self.router = router
        self._default_sync = default_sync
        self._default_async = default_async

    def get_available_deployment(
        self, model, messages=None, input=None, specific_deployment=False,
        request_kwargs=None,
    ):
        if model != AUTO_MODEL_GROUP:
            return self._default_sync(
                model=model, messages=messages, input=input,
                specific_deployment=specific_deployment, request_kwargs=request_kwargs,
            )
        # Sync path: heuristic only (no blocking LLM call here).
        text = _first_user_message(messages)
        tier = _heuristic_tier(text) or DEFAULT_TIER
        deployment = _find_deployment(self.router, TIER_MODEL_NAMES[tier])
        return deployment or _find_deployment(self.router, TIER_MODEL_NAMES[DEFAULT_TIER])

    async def async_get_available_deployment(
        self, model, messages=None, input=None, specific_deployment=False,
        request_kwargs=None,
    ):
        if model != AUTO_MODEL_GROUP:
            return await self._default_async(
                model=model, messages=messages, input=input,
                specific_deployment=specific_deployment, request_kwargs=request_kwargs,
            )
        tier = await _classify(messages or [])
        deployment = _find_deployment(self.router, TIER_MODEL_NAMES[tier])
        if deployment is None:
            deployment = _find_deployment(self.router, TIER_MODEL_NAMES[DEFAULT_TIER])
        return deployment


async def _wait_and_register():
    import asyncio
    import litellm.proxy.proxy_server as proxy_server

    llm_router = None
    for attempt in range(180):
        llm_router = proxy_server.llm_router
        if llm_router is not None:
            break
        await asyncio.sleep(1)

    if llm_router is None:
        print(f"[complexity-router] llm_router never became ready, giving up after 180s")
        return

    strategy = ComplexityRoutingStrategy(
        router=llm_router,
        default_sync=llm_router.get_available_deployment,
        default_async=llm_router.async_get_available_deployment,
    )
    llm_router.set_custom_routing_strategy(strategy)
    print(f"[complexity-router] registered after {attempt}s")


def register():
    """
    Startup hook entrypoint. Must return immediately -- LITELLM_WORKER_STARTUP_HOOKS
    fires before llm_router is constructed, and awaiting the wait loop directly here
    would block the proxy's own startup indefinitely. Schedule it as a background
    task instead.
    """
    import asyncio

    asyncio.create_task(_wait_and_register())
    print("[complexity-router] scheduled background registration task")
