"""Typed per-test metadata for the e2e suite: what a test drives (`Subject`) and what it did (`steps`). See AGENTS.md"""

from __future__ import annotations

import inspect
import os
import re
import string
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from functools import reduce, wraps
from itertools import chain
from types import MappingProxyType, TracebackType
from typing import Final, ParamSpec, TypeVar, cast

import pytest
from pydantic import BaseModel


class Domain(str, Enum):
    """The OSS issue-label taxonomy, so an issue and a test join on one string"""

    LLM_TRANSLATION = "llm-translation"
    SPEND_BUDGETS = "spend-budgets"
    UI = "ui"
    MCP = "mcp"
    OBSERVABILITY = "observability"
    ROUTING = "routing"
    DEPLOY_OPS = "deploy-ops"
    COST_MAP = "cost-map"
    PROXY_AUTH = "proxy-auth"
    GUARDRAILS = "guardrails"
    MANAGEMENT = "management"
    SDK = "sdk"
    PASSTHROUGH = "passthrough"
    DB = "db"
    CACHING = "caching"
    DOCS = "docs"
    AGENTS_API = "agents-api"
    UNKNOWN = "unknown"


class Route(str, Enum):
    """The endpoint the test is checking; unset when the call only triggers the behavior under test"""

    CHAT_COMPLETIONS = "chat_completions"
    MESSAGES = "messages"
    RESPONSES = "responses"
    EMBEDDINGS = "embeddings"
    COMPLETIONS = "completions"
    FILES = "files"
    BATCHES = "batches"
    PASSTHROUGH = "passthrough"
    MCP = "mcp"
    GUARDRAILS = "guardrails"
    KEY_MANAGEMENT = "key_management"
    TEAM_MANAGEMENT = "team_management"
    SPEND_REPORTING = "spend_reporting"
    MODEL_MANAGEMENT = "model_management"
    IMAGES = "images"
    AUDIO = "audio"
    MODERATIONS = "moderations"
    RERANK = "rerank"
    OCR = "ocr"
    VECTOR_STORES = "vector_stores"
    REALTIME = "realtime"
    A2A = "a2a"
    USER_MANAGEMENT = "user_management"
    BUDGET_MANAGEMENT = "budget_management"
    ORGANIZATION_MANAGEMENT = "organization_management"
    CUSTOMER_MANAGEMENT = "customer_management"
    HEALTH = "health"
    METRICS = "metrics"
    PROXY_CONFIG = "proxy_config"
    ADMIN_UI = "admin_ui"
    CONTAINERS = "containers"
    COUNT_TOKENS = "count_tokens"
    GOOGLE_GENAI = "google_genai"
    TAG_MANAGEMENT = "tag_management"


class Provider(str, Enum):
    """Mirrors litellm's `LlmProviders` without importing litellm; `TestProviderMirrorsLitellm` catches drift"""

    OPENAI = "openai"
    OPENAI_LIKE = "openai_like"
    CUSTOM_OPENAI = "custom_openai"
    AZURE = "azure"
    AZURE_AI = "azure_ai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    VERTEX_AI = "vertex_ai"
    BEDROCK = "bedrock"
    SAGEMAKER = "sagemaker"
    XAI = "xai"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    COHERE = "cohere"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"
    TOGETHER_AI = "together_ai"
    FIREWORKS_AI = "fireworks_ai"
    CEREBRAS = "cerebras"
    SAMBANOVA = "sambanova"
    NVIDIA_NIM = "nvidia_nim"
    DATABRICKS = "databricks"
    WATSONX = "watsonx"
    OLLAMA = "ollama"
    VLLM = "vllm"
    HOSTED_VLLM = "hosted_vllm"
    VOYAGE = "voyage"
    JINA_AI = "jina_ai"
    DEEPGRAM = "deepgram"
    ELEVENLABS = "elevenlabs"
    ASSEMBLYAI = "assemblyai"
    LITELLM_PROXY = "litellm_proxy"
    BEDROCK_MANTLE = "bedrock_mantle"
    AWS_POLLY = "aws_polly"
    SAIL = "sail"


class Capability(str, Enum):
    """A model feature, 1:1 with a `supports_*` key in model_prices_and_context_window.json"""

    FUNCTION_CALLING = "function_calling"
    PARALLEL_FUNCTION_CALLING = "parallel_function_calling"
    TOOL_CHOICE = "tool_choice"
    TOOL_SEARCH = "tool_search"
    VISION = "vision"
    PDF_INPUT = "pdf_input"
    AUDIO_INPUT = "audio_input"
    REASONING = "reasoning"
    WEB_SEARCH = "web_search"
    PROMPT_CACHING = "prompt_caching"
    RESPONSE_SCHEMA = "response_schema"
    MID_CONVERSATION_SYSTEM = "mid_conversation_system"
    AUDIO_OUTPUT = "audio_output"


class Mode(str, Enum):
    """How the route was driven"""

    NONSTREAM = "nonstream"
    STREAM = "stream"
    BATCH = "batch"
    WEBSOCKET = "websocket"


_M = TypeVar("_M")


def _scalar(value: object) -> str:
    """`str()` on a (str, Enum) gives `Route.RESPONSES`, and StrEnum needs 3.11"""
    if isinstance(value, Enum):
        return str(value.value)  # pyright: ignore[reportAny]  # Enum.value is Any for every enum
    return str(value)


def _members(value: object) -> tuple[object, ...] | None:
    return cast("tuple[object, ...]", value) if isinstance(value, tuple) else None


def _canonical(name: str, value: object, member_type: type[_M]) -> tuple[_M, ...]:
    """Validated, deduped and sorted; a bare str like `("gpt-5.5")` raises at import"""
    members = _members(value)
    if members is None:
        raise TypeError(
            f"Subject.{name} must be a tuple, got {type(value).__name__}: {value!r}."
            f" A one-member tuple needs its trailing comma: {name}=(x,), not {name}=(x)"
        )
    typed = tuple(member for member in members if isinstance(member, member_type))
    if len(typed) != len(members):
        raise TypeError(f"Subject.{name} takes {member_type.__name__} members, got {value!r}")
    return tuple(sorted(frozenset(member for member in typed if _scalar(member)), key=_scalar))


@dataclass(frozen=True, slots=True)
class Subject:
    """What a test is about. Not named `Test*` so pytest does not try to collect it"""

    domain: Domain | None = None
    route: Route | None = None
    providers: tuple[Provider, ...] = ()
    models: tuple[str, ...] = ()
    capabilities: tuple[Capability, ...] = ()
    mode: Mode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", _canonical("providers", self.providers, Provider))
        object.__setattr__(self, "models", _canonical("models", self.models, str))
        object.__setattr__(self, "capabilities", _canonical("capabilities", self.capabilities, Capability))


def meta(subject: Subject) -> pytest.MarkDecorator:
    """Attach a `Subject` to a test: `@meta(Subject(route=Route.RESPONSES, ...))`"""
    return pytest.mark.meta(subject)


_P = ParamSpec("_P")
_R = TypeVar("_R")
_Y = TypeVar("_Y")

MAX_STEPS: Final = 50
MAX_STEP_CHARS: Final = 200

SECRET_ENV_NAME: Final = re.compile(r"(^|_)(KEY|SECRET|TOKEN|PASSWORD|CREDENTIALS?)(_|$)", re.IGNORECASE)
MIN_SECRET_CHARS: Final = 8
MASK: Final = "***"


def environment_secrets(environ: Mapping[str, str] = os.environ) -> frozenset[str]:
    """The credentials a live run holds: every secret-named environment variable's
    value, long enough that masking it can't blank out ordinary words."""
    return frozenset(
        value for name, value in environ.items() if SECRET_ENV_NAME.search(name) and len(value) >= MIN_SECRET_CHARS
    )


def _masked(label: str, secrets: Iterable[str]) -> str:
    longest_first: Final = sorted(secrets, key=len, reverse=True)
    return reduce(lambda text, secret: text.replace(secret, MASK), longest_first, label)


STEP_FRAMES: Final = 1
"""Frames a `@step` wrapper puts between a helper and its caller. A decorated
helper that warns about its caller adds this to `stacklevel`
(`stacklevel=2 + STEP_FRAMES`), or the warning is reported at the wrapper."""


class StepRecorder:
    """The ordered step log for the running test.

    A plain lock-guarded list rather than a ContextVar: ContextVars do not
    propagate into worker threads, and several e2e helpers call out from
    threads. Under xdist each worker is its own process, so there is no
    cross-test bleed beyond what the per-test reset already handles.
    """

    def __init__(self, secrets: Callable[[], Iterable[str]] = environment_secrets) -> None:
        self._secrets = secrets
        self._lock = threading.Lock()
        self._steps: deque[str] = deque(maxlen=MAX_STEPS)
        self._dropped = 0

    def reset(self) -> None:
        """Called first thing in every test's setup phase, so each test starts
        empty."""
        with self._lock:
            self._steps.clear()
            self._dropped = 0

    def record(self, label: str) -> None:
        """Append `label`, unless it repeats the previous step.

        A retrying helper (poll_cost_row) or a load test calling a decorated
        helper in a loop would otherwise emit thousands of <property> entries per
        testcase: a consecutive repeat collapses, so a poll loop is one step in
        the story rather than fifty, and past MAX_STEPS the oldest step makes way.
        It is the oldest that goes because the last step is the one that has to
        survive: it is where a failing test died.

        Any credential the run holds is masked before the label is kept, however it
        got into the label, since the steps are published with the results.
        """
        cleaned = " ".join(_masked(label, self._secrets()).split())[:MAX_STEP_CHARS]
        if not cleaned:
            return
        with self._lock:
            if self._steps and self._steps[-1] == cleaned:
                return
            if len(self._steps) == MAX_STEPS:
                self._dropped += 1
            self._steps.append(cleaned)

    def taken(self) -> tuple[str, ...]:
        """The story so far, led by a line counting the steps a full log dropped,
        so a story that starts mid-test says so rather than reading as complete."""
        with self._lock:
            dropped: Final = (f"({self._dropped} earlier steps not recorded)",) if self._dropped else ()
            return dropped + tuple(self._steps)


STEPS: Final = StepRecorder()


def _joined(phrases: tuple[str, ...]) -> str:
    if len(phrases) <= 1:
        return "".join(phrases)
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def _model_phrase(model: BaseModel) -> str:
    """The fields the caller set, as "models: a, b and rpm limit: 3". A
    `Field(repr=False)` field, pydantic's flag for a secret, is never shown."""
    values: Final = (
        (name, cast("object", getattr(model, name)))
        for name, field in type(model).model_fields.items()
        if name in model.model_fields_set and field.repr
    )
    phrases: Final = tuple(f"{name.replace('_', ' ')}: {_phrase(value)}" for name, value in values if _given(value))
    return _joined(phrases) or "default settings"


def _given(value: object) -> bool:
    return value is not None and value != [] and value != ()


def _phrase(value: object) -> str:
    if isinstance(value, BaseModel):
        return _model_phrase(value)
    if isinstance(value, Enum):
        return _phrase(cast("object", value.value))
    if isinstance(value, Mapping):
        entries: Final = cast("Mapping[object, object]", value)
        return _joined(tuple(f"{str(key).replace('_', ' ')}: {_phrase(item)}" for key, item in entries.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(map(_phrase, cast("Iterable[object]", value)))
    return str(value)


_PLACEHOLDER: Final = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")


def _placeholders(label: str) -> frozenset[str]:
    return frozenset(field for _, field, _, _ in string.Formatter().parse(label) if field is not None)


def _resolved(field: str, arguments: Mapping[str, object]) -> object:
    """`body.litellm_params.model` is the `body` argument's `litellm_params.model`."""
    root, *attributes = field.split(".")
    return reduce(lambda value, attribute: cast("object", getattr(value, attribute)), attributes, arguments[root])


def _filled(label: str, bound: inspect.BoundArguments) -> str:
    bound.apply_defaults()
    arguments: Final = cast("Mapping[str, object]", bound.arguments)
    return "".join(
        literal + ("" if field is None else _phrase(_resolved(field, arguments)))
        for literal, field, _, _ in string.Formatter().parse(label)
    )


class _Nesting(threading.local):
    """Whether this thread is already inside a `@step` helper.

    Per thread, like the helpers themselves: a worker thread a step fans out to
    starts outside any step, so its own decorated calls still record."""

    def __init__(self) -> None:
        self.inside: bool = False


_NESTING: Final = _Nesting()


@contextmanager
def _inside_step() -> Generator[None]:
    """Hold the nesting guard for the duration, restoring whatever it was."""
    outer: Final = _NESTING.inside
    _NESTING.inside = True
    try:
        yield
    finally:
        _NESTING.inside = outer


class _StepContext(AbstractContextManager[_Y]):
    """A `@contextmanager` helper's context, entered and exited inside its step.

    Calling a `@contextmanager` function runs none of its body: the setup runs at
    `__enter__` and the cleanup at `__exit__`, both after the call has returned
    and so both outside the guard the call held. Here each runs inside it, so the
    helpers they call stay out of the story, while the `with` body in between --
    the test's own code -- still records. Without this, a test that died inside
    the `with` would have the cleanup's steps appended behind the one it died on.
    """

    def __init__(self, inner: AbstractContextManager[_Y]) -> None:
        self._inner: Final = inner

    def __enter__(self) -> _Y:
        with _inside_step():
            return self._inner.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        with _inside_step():
            return self._inner.__exit__(exc_type, exc, traceback)


def step(label: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Record `label` on the running test whenever this helper is called.

    Goes on HARNESS helpers (client methods, fixtures), never on tests. The
    label is recorded BEFORE the wrapped call, so a helper that raises still
    leaves its own label as the last element -- which is the whole point: the
    last step is where the test died.

    Only the outermost step records. Harness layers call each other --
    `ProxyClient.create_model` goes through `register_model`, a domain
    client wraps the shared `ProxyClient` -- so every layer can carry its own
    label without one action showing up in the story once per layer. The story
    reads at the level the test called in at, and the label of the helper the
    test called is still the last one when anything beneath it raises.

    On a `@contextmanager` helper `@step` goes ABOVE `@contextmanager`, and the
    setup and cleanup around its `yield` count as part of the step (see
    `_StepContext`). A bare generator function is refused where the decorator
    runs: its body only runs as the caller iterates, interleaved with the
    caller's own steps, so no single point in the story is where it happened.
    """

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        signature: Final = inspect.signature(fn)
        placeholders: Final = _placeholders(label)
        malformed: Final = sorted(field for field in placeholders if not _PLACEHOLDER.fullmatch(field))
        if malformed:
            raise TypeError(f"@step({label!r}) has {malformed}: a placeholder is a parameter or its dotted attribute")
        unknown: Final = {field.split(".")[0] for field in placeholders} - signature.parameters.keys()
        if unknown:
            raise TypeError(f"@step({label!r}) names {sorted(unknown)}, which {fn.__qualname__} doesn't take")
        static_label: Final = None if placeholders else label.format()
        if inspect.isgeneratorfunction(fn):
            raise TypeError(
                f"@step({label!r}) cannot wrap the generator function {fn!r}: put it on a helper that"
                " returns, or above @contextmanager on one that yields a context"
            )
        underlying: Final[object] = inspect.unwrap(fn)  # pyright: ignore[reportAny]  # inspect.unwrap is typed as returning Any
        opens_a_context: Final = inspect.isgeneratorfunction(underlying)

        @wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if not _NESTING.inside:
                STEPS.record(static_label or _filled(label, signature.bind(*args, **kwargs)))
            with _inside_step():
                result = fn(*args, **kwargs)
            if opens_a_context and isinstance(result, AbstractContextManager):
                context: Final = cast("AbstractContextManager[object]", result)
                return cast("_R", _StepContext(context))
            return result

        return wrapper

    return decorate


_REPEATED: Final = MappingProxyType({"providers": "provider", "models": "model", "capabilities": "capability"})


def _declared_subject(args: tuple[object, ...]) -> Subject | None:
    first = args[0] if args else None
    return first if isinstance(first, Subject) else None


def subject_properties(item: pytest.Item) -> tuple[tuple[str, str], ...]:
    """The declared fields as <property> pairs, plural fields repeated under their singular name"""
    marker: Final = item.get_closest_marker("meta")
    if marker is None:
        return ()
    subject: Final = _declared_subject(marker.args)
    if subject is None:
        return ()
    declared: Final[dict[str, object]] = asdict(subject)
    return tuple(chain.from_iterable(_field_properties(name, value) for name, value in declared.items()))


def _field_properties(name: str, value: object) -> tuple[tuple[str, str], ...]:
    repeated: Final = _REPEATED.get(name)
    if repeated is not None:
        return tuple((repeated, _scalar(member)) for member in _members(value) or ())
    if value is None or value == "":
        return ()
    return ((name, _scalar(value)),)


def step_properties() -> tuple[tuple[str, str], ...]:
    """The step log as repeated `step` properties. Appended after the setup and
    call phases, never at collection."""
    return tuple(("step", label) for label in STEPS.taken())
