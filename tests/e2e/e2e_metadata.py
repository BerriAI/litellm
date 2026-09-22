"""Typed per-test metadata for the e2e suite: what a test drives, and what it did.

Two halves, deliberately separated.

The DECLARED half is `Subject`: one frozen dataclass passed as the single
positional argument of `@meta(...)`. Every field is a closed enum (or free
strings for `models`), so a typo is a basedpyright error at the call site rather
than a silently dropped property. `dataclasses.asdict()` turns the whole thing
into <property> pairs with no per-field plumbing -- adding a scalar field later
needs zero serializer changes.

The RECORDED half is `steps`, and it is NOT a field of `Subject`. Steps are
appended at runtime by `@step`-decorated harness helpers, in call order, so the
list IS the test's user story and its last element is where a failing test died.
Putting it on the declarable dataclass would invite hand-writing it, which is
exactly what it replaces.

Stdlib-only on purpose, and that includes the call sites. tests/e2e is a
black-box HTTP suite that imports litellm in zero files and is shipped to the
runner image as tests/e2e alone; a `from litellm...` at the top of a test module
would make the litellm package a COLLECTION-time dependency of the whole suite,
so an image without it would fail collection rather than run tests. `Provider`
below therefore mirrors litellm's `LlmProviders` values here instead of
importing them, and `TestProviderMirrorsLitellm` in test_e2e_metadata.py
fails wherever litellm IS importable if the two ever drift.
"""

from __future__ import annotations

import inspect
import threading
from collections import deque
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from types import TracebackType
from typing import Final, ParamSpec, TypeVar, cast

import pytest


class Domain(str, Enum):
    """The OSS issue-label taxonomy, verbatim.

    Shared with GitHub issue labels so an issue and a test join on one string.
    Exactly one per test; `UNKNOWN` is the honest answer, not an omission.
    """

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
    """The customer-facing HTTP surface the test DRIVES.

    Orthogonal to the suite directory: a reliability test in router/ and a
    logging test in logging/ both drive `CHAT_COMPLETIONS`, which is why this is
    per-test and not per-module.

    "route" here means endpoint, matching litellm's own `LiteLLMRoutes`
    (litellm/proxy/_types.py). The coverage registry's `LlmCell.route` uses the
    same word for PROVIDER; that is a different namespace and is left alone.

    Deliberately collapsed against the registry's `LlmEndpoint`:
    images_generations + images_edits -> IMAGES, audio_speech +
    audio_transcriptions -> AUDIO, bedrock_native + google_native ->
    PASSTHROUGH. Those splits are wire detail, not a customer-facing surface,
    and `models` + `capabilities` already carry them.

    The last four are ops surfaces: logging/, load/, other/ and ui/ have no LLM
    route of their own and would otherwise have to lie.
    """

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
    HEALTH = "health"
    METRICS = "metrics"
    PROXY_CONFIG = "proxy_config"
    ADMIN_UI = "admin_ui"


class Provider(str, Enum):
    """The upstream LLM provider the test drives, spelled exactly as litellm's
    own `LlmProviders` (litellm/types/utils.py) spells it.

    A deliberate mirror, not an import. Importing `LlmProviders` at the top of a
    test module pulls `litellm/__init__` (measured: 1.44s, 2474 modules) and,
    worse, makes the litellm package a hard dependency of COLLECTING tests/e2e --
    which is shipped to the e2e runner image on its own, so a missing package
    would not slow the suite down, it would error every test out at collection.
    The suite has zero runtime litellm imports and this keeps it that way.

    The mirror cannot drift silently: `TestProviderMirrorsLitellm` in
    test_e2e_metadata.py asserts every value here is a real `LlmProviders`
    value, and runs wherever litellm is importable (dev checkouts, the repo's own
    CI) while skipping where it is not. Adding a provider is one line here.
    """

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


class Capability(str, Enum):
    """A MODEL feature the test depends on, anchored 1:1 to a `supports_*` key
    in model_prices_and_context_window.json.

    Not to be confused with the coverage registry's `LlmCell.capability`, which
    means the endpoint feature under test (`basic`, `multi_turn`, ...) and half
    of whose values have no `supports_*` key at all.

    Plural by necessity, never a single enum: `thinking_with_tool_use` and
    `tool_search_history` only exist in the registry because a single-enum field
    had nowhere to put a conjunction. Here they are
    (REASONING, FUNCTION_CALLING) and (TOOL_SEARCH,).

    `audio_output` is deliberately absent: litellm/utils.py reads
    `supports_audio_input` for it, so the value would silently alias
    AUDIO_INPUT. Add it once that bug is fixed upstream.
    """

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


class Mode(str, Enum):
    """How the route was driven. The registry's cell ids already carry this
    axis as a segment (221 `.nonstream.`, 37 `.stream.`)."""

    NONSTREAM = "nonstream"
    STREAM = "stream"
    BATCH = "batch"
    WEBSOCKET = "websocket"


_M = TypeVar("_M")


def _scalar(value: object) -> str:
    """`str(member)` on a (str, Enum) gives 'Route.RESPONSES', not 'responses'
    -- StrEnum would not, but it is 3.11+ and this repo floors at 3.10. So the
    value is read explicitly, once, for every enum field."""
    if isinstance(value, Enum):
        return str(value.value)  # pyright: ignore[reportAny]  # Enum.value is Any for every enum
    return str(value)


def _members(value: object) -> tuple[object, ...] | None:
    """The elements of a plural field, or None for anything that is not a tuple.

    Both callers hold the value as a plain object: `_canonical` because a call
    site can pass anything at runtime, the serializer because `asdict` hands the
    tuple back inside an untyped dict. The elements are re-declared as plain
    objects here and converted by `_scalar` like any other value.
    """
    return cast("tuple[object, ...]", value) if isinstance(value, tuple) else None


def _canonical(name: str, value: object, member_type: type[_M]) -> tuple[_M, ...]:
    """A plural field's members: validated, deduped, and sorted by the value
    they serialize to.

    `models=("gpt-5.5")` is a str, not a tuple, and iterating it would declare
    one model per character. Anything that is not a tuple is refused here, which
    runs where the decorator does: at import, so pytest reports a collection
    error naming the file instead of shipping garbage properties. An empty
    string member is dropped rather than refused, because `models` is fed from
    env-overridable constants and a blank override must not break collection.
    """
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
    """What a test is about.

    Every field is optional in this first phase -- nothing is enforced, and the
    backfill of the existing ~908 tests comes later. Named `Subject` rather than
    `TestMeta` because pytest tries to collect any imported class named `Test*`
    and would warn in every one of the ~570 modules that import it.

    `providers`, `models` and `capabilities` are plural because one test node
    routinely drives several: the claude_code matrix runs haiku, sonnet and opus
    in a single body, and a spend test calls two providers on one key. Each is
    an independent set. No positional pairing is implied between `providers` and
    `models` (one provider x three models is the common case), and none could
    survive anyway, since each tuple is deduped and sorted on its own.
    """

    domain: Domain | None = None
    route: Route | None = None
    providers: tuple[Provider, ...] = ()
    models: tuple[str, ...] = ()
    capabilities: tuple[Capability, ...] = ()
    mode: Mode | None = None

    def __post_init__(self) -> None:
        """Canonicalize every plural field at declaration, so the committed run
        files diff cleanly however a test spelled the tuple, and the serializer
        stays field-agnostic."""
        object.__setattr__(self, "providers", _canonical("providers", self.providers, Provider))
        object.__setattr__(self, "models", _canonical("models", self.models, str))
        object.__setattr__(self, "capabilities", _canonical("capabilities", self.capabilities, Capability))


def meta(subject: Subject) -> pytest.MarkDecorator:
    """Attach a `Subject` to a test: `@meta(Subject(route=Route.RESPONSES, ...))`.

    A typed wrapper around `pytest.mark.meta` (registered in conftest.py's
    `pytest_configure`, like `covers`) so passing the wrong thing is a type
    error rather than a property that silently never appears.

    Separate from `@pytest.mark.covers` on purpose: `covers` args are flattened
    by `dedupe_covers`, which drops non-strings silently, and by
    tests/integration/conftest.py, which has no such filter and would hard-fail
    collection. `covers` is untouched by this change.
    """
    return pytest.mark.meta(subject)


_P = ParamSpec("_P")
_R = TypeVar("_R")
_Y = TypeVar("_Y")

MAX_STEPS: Final = 50
MAX_STEP_CHARS: Final = 200

STEP_FRAMES: Final = 1
"""Frames a `@step` wrapper puts between a helper and its caller. A decorated
helper that warns about its caller adds this to `stacklevel`
(`stacklevel=2 + STEP_FRAMES`), or the warning is reported at the wrapper."""


class _StepRecorder:
    """The ordered step log for the running test.

    A plain lock-guarded list rather than a ContextVar: ContextVars do not
    propagate into worker threads, and several e2e helpers call out from
    threads. Under xdist each worker is its own process, so there is no
    cross-test bleed beyond what the per-test reset already handles.
    """

    def __init__(self) -> None:
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
        """
        cleaned = " ".join(label.split())[:MAX_STEP_CHARS]
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


STEPS: Final = _StepRecorder()


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
    `ResourceManager.key` goes through `ProxyClient.generate_key`, a domain
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
                STEPS.record(label)
            with _inside_step():
                result = fn(*args, **kwargs)
            if opens_a_context and isinstance(result, AbstractContextManager):
                context: Final = cast("AbstractContextManager[object]", result)
                return cast("_R", _StepContext(context))
            return result

        return wrapper

    return decorate


_REPEATED: Final[dict[str, str]] = {"providers": "provider", "models": "model", "capabilities": "capability"}


def _declared_subject(args: tuple[object, ...]) -> Subject | None:
    """The `Subject` a `meta` marker carries, or None for anything else.

    `Mark.args` is `tuple[Any, ...]`; taking it as `tuple[object, ...]` is what
    keeps the Any from leaking past this line. A bare `@pytest.mark.meta` (no
    args) and a `@pytest.mark.meta("spend-budgets")` (wrong type) both land here,
    and neither may produce a property whose value is a repr.
    """
    first = args[0] if args else None
    return first if isinstance(first, Subject) else None


def subject_properties(item: pytest.Item) -> tuple[tuple[str, str], ...]:
    """The declared half, in dataclass field order.

    `_REPEATED` is the only field-specific knowledge here: which fields are
    plural, and the SINGULAR name their repeated <property> goes out under. A new
    scalar field needs no edit. Empty fields emit nothing; the emitter is what
    guarantees every key exists in the JSON, with `providers` and `models` as
    `[]` when nothing was declared."""
    marker = item.get_closest_marker("meta")
    if marker is None:
        return ()
    subject = _declared_subject(marker.args)
    if subject is None:
        return ()
    fields: dict[str, object] = asdict(subject)
    pairs: list[tuple[str, str]] = []
    for name, value in fields.items():
        repeated = _REPEATED.get(name)
        if repeated is not None:
            pairs.extend((repeated, _scalar(member)) for member in _members(value) or ())
        elif value is not None and value != "":
            pairs.append((name, _scalar(value)))
    return tuple(pairs)


def step_properties() -> tuple[tuple[str, str], ...]:
    """The step log as repeated `step` properties. Appended after the setup and
    call phases, never at collection."""
    return tuple(("step", label) for label in STEPS.taken())
