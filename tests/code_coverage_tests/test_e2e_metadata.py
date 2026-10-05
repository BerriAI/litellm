"""The e2e test metadata: `@meta(Subject(...))` properties and the step recorder's edge cases.

Harness logic, so it lives here rather than under tests/e2e, which holds only
tests that drive a live proxy. The harness modules are imported off
``PYTHONPATH=tests/e2e``, the way the Code Quality workflow's
test_e2e_metadata step runs this file. Call order, the failing test's last step,
the per-test reset and the JUnit attach are pinned end to end in
test_e2e_junit_report.py.
"""

from __future__ import annotations

import ast
import inspect
import re
import string
import threading
import warnings
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from types import UnionType
from typing import Final, cast, get_args, get_type_hints

import pytest
from e2e_metadata import (
    MASK,
    MAX_STEPS,
    STEP_FRAMES,
    STEPS,
    Capability,
    Domain,
    Mode,
    Provider,
    Route,
    StepRecorder,
    Subject,
    environment_secrets,
    meta,
    step,
    subject_properties,
)
from junit_properties import package_from_nodeid, result_properties, source_from_item
from proxy_client import ProxyClient
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo


@pytest.fixture(autouse=True)
def empty_step_log() -> Generator[None]:
    """Each test starts from an empty log and leaves none behind, as conftest's
    `pytest_runtest_setup` hook arranges for every live test."""
    STEPS.reset()
    yield
    STEPS.reset()


def collected_item(request: pytest.FixtureRequest, name: str) -> pytest.Item:
    return next(item for item in request.session.items if item.path == request.path and item.name == name)


def fixed_prefix(item: pytest.Item, covers: str) -> tuple[tuple[str, str], ...]:
    """Spelled out rather than taken from `result_properties`, so a change to either fails a test."""
    return (
        ("package", package_from_nodeid(item.nodeid)),
        ("covers", covers),
        ("source", source_from_item(item)),
    )


class TestSubjectProperties:
    """Markers go on via `request.applymarker` so the coverage registry's collect-only pass never sees them."""

    def test_every_declared_field_becomes_a_property_in_field_order(self, request: pytest.FixtureRequest) -> None:
        test = type(self).test_every_declared_field_becomes_a_property_in_field_order
        request.applymarker(
            meta(
                Subject(
                    domain=Domain.SPEND_BUDGETS,
                    route=Route.CHAT_COMPLETIONS,
                    providers=(Provider.GEMINI, Provider.ANTHROPIC),
                    models=("gemini-2.5-flash", "claude-haiku-4-5"),
                    capabilities=(Capability.VISION, Capability.FUNCTION_CALLING, Capability.VISION),
                    mode=Mode.NONSTREAM,
                )
            )
        )
        assert subject_properties(collected_item(request, test.__name__)) == (
            ("domain", "spend-budgets"),
            ("route", "chat_completions"),
            ("provider", "anthropic"),
            ("provider", "gemini"),
            ("model", "claude-haiku-4-5"),
            ("model", "gemini-2.5-flash"),
            ("capability", "function_calling"),
            ("capability", "vision"),
            ("mode", "nonstream"),
        )

    def test_one_provider_with_three_models_pairs_nothing(self, request: pytest.FixtureRequest) -> None:
        test = type(self).test_one_provider_with_three_models_pairs_nothing
        request.applymarker(
            meta(
                Subject(
                    providers=(Provider.BEDROCK,),
                    models=("claude-sonnet-4-5", "claude-opus-4-7", "claude-haiku-4-5"),
                )
            )
        )
        assert subject_properties(collected_item(request, test.__name__)) == (
            ("provider", "bedrock"),
            ("model", "claude-haiku-4-5"),
            ("model", "claude-opus-4-7"),
            ("model", "claude-sonnet-4-5"),
        )

    def test_an_empty_plural_field_emits_nothing(self, request: pytest.FixtureRequest) -> None:
        test = type(self).test_an_empty_plural_field_emits_nothing
        request.applymarker(meta(Subject(domain=Domain.MANAGEMENT)))
        assert subject_properties(collected_item(request, test.__name__)) == (("domain", "management"),)

    def test_scalar_property_names_are_the_dataclass_field_names(self, request: pytest.FixtureRequest) -> None:
        test = type(self).test_scalar_property_names_are_the_dataclass_field_names
        request.applymarker(meta(Subject(domain=Domain.UNKNOWN, route=Route.HEALTH, mode=Mode.STREAM)))
        declared = tuple(field.name for field in fields(Subject))
        emitted = tuple(name for name, _ in subject_properties(collected_item(request, test.__name__)))
        assert emitted == tuple(name for name in declared if name in {"domain", "route", "mode"})

    def test_every_plural_field_is_deduped_and_sorted_at_declaration(self) -> None:
        subject = Subject(
            providers=(Provider.OPENAI, Provider.ANTHROPIC, Provider.OPENAI),
            models=("gpt-5.5", "claude-haiku-4-5", "gpt-5.5"),
            capabilities=(Capability.VISION, Capability.REASONING, Capability.VISION),
        )
        assert subject.providers == (Provider.ANTHROPIC, Provider.OPENAI)
        assert subject.models == ("claude-haiku-4-5", "gpt-5.5")
        assert subject.capabilities == (Capability.REASONING, Capability.VISION)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("models", "gpt-5.5"),
            ("models", ["gpt-5.5"]),
            ("providers", Provider.OPENAI),
            ("providers", [Provider.OPENAI]),
            ("capabilities", Capability.VISION),
            ("capabilities", frozenset({Capability.VISION})),
        ],
    )
    def test_a_plural_field_refuses_anything_but_a_tuple(self, field: str, value: object) -> None:
        """`replace` is the untyped way in, since the typed constructor would not let the test spell the mistake."""
        with pytest.raises(TypeError, match=rf"Subject\.{field} must be a tuple"):
            _ = replace(Subject(), **{field: value})

    @pytest.mark.parametrize(
        ("field", "value", "member_type"),
        [
            ("providers", ("openai",), "Provider"),
            ("capabilities", ("vision",), "Capability"),
            ("models", (5,), "str"),
        ],
    )
    def test_a_plural_field_refuses_a_member_of_the_wrong_type(
        self, field: str, value: object, member_type: str
    ) -> None:
        with pytest.raises(TypeError, match=rf"Subject\.{field} takes {member_type} members"):
            _ = replace(Subject(), **{field: value})

    def test_a_blank_model_is_dropped_rather_than_refused(self) -> None:
        """A blank env override must cost one missing property, not collection of the whole module."""
        assert Subject(models=("", "gpt-5.5")).models == ("gpt-5.5",)

    def test_the_typed_marker_only_ever_appends_to_the_fixed_prefix(self, request: pytest.FixtureRequest) -> None:
        test = type(self).test_the_typed_marker_only_ever_appends_to_the_fixed_prefix
        request.applymarker(pytest.mark.covers("quota_management.budget.key.blocks_over_limit"))
        request.applymarker(meta(Subject(route=Route.SPEND_REPORTING)))
        item = collected_item(request, test.__name__)
        assert result_properties(item) == fixed_prefix(item, "quota_management.budget.key.blocks_over_limit") + (
            ("route", "spend_reporting"),
        )

    def test_a_test_with_only_the_old_string_covers_is_unchanged(self, request: pytest.FixtureRequest) -> None:
        test = type(self).test_a_test_with_only_the_old_string_covers_is_unchanged
        request.applymarker(pytest.mark.covers("llm.responses.openai.tool_use.nonstream.works"))
        item = collected_item(request, test.__name__)
        assert result_properties(item) == fixed_prefix(item, "llm.responses.openai.tool_use.nonstream.works")

    def test_a_test_with_neither_marker_carries_only_the_prefix(self, request: pytest.FixtureRequest) -> None:
        test = type(self).test_a_test_with_neither_marker_carries_only_the_prefix
        item = collected_item(request, test.__name__)
        assert subject_properties(item) == ()
        assert result_properties(item) == fixed_prefix(item, "")

    def test_a_marker_carrying_something_other_than_a_subject_emits_nothing(
        self, request: pytest.FixtureRequest
    ) -> None:
        test = type(self).test_a_marker_carrying_something_other_than_a_subject_emits_nothing
        request.applymarker(pytest.mark.meta("spend-budgets"))
        assert subject_properties(collected_item(request, test.__name__)) == ()


class TestProviderMirrorsLitellm:
    """`Provider` copies `LlmProviders` values so collecting tests/e2e never needs litellm; skips where it is absent."""

    def test_every_provider_value_is_a_real_litellm_provider(self) -> None:
        try:
            from litellm.types.utils import LlmProviders
        except ImportError:  # pragma: no cover - the runner image's shape
            pytest.skip("litellm is not importable here, which is the property under test")
        known = {str(member.value) for member in LlmProviders}
        unknown = sorted(member.value for member in Provider if member.value not in known)
        assert not unknown, f"not LlmProviders values: {unknown}"


E2E_DIR: Final = Path(__file__).resolve().parents[1] / "e2e"


def _hand_typed_models(path: Path) -> Iterator[str]:
    for node in ast.walk(ast.parse(path.read_text())):
        match node:
            case ast.Call(func=ast.Name(id="Subject"), keywords=keywords):
                for keyword in keywords:
                    match keyword:
                        case ast.keyword(arg="models", value=ast.Tuple(elts=models)):
                            yield from (
                                f"{path.relative_to(E2E_DIR)}:{model.lineno} {model.value!r}"
                                for model in models
                                if isinstance(model, ast.Constant)
                            )
                        case _:
                            pass
            case _:
                pass


def test_a_declared_model_names_the_constant_the_test_drives() -> None:
    offenders: Final = tuple(
        offender for path in sorted(E2E_DIR.rglob("*.py")) for offender in _hand_typed_models(path)
    )
    assert offenders == ()


class TestStepRecording:
    """`@step`-decorated harness helpers append to the running test's story as
    they execute.

    Each test here starts from an empty log because `empty_step_log` resets the
    recorder first, the same reset conftest's `pytest_runtest_setup` gives every
    live test.
    """

    def test_a_decorated_helper_still_returns_exactly_what_it_did(self) -> None:
        """`@step` records, it does not intercept: arguments, return value and
        `__name__` all survive it, so decorating a live harness method cannot
        change what the test observes."""

        @step("POST /chat/completions")
        def chat(key: str, *, model: str) -> str:
            return f"{key}:{model}"

        assert chat("sk-x", model="gpt-5.5") == "sk-x:gpt-5.5"
        assert chat.__name__ == "chat"

    def test_a_poll_loop_is_one_step_in_the_story_not_fifty(self) -> None:
        @step("poll /spend/logs for the request id")
        def poll() -> None:
            return None

        for _ in range(20):
            poll()
        assert STEPS.taken() == ("poll /spend/logs for the request id",)

    def test_the_same_label_recorded_again_later_is_a_new_step(self) -> None:
        """Only CONSECUTIVE duplicates collapse; a helper called again after
        something else happened is a genuine second beat of the story."""
        STEPS.record("POST /chat/completions")
        STEPS.record("poll /spend/logs")
        STEPS.record("POST /chat/completions")
        assert STEPS.taken() == ("POST /chat/completions", "poll /spend/logs", "POST /chat/completions")

    def test_a_full_log_keeps_the_latest_steps_so_the_last_is_where_the_test_died(self) -> None:
        """A load test cannot bury the story in thousands of entries, and the cap
        drops from the front: the step a test died on is the newest, so it is the
        one that has to survive. The leading line says the story is partial."""
        for index in range(MAX_STEPS + 10):
            STEPS.record(f"call {index}")
        assert STEPS.taken() == (
            "(10 earlier steps not recorded)",
            *(f"call {index}" for index in range(10, MAX_STEPS + 10)),
        )

    def test_reset_forgets_what_a_full_log_dropped(self) -> None:
        for index in range(MAX_STEPS + 1):
            STEPS.record(f"call {index}")
        STEPS.reset()
        STEPS.record("register deployment")
        assert STEPS.taken() == ("register deployment",)

    def test_whitespace_is_normalized_and_an_empty_label_records_nothing(self) -> None:
        STEPS.record("  POST   /chat/completions\n  ")
        STEPS.record("   ")
        assert STEPS.taken() == ("POST /chat/completions",)

    def test_a_decorated_helper_warns_at_its_caller_with_step_frames(self) -> None:
        """`stacklevel` counts frames, and the wrapper is one of them: a cleanup
        helper that warns about its caller would otherwise report every warning at
        e2e_metadata.py. Pins `STEP_FRAMES` to the frames the wrapper really adds."""

        @step("delete team")
        def delete_team() -> None:
            warnings.warn("delete_team('t') failed", stacklevel=2 + STEP_FRAMES)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            delete_team()
        assert [Path(warning.filename).name for warning in caught] == [Path(__file__).name]


class _KeyBody(BaseModel):
    models: list[str] = []
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    team_id: str | None = None
    api_key: str | None = Field(default=None, repr=False)


class _Params(BaseModel):
    model: str
    api_key: str | None = Field(default=None, repr=False)


class _DeploymentBody(BaseModel):
    model_name: str
    params: _Params


def _field_type(annotation: object) -> object:
    """`X | None` is `X`: a placeholder reads the field when it is set."""
    present: Final = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    return present[0] if isinstance(annotation, UnionType) and len(present) == 1 else annotation


def _placeholders(owner: type) -> Iterator[tuple[str, str]]:
    tree: Final = ast.parse(inspect.getsource(owner))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            match decorator:
                case ast.Call(func=ast.Name(id="step"), args=[ast.Constant(value=str(label))]):
                    for _, field, _, _ in string.Formatter().parse(label):
                        if field is not None:
                            yield node.name, field
                case _:
                    pass


def _dotted_placeholders(owner: type) -> Iterator[tuple[str, str]]:
    return ((method, field) for method, field in _placeholders(owner) if "." in field)


def _fields_read(owner: type, method: str, field: str) -> tuple[FieldInfo, ...] | None:
    """The model fields a dotted placeholder reads, outermost first, or None if one doesn't exist."""
    root, *attributes = field.split(".")
    wrapped: Final = cast("Callable[..., object]", getattr(owner, method))
    hints: Final[Mapping[str, object]] = get_type_hints(inspect.unwrap(wrapped))
    current: object = _field_type(hints[root])  # rebind-ok: walks one type per attribute
    read: tuple[FieldInfo, ...] = ()  # rebind-ok: grows one field per attribute
    for attribute in attributes:
        if not (isinstance(current, type) and issubclass(current, BaseModel) and attribute in current.model_fields):
            return None
        read = (*read, current.model_fields[attribute])  # rebind-ok: grows one field per attribute
        current = _field_type(read[-1].annotation)  # rebind-ok: walks one type per attribute
    return read


SECRET_NAME: Final = re.compile(
    r"secret|password|api_key|access_key|private_key|credential_values|^token$|(access|auth|bearer|refresh|session)_token$"
)


def _models_in(annotation: object, seen: frozenset[type] = frozenset()) -> frozenset[type[BaseModel]]:
    """Every request model a value of this type can print, however deeply nested."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in seen:
            return frozenset()
        nested: Final = (
            _models_in(field.annotation, seen | {annotation}) for field in annotation.model_fields.values()
        )
        return frozenset({annotation}).union(*nested)
    args: Final = cast("tuple[object, ...]", get_args(annotation))
    return frozenset[type[BaseModel]]().union(*(_models_in(arg, seen) for arg in args))


def _printed_models(owner: type) -> frozenset[type[BaseModel]]:
    def hint(method: str, field: str) -> object:
        wrapped: Final = cast("Callable[..., object]", getattr(owner, method))
        hints: Final = cast("Mapping[str, object]", get_type_hints(inspect.unwrap(wrapped)))
        return hints[field.split(".")[0]]

    return frozenset[type[BaseModel]]().union(
        *(_models_in(hint(method, field)) for method, field in _placeholders(owner))
    )


class TestLabelTemplates:
    """A label's `{placeholders}` are filled from the call's own arguments, so the
    story says what the test asked for in words, and nothing the label doesn't name
    ever reaches the report."""

    def test_placeholders_take_the_call_arguments_and_defaults(self) -> None:
        @step('Send a request to {model} with the prompt "{content}" capped at {max_tokens} tokens')
        def chat(key: str, model: str, content: str, *, max_tokens: int = 16) -> None:
            return None

        chat("sk-live", "claude-haiku-4-5", content="hi")
        assert STEPS.taken() == ('Send a request to claude-haiku-4-5 with the prompt "hi" capped at 16 tokens',)

    def test_a_request_model_reads_as_only_the_fields_the_test_set(self) -> None:
        @step("Generate a virtual key with {body}")
        def generate_key(body: _KeyBody) -> None:
            return None

        generate_key(_KeyBody(models=["a", "b"], rpm_limit=3, tpm_limit=None, api_key="sk-live"))
        generate_key(_KeyBody())
        assert STEPS.taken() == (
            "Generate a virtual key with models: a, b and rpm limit: 3",
            "Generate a virtual key with default settings",
        )

    def test_calls_differing_only_in_arguments_are_separate_steps(self) -> None:
        @step('Send "{content}"')
        def chat(content: str) -> None:
            return None

        for content in ("one", "one", "two"):
            chat(content)
        assert STEPS.taken() == ('Send "one"', 'Send "two"')

    def test_a_placeholder_the_helper_does_not_take_fails_at_import(self) -> None:
        def chat(model: str) -> None:
            return None

        with pytest.raises(TypeError, match="modle"):
            _ = step("Send a request to {modle}")(chat)

    def test_a_dotted_placeholder_reads_one_field_of_a_request_model(self) -> None:
        @step("Add a deployment named {body.model_name} that calls {body.params.model}")
        def register_model(body: _DeploymentBody) -> None:
            return None

        register_model(_DeploymentBody(model_name="gpt", params=_Params(model="openai/gpt-5.5")))
        assert STEPS.taken() == ("Add a deployment named gpt that calls openai/gpt-5.5",)

    def test_a_placeholder_that_indexes_or_calls_is_refused(self) -> None:
        def chat(body: _DeploymentBody) -> None:
            return None

        with pytest.raises(TypeError, match=r"body\.messages\[0\]"):
            _ = step("Send {body.messages[0]}")(chat)

    @pytest.mark.parametrize("owner", [ProxyClient], ids=["ProxyClient"])
    def test_every_dotted_placeholder_in_the_harness_names_a_real_field(self, owner: type) -> None:
        """A dotted placeholder is read on every live call, so one naming a field the
        request model doesn't have would fail the test calling it, not the label."""
        placeholders: Final = tuple(_dotted_placeholders(owner))
        assert placeholders
        assert [
            f"{method}: {field}" for method, field in placeholders if _fields_read(owner, method, field) is None
        ] == []

    @pytest.mark.parametrize("owner", [ProxyClient], ids=["ProxyClient"])
    def test_every_dotted_placeholder_in_the_harness_reads_a_field_the_caller_must_set(self, owner: type) -> None:
        """A field with a default is usually left unset, and an unset field prints
        nothing, so the step would read "Save a provider credential for "."""
        unset: Final = tuple(
            f"{method}: {field}"
            for method, field in _dotted_placeholders(owner)
            if not all(info.is_required() for info in _fields_read(owner, method, field) or ())
        )
        assert unset == ()

    @pytest.mark.parametrize("owner", [ProxyClient], ids=["ProxyClient"])
    def test_every_secret_field_a_label_can_print_is_hidden(self, owner: type) -> None:
        """A `{body}` label prints nested models too, so a callback's credentials
        inside key metadata would land in the public report unless marked `repr=False`."""
        models: Final = _printed_models(owner)
        assert models
        exposed: Final = sorted(
            f"{model.__name__}.{name}"
            for model in models
            for name, field in model.model_fields.items()
            if field.repr and SECRET_NAME.search(name)
        )
        assert exposed == []

    def test_escaped_braces_stay_literal(self) -> None:
        @step("GET /v1/batches/{{id}}")
        def retrieve_batch(batch_id: str) -> None:
            return None

        retrieve_batch("batch_123")
        assert STEPS.taken() == ("GET /v1/batches/{id}",)


class TestSecretMasking:
    """Steps are published with the results, so a credential the run holds is
    masked wherever it shows up in a label: a nested model field nobody marked
    `repr=False`, a dict value, or a prompt."""

    def test_a_secret_anywhere_in_a_label_is_masked(self) -> None:
        recorder: Final = StepRecorder(secrets=lambda: ("sk-live-abcdef123", "wandb-9f8e7d6c"))
        recorder.record("Generate a virtual key with callback vars: wandb api key: wandb-9f8e7d6c")
        recorder.record('Send "use sk-live-abcdef123 please" to claude-haiku-4-5')
        assert recorder.taken() == (
            f"Generate a virtual key with callback vars: wandb api key: {MASK}",
            f'Send "use {MASK} please" to claude-haiku-4-5',
        )

    def test_a_secret_is_masked_before_the_label_is_cut(self) -> None:
        secret: Final = "s3cr3t-" + "x" * 40
        recorder: Final = StepRecorder(secrets=lambda: (secret,))
        recorder.record("a" * 170 + " " + secret)
        assert recorder.taken() == ("a" * 170 + f" {MASK}",)

    def test_a_longer_secret_containing_a_shorter_one_is_masked_whole(self) -> None:
        recorder: Final = StepRecorder(secrets=lambda: ("abcdefgh", "abcdefgh-ijklmnop"))
        recorder.record("key abcdefgh-ijklmnop")
        assert recorder.taken() == (f"key {MASK}",)

    def test_only_secret_named_variables_long_enough_to_be_credentials_count(self) -> None:
        environ: Final = {
            "OPENAI_API_KEY": "sk-proj-0123456789",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG",
            "LITELLM_MASTER_KEY": "sk-test",
            "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/vertex.json",
            "KEYCLOAK_URL": "http://localhost:8080",
            "E2E_MODEL": "claude-haiku-4-5",
        }
        assert environment_secrets(environ) == frozenset(
            {"sk-proj-0123456789", "wJalrXUtnFEMI/K7MDENG", "/secrets/vertex.json"}
        )

    def test_the_shared_log_masks_the_live_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "wandb-live-5a4b3c2d")

        @step("Generate a virtual key with {body}")
        def generate_key(body: _KeyBody) -> None:
            return None

        generate_key(_KeyBody(team_id="wandb-live-5a4b3c2d"))
        assert STEPS.taken() == (f"Generate a virtual key with team id: {MASK}",)


class TestNestedSteps:
    """Harness layers call each other, so a step's helper routinely calls other
    decorated helpers. Only the outermost records."""

    def test_a_step_called_inside_a_step_is_not_recorded(self) -> None:
        """`ProxyClient.create_model` wraps `register_model`: one action, one
        beat of the story, at the level the test called in at."""

        @step("POST /key/generate")
        def generate_key() -> str:
            return "sk-x"

        @step("generate virtual key")
        def key() -> str:
            return generate_key()

        assert key() == "sk-x"
        assert STEPS.taken() == ("generate virtual key",)

    def test_the_inner_step_records_again_once_the_outer_one_returns(self) -> None:
        @step("POST /key/generate")
        def generate_key() -> str:
            return "sk-x"

        @step("generate virtual key")
        def key() -> str:
            return generate_key()

        _ = key()
        _ = generate_key()
        assert STEPS.taken() == ("generate virtual key", "POST /key/generate")

    def test_an_inner_step_that_raises_leaves_the_outer_label_last_and_unwinds(self) -> None:
        """The helper the test called is where it died, and the nesting flag is
        released on the way out, so the next top-level call still records."""

        @step("POST /team/new")
        def post_team() -> None:
            raise RuntimeError("/team/new answered 500")

        @step("create team with a budget")
        def create_team() -> None:
            post_team()

        @step("POST /chat/completions")
        def chat() -> None:
            return None

        with pytest.raises(RuntimeError, match="answered 500"):
            create_team()
        chat()
        assert STEPS.taken() == ("create team with a budget", "POST /chat/completions")

    def test_a_worker_thread_a_step_fans_out_to_records_its_own_steps(self) -> None:
        """Nesting is per thread: a load helper that fans chats out to workers is
        not inside a step on those workers, so their calls are still recorded."""

        @step("POST /chat/completions")
        def chat() -> None:
            return None

        @step("fire concurrent chats")
        def fan_out() -> None:
            worker = threading.Thread(target=chat)
            worker.start()
            worker.join()

        fan_out()
        assert STEPS.taken() == ("fire concurrent chats", "POST /chat/completions")


class TestContextManagerSteps:
    """A `@contextmanager` helper's setup and cleanup run at `__enter__` and
    `__exit__`, after the decorated call has returned. Both still count as part
    of its step; the `with` body is the test's own code and records as usual."""

    def test_setup_and_cleanup_stay_inside_the_step_and_the_body_records(self) -> None:
        @step("run a SQL statement")
        def execute() -> None:
            return None

        @step("create a read-only database role")
        @contextmanager
        def restricted_user() -> Generator[str]:
            execute()
            try:
                yield "reader"
            finally:
                execute()

        @step("POST /chat/completions")
        def chat() -> None:
            return None

        with restricted_user() as user:
            assert user == "reader"
            chat()
        assert STEPS.taken() == ("create a read-only database role", "POST /chat/completions")

    def test_a_test_that_dies_in_the_with_body_keeps_its_last_step_last(self) -> None:
        """The guarantee the field makes: the cleanup that runs on the way out of
        the `with` must not append a step behind the one the test died on."""

        @step("drop the role")
        def drop_role() -> None:
            return None

        @step("create a read-only database role")
        @contextmanager
        def restricted_user() -> Generator[None]:
            try:
                yield
            finally:
                drop_role()

        @step("POST /chat/completions")
        def chat() -> None:
            raise RuntimeError("502 from upstream")

        with pytest.raises(RuntimeError, match="502 from upstream"), restricted_user():
            chat()
        assert STEPS.taken() == ("create a read-only database role", "POST /chat/completions")

    def test_the_wrapped_context_keeps_its_exception_handling(self) -> None:
        """`__exit__` is forwarded, return value included, so a context that
        suppresses an exception still does."""

        @step("hold an advisory lock")
        @contextmanager
        def swallowing() -> Generator[None]:
            try:
                yield
            except KeyError:
                pass

        with swallowing():
            raise KeyError("suppressed by the context")
        assert STEPS.taken() == ("hold an advisory lock",)

    def test_a_bare_generator_is_refused_where_the_decorator_runs(self) -> None:
        """Its body runs only as the caller iterates, interleaved with the caller's
        own steps, so no single point in the story is where it happened. Refused at
        decoration, which for a harness module is import, so it lands as a
        collection error rather than a story that quietly reads out of order."""

        def rows() -> Generator[int]:
            yield 1

        with pytest.raises(TypeError, match="cannot wrap the generator function"):
            _ = step("poll /spend/logs")(rows)
