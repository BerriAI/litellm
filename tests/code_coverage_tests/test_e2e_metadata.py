"""The e2e step log: what `@step` records, and how it attaches to a JUnit item.

Harness logic, so it lives here rather than under tests/e2e, which holds only
tests that drive a live proxy. The harness modules are imported off
``-o pythonpath=tests/e2e``, the way CI's provider_replay_harness job runs this
file. test_e2e_junit_report.py pins what reaches the XML through the real
conftest.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path

import pytest
from e2e_metadata import (
    MAX_STEPS,
    STEP_FRAMES,
    STEPS,
    Capability,
    Domain,
    Mode,
    Provider,
    Route,
    Subject,
    meta,
    step,
    step_properties,
    subject_properties,
)
from junit_properties import (
    attach_result_properties,
    attach_step_properties,
    package_from_nodeid,
    result_properties,
    source_from_item,
)


@pytest.fixture(autouse=True)
def empty_step_log() -> Generator[None]:
    """Each test starts from an empty log and leaves none behind, as conftest's
    `pytest_runtest_setup` hook arranges for every live test."""
    STEPS.reset()
    yield
    STEPS.reset()


def collected_item(request: pytest.FixtureRequest, name: str) -> pytest.Item:
    """The Item pytest collected for test ``name`` in this file, as pytest built it."""
    return next(item for item in request.session.items if item.path == request.path and item.name == name)


def fixed_prefix(item: pytest.Item, covers: str) -> tuple[tuple[str, str], ...]:
    """The three-tuple every testcase in the suite has carried since before the
    typed marker existed. Its shape and order are spelled out rather than taken
    from `result_properties`, so a change to either fails a test instead of
    agreeing with itself. `package` and `source` are read off the item, since
    this file sits outside the suite root; tests/e2e/test_junit_properties.py
    pins their values."""
    return (
        ("package", package_from_nodeid(item.nodeid)),
        ("covers", covers),
        ("source", source_from_item(item)),
    )


class TestSubjectProperties:
    """The declared half: `@meta(Subject(...))` -> `<property>` pairs.

    Markers are applied at run time via `request.applymarker`, the idiom the
    `covers` tests above already use, so the coverage registry's collect-only pass
    never sees a marker that exists only to be serialized.
    """

    def test_every_declared_field_becomes_a_property_in_field_order(self, request: pytest.FixtureRequest) -> None:
        """One pass over `dataclasses.asdict`: declaration order is emission order,
        a (str, Enum) member is written as its `.value` and never `str(member)`,
        and every plural field emits a repeated SINGULAR name (`provider`, `model`,
        `capability`), one <property> per member and never a delimiter-joined value."""
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
        """The claude_code matrix shape: one test node drives haiku, sonnet and opus
        through a single provider. The two lists are independent sets, so their
        lengths need not agree and no model is tied to a provider by position."""
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
        """No `provider`, `model` or `capability` property at all, rather than one
        with an empty value: the emitter is what turns absence into `[]`."""
        test = type(self).test_an_empty_plural_field_emits_nothing
        request.applymarker(meta(Subject(domain=Domain.MANAGEMENT)))
        assert subject_properties(collected_item(request, test.__name__)) == (("domain", "management"),)

    def test_scalar_property_names_are_the_dataclass_field_names(self, request: pytest.FixtureRequest) -> None:
        """The mapping is `asdict`, not a hand-written table: a scalar field added
        to `Subject` later serializes under its own name with no edit to the
        serializer. Proven by reading the field list back off the dataclass."""
        test = type(self).test_scalar_property_names_are_the_dataclass_field_names
        request.applymarker(meta(Subject(domain=Domain.UNKNOWN, route=Route.HEALTH, mode=Mode.STREAM)))
        declared = tuple(field.name for field in fields(Subject))
        emitted = tuple(name for name, _ in subject_properties(collected_item(request, test.__name__)))
        assert emitted == tuple(name for name in declared if name in {"domain", "route", "mode"})

    def test_every_plural_field_is_deduped_and_sorted_at_declaration(self) -> None:
        """Canonicalized in `__post_init__`, so two tests that spelled the same set
        in different orders produce byte-identical properties and the committed run
        files diff cleanly. Sorted by the value that is serialized, which for an
        enum is its `.value` and not its member name."""
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
        """`models=("gpt-5.5")` is a str, not a one-member tuple: the parentheses
        do nothing without the trailing comma, and iterating the str would declare
        one model per character. basedpyright flags it at the call site; this is
        the runtime half, raised where the decorator runs, so it lands as a
        collection error naming the file. `replace` is the untyped way in, since
        the typed constructor would not let the test spell the mistake."""
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
        """A bare "openai" where `Provider.OPENAI` belongs would serialize fine
        today and stop joining the day the enum value is renamed."""
        with pytest.raises(TypeError, match=rf"Subject\.{field} takes {member_type} members"):
            _ = replace(Subject(), **{field: value})

    def test_a_blank_model_is_dropped_rather_than_refused(self) -> None:
        """`models` is fed from env-overridable constants. A blank override is the
        operator's mistake, and it must cost one missing property, not the
        collection of the whole module."""
        assert Subject(models=("", "gpt-5.5")).models == ("gpt-5.5",)

    def test_the_typed_marker_only_ever_appends_to_the_fixed_prefix(self, request: pytest.FixtureRequest) -> None:
        """Loki, Grafana and the status page read `package`/`covers`/`source`; the
        typed fields ride behind them and must not disturb them."""
        test = type(self).test_the_typed_marker_only_ever_appends_to_the_fixed_prefix
        request.applymarker(pytest.mark.covers("quota_management.budget.key.blocks_over_limit"))
        request.applymarker(meta(Subject(route=Route.SPEND_REPORTING)))
        item = collected_item(request, test.__name__)
        assert result_properties(item) == fixed_prefix(item, "quota_management.budget.key.blocks_over_limit") + (
            ("route", "spend_reporting"),
        )

    def test_a_test_with_only_the_old_string_covers_is_unchanged(self, request: pytest.FixtureRequest) -> None:
        """The existing `@pytest.mark.covers("cell.id")` call sites keep emitting
        exactly what they emitted before the typed marker existed."""
        test = type(self).test_a_test_with_only_the_old_string_covers_is_unchanged
        request.applymarker(pytest.mark.covers("llm.responses.openai.tool_use.nonstream.works"))
        item = collected_item(request, test.__name__)
        assert result_properties(item) == fixed_prefix(item, "llm.responses.openai.tool_use.nonstream.works")

    def test_a_test_with_neither_marker_carries_only_the_prefix(self, request: pytest.FixtureRequest) -> None:
        """Which is every test in the suite until the backfill lands: an empty
        `covers` and no typed properties at all, never five empty ones."""
        test = type(self).test_a_test_with_neither_marker_carries_only_the_prefix
        item = collected_item(request, test.__name__)
        assert subject_properties(item) == ()
        assert result_properties(item) == fixed_prefix(item, "")

    def test_a_marker_carrying_something_other_than_a_subject_emits_nothing(
        self, request: pytest.FixtureRequest
    ) -> None:
        """`@meta` is typed, but `pytest.mark.meta` is not, and a bare
        `@pytest.mark.meta` carries no args at all. Neither may produce a property
        whose value is a repr."""
        test = type(self).test_a_marker_carrying_something_other_than_a_subject_emits_nothing
        request.applymarker(pytest.mark.meta("spend-budgets"))
        assert subject_properties(collected_item(request, test.__name__)) == ()


class TestProviderMirrorsLitellm:
    """`Provider` copies litellm's `LlmProviders` values rather than importing
    them, so collecting tests/e2e never needs the litellm package -- the suite is
    shipped to the e2e runner image on its own, and a `from litellm...` at module
    scope in a test file would turn a missing package into a collection error for
    every test rather than a slow import.

    A copy can drift, so it is checked here, wherever litellm IS importable (a dev
    checkout, this repo's own CI). Where it is not, the whole point is that
    nothing fails, so the check skips.
    """

    def test_every_provider_value_is_a_real_litellm_provider(self) -> None:
        """One direction only. litellm ships 155 providers and the e2e suite names
        a handful; a value missing from `Provider` is a line to add when a test
        needs it, but a value that is not a provider at all would ship a property
        no consumer can join on."""
        try:
            from litellm.types.utils import LlmProviders
        except ImportError:  # pragma: no cover - the runner image's shape
            pytest.skip("litellm is not importable here, which is the property under test")
        known = {str(member.value) for member in LlmProviders}
        unknown = sorted(member.value for member in Provider if member.value not in known)
        assert not unknown, f"not LlmProviders values: {unknown}"


class TestStepRecording:
    """`@step`-decorated harness helpers append to the running test's story as
    they execute.

    Each test here starts from an empty log because `empty_step_log` resets the
    recorder first, the same reset conftest's `pytest_runtest_setup` gives every
    live test.
    """

    def test_steps_land_in_call_order(self) -> None:
        @step("register deployment")
        def register() -> str:
            return "model-id"

        @step("generate virtual key")
        def generate() -> str:
            return "sk-x"

        _ = register()
        _ = generate()
        assert STEPS.taken() == ("register deployment", "generate virtual key")

    def test_a_decorated_helper_still_returns_exactly_what_it_did(self) -> None:
        """`@step` records, it does not intercept: arguments, return value and
        `__name__` all survive it, so decorating a live harness method cannot
        change what the test observes."""

        @step("POST /chat/completions")
        def chat(key: str, *, model: str) -> str:
            return f"{key}:{model}"

        assert chat("sk-x", model="gpt-5.5") == "sk-x:gpt-5.5"
        assert chat.__name__ == "chat"

    def test_a_helper_that_raises_leaves_its_own_label_last(self) -> None:
        """The whole point of the field. The label is recorded BEFORE the call, so
        a test that dies inside a helper keeps a partial story whose last element
        names the helper it died in."""

        @step("generate virtual key")
        def generate() -> str:
            return "sk-x"

        @step("POST /chat/completions")
        def chat() -> None:
            raise RuntimeError("502 from upstream")

        _ = generate()
        with pytest.raises(RuntimeError, match="502 from upstream"):
            chat()
        assert STEPS.taken() == ("generate virtual key", "POST /chat/completions")

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

    def test_reset_empties_the_log_so_one_test_never_inherits_another_s(self) -> None:
        STEPS.record("register deployment")
        STEPS.reset()
        assert STEPS.taken() == ()
        assert step_properties() == ()

    def test_steps_serialize_as_repeated_properties_in_order(self) -> None:
        """Repeated rather than joined on a delimiter: the labels are free text, so
        no separator can be reserved, and a repeated property has none to corrupt."""
        STEPS.record('attach guardrail, comma & "quoted" <tag>')
        STEPS.record("POST /chat/completions")
        assert step_properties() == (
            ("step", 'attach guardrail, comma & "quoted" <tag>'),
            ("step", "POST /chat/completions"),
        )

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


class TestNestedSteps:
    """Harness layers call each other, so a step's helper routinely calls other
    decorated helpers. Only the outermost records."""

    def test_a_step_called_inside_a_step_is_not_recorded(self) -> None:
        """`ResourceManager.key` wraps `ProxyClient.generate_key`: one action, one
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


class TestAttachStepProperties:
    def test_steps_are_appended_after_the_collected_properties(self, request: pytest.FixtureRequest) -> None:
        """Order inside `<properties>` is list order, so the story reads after the
        fixed prefix the collection hook already attached."""
        test = type(self).test_steps_are_appended_after_the_collected_properties
        item = collected_item(request, test.__name__)
        attach_result_properties(item)
        STEPS.record("register deployment")
        STEPS.record("POST /chat/completions")
        attach_step_properties(item)
        assert [name for name, _ in item.user_properties] == ["package", "covers", "source", "step", "step"]
        assert [value for name, value in item.user_properties if name == "step"] == [
            "register deployment",
            "POST /chat/completions",
        ]

    def test_a_rerun_replaces_the_story_rather_than_appending_a_second_one(
        self, request: pytest.FixtureRequest
    ) -> None:
        """The suite runs with `--reruns 1`. Without this the retry's steps would
        queue up behind the first attempt's and the report would read as one test
        that did everything twice."""
        test = type(self).test_a_rerun_replaces_the_story_rather_than_appending_a_second_one
        item = collected_item(request, test.__name__)
        STEPS.record("attempt one died here")
        attach_step_properties(item)
        STEPS.reset()
        STEPS.record("attempt two got further")
        attach_step_properties(item)
        assert [value for name, value in item.user_properties if name == "step"] == ["attempt two got further"]
