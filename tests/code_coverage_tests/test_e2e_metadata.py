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
from pathlib import Path

import pytest
from e2e_metadata import MAX_STEPS, STEP_FRAMES, STEPS, step, step_properties
from junit_properties import attach_result_properties, attach_step_properties


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
