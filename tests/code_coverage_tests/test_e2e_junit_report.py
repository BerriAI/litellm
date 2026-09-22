"""The JUnit report itself, written by a real pytest run.

No proxy. test_e2e_metadata.py pins the functions that build the properties;
this pins what reaches the XML once pytest, its junitxml plugin,
pytest-rerunfailures and xdist are all in the loop. Each case writes a throwaway
suite into a tmp dir and runs it in a child interpreter with tests/e2e's
conftest.py loaded as a plugin, so the hooks under test are the ones the live
suite runs and the recorder is the real one, never a copy of either.

The timing that makes the recorded half work is pytest's, which is why it is
pinned here against the real thing: junitxml writes a testcase's properties from
its TEARDOWN report, and pytest builds that report from ``item.user_properties``
after the setup and call phases have both attached the steps. The suite runs
distributed, so every assertion is made in-process and again under ``-n 2``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType
from typing import Final
from xml.etree import ElementTree

import pytest

SUITE_DIR: Final = Path(__file__).resolve().parents[1] / "e2e"
CHILD_TIMEOUT_SECONDS: Final = 180

STORY_SUITE: Final = """
from collections.abc import Iterator
from pathlib import Path

import pytest
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta, step

FIRST_ATTEMPT_MADE = Path(__file__).with_name("first-attempt-made")


@step("generate virtual key")
def generate_key() -> None:
    return None


@step("create team")
def create_team() -> None:
    raise RuntimeError("/team/new answered 500")


@step("POST /chat/completions")
def chat(*, ok: bool) -> None:
    if not ok:
        raise AssertionError("status_code=502 from upstream")


@step("poll /spend/logs")
def poll_spend_logs() -> None:
    return None


@step("delete virtual key")
def delete_key() -> None:
    return None


@pytest.fixture
def key() -> Iterator[None]:
    generate_key()
    yield
    delete_key()


@pytest.fixture
def team(key: None) -> None:
    create_team()


def test_passes(key: None) -> None:
    chat(ok=True)
    poll_spend_logs()


def test_fails(key: None) -> None:
    chat(ok=False)
    poll_spend_logs()


def test_errors_in_setup(team: None) -> None:
    poll_spend_logs()


def test_passes_on_the_rerun(key: None) -> None:
    first_attempt = not FIRST_ATTEMPT_MADE.exists()
    FIRST_ATTEMPT_MADE.touch()
    chat(ok=not first_attempt)
    poll_spend_logs()


@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.BEDROCK, Provider.ANTHROPIC),
        models=("claude-sonnet-4-5", "claude-opus-4-7", "claude-haiku-4-5"),
        capabilities=(Capability.VISION, Capability.FUNCTION_CALLING),
        mode=Mode.STREAM,
    )
)
def test_declares_two_providers_and_three_models() -> None:
    assert Provider.BEDROCK.value == "bedrock"
"""

WIDE_FINALIZER_SUITE: Final = """
from collections.abc import Iterator

import pytest
from e2e_metadata import step


@step("generate virtual key")
def generate_key() -> None:
    return None


@step("delete shared team")
def delete_shared_team() -> None:
    return None


@pytest.fixture(scope="module")
def shared_team() -> Iterator[None]:
    yield
    delete_shared_team()


def test_uses_the_shared_team(shared_team: None) -> None:
    generate_key()
"""

WIDE_SETUP_ERROR_SUITE: Final = """
import pytest
from e2e_metadata import step


@step("log in to the identity provider")
def log_in() -> None:
    raise RuntimeError("identity provider is down")


@pytest.fixture(scope="module")
def identity() -> None:
    log_in()


def test_dies_in_a_module_scoped_fixture(identity: None) -> None:
    assert identity is None
"""

BARE_STR_SUITE: Final = """
from e2e_metadata import Subject, meta


@meta(Subject(models=("gpt-5.5")))
def test_never_collected() -> None:
    assert Subject is not None
"""

Properties = tuple[tuple[str, str], ...]


def write_suite(directory: Path, modules: Mapping[str, str]) -> None:
    """Lay a child suite out in ``directory``, with an ini file of its own.

    The ini pins the child's rootdir to the tmp dir wherever that lives, and its
    ``pythonpath`` is what makes tests/e2e's conftest.py, and the harness
    modules the child suite imports, importable under ``-I``.
    """
    _ = (directory / "pytest.ini").write_text(f"[pytest]\npythonpath = {shlex.quote(str(SUITE_DIR))}\n")
    for name, source in modules.items():
        _ = (directory / name).write_text(source)


def run_child_pytest(suite: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run pytest over ``suite`` in a fresh interpreter, hooked up like the live suite.

    ``-p conftest`` registers tests/e2e's conftest.py as a plugin, since a
    tmp dir outside tests/e2e would never pick it up by location. The parent's
    fixture-mode and addopts settings are dropped so a replay lane cannot leak
    into the child.
    """
    inherited: Final = {
        name: value
        for name, value in os.environ.items()
        if name != "PYTEST_ADDOPTS" and not name.startswith("E2E_FIXTURE_")
    }
    return subprocess.run(
        [sys.executable, "-I", "-m", "pytest", "-p", "conftest", "-p", "no:cacheprovider", *args, str(suite)],
        cwd=suite,
        env=inherited,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,
    )


def properties_by_test(testsuite: ElementTree.Element) -> Mapping[str, Properties]:
    """Every testcase's <property> pairs, in document order, keyed by test name."""
    return MappingProxyType(
        {
            testcase.get("name", ""): tuple(
                (prop.get("name", ""), prop.get("value", "")) for prop in testcase.iter("property")
            )
            for testcase in testsuite.iter("testcase")
        }
    )


def values(properties: Properties, name: str) -> tuple[str, ...]:
    return tuple(value for prop, value in properties if prop == name)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((), id="in-process"),
        pytest.param(
            ("-n", "2"),
            id="xdist",
            marks=pytest.mark.skipif(find_spec("xdist") is None, reason="pytest-xdist is not installed"),
        ),
    ],
)
def report(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Properties]:
    """One child run per distribution mode, shared by every assertion below.

    ``--reruns 1`` and the ``--only-rerun`` pattern are the live suite's own
    addopts. The two wide-scope modules sort ahead of the story, and next to each
    other, so in-process the second one's setup runs right after the first one's
    module-scoped finalizer.
    """
    distribution: Final[tuple[str, ...]] = request.param  # pyright: ignore[reportAny]  # pytest types request.param as Any
    suite: Final = tmp_path_factory.mktemp("suite")
    write_suite(
        suite,
        {
            "test_scope_a_finalizer.py": WIDE_FINALIZER_SUITE,
            "test_scope_b_setup_error.py": WIDE_SETUP_ERROR_SUITE,
            "test_story.py": STORY_SUITE,
        },
    )
    xml: Final = suite / "report.xml"
    child: Final = run_child_pytest(
        suite, f"--junitxml={xml}", "--reruns", "1", "--only-rerun", "status_code=5[0-9][0-9]", *distribution
    )
    assert xml.exists(), f"the child run wrote no JUnit report:\n{child.stdout}\n{child.stderr}"
    testsuite: Final = next(ElementTree.parse(xml).getroot().iter("testsuite"))
    outcomes: Final = {name: testsuite.get(name) for name in ("tests", "failures", "errors", "skipped")}
    assert outcomes == {"tests": "7", "failures": "1", "errors": "2", "skipped": "0"}, child.stdout
    return properties_by_test(testsuite)


class TestStepsReachTheReport:
    def test_a_passing_test_tells_its_story_in_call_order(self, report: Mapping[str, Properties]) -> None:
        """Fixture setup first, then the body. The finalizer's "delete virtual key"
        is cleanup and is deliberately not part of the story."""
        assert values(report["test_passes"], "step") == (
            "generate virtual key",
            "POST /chat/completions",
            "poll /spend/logs",
        )

    def test_a_failing_test_s_last_step_is_where_it_died(self, report: Mapping[str, Properties]) -> None:
        """The reason the field exists. Nothing the test never reached is listed,
        and no teardown step is appended behind the one it died on."""
        assert values(report["test_fails"], "step") == ("generate virtual key", "POST /chat/completions")

    def test_a_setup_error_keeps_the_steps_recorded_before_the_crash(self, report: Mapping[str, Properties]) -> None:
        """A fixture that raises never reaches the call phase, and setup is where
        an e2e test most often dies (proxy not ready, key creation failing), so
        the steps have to be attached after setup too."""
        assert values(report["test_errors_in_setup"], "step") == ("generate virtual key", "create team")

    def test_a_rerun_reports_only_the_attempt_junit_records(self, report: Mapping[str, Properties]) -> None:
        """The first attempt died on the chat call and the rerun got through. Steps
        are attached twice per attempt, and none of that may show up as a doubled
        or a stale story."""
        assert values(report["test_passes_on_the_rerun"], "step") == (
            "generate virtual key",
            "POST /chat/completions",
            "poll /spend/logs",
        )

    def test_a_setup_error_does_not_inherit_a_wider_finalizer_s_steps(self, report: Mapping[str, Properties]) -> None:
        """A module-scoped finalizer runs after the last test of its module, and
        a module-scoped fixture is set up before any function-scoped one. The log
        is emptied ahead of both, so the next test's setup error reports its own
        steps and not "delete shared team"."""
        assert values(report["test_uses_the_shared_team"], "step") == ("generate virtual key",)
        assert values(report["test_dies_in_a_module_scoped_fixture"], "step") == ("log in to the identity provider",)

    def test_steps_ride_behind_the_fixed_prefix(self, report: Mapping[str, Properties]) -> None:
        """`package`/`covers`/`source` are what Loki, Grafana and the status page
        already read, on every outcome including a setup error."""
        for name in ("test_passes", "test_fails", "test_errors_in_setup"):
            assert tuple(prop for prop, _ in report[name])[:4] == ("package", "covers", "source", "step"), name


class TestDeclaredPropertiesReachTheReport:
    def test_repeated_provider_model_and_capability_round_trip(self, report: Mapping[str, Properties]) -> None:
        """One <property> per member under the SINGULAR name, deduped and sorted,
        with no pairing between the two providers and the three models."""
        declared: Final = tuple(
            (prop, value)
            for prop, value in report["test_declares_two_providers_and_three_models"]
            if prop not in {"package", "covers", "source"}
        )
        assert declared == (
            ("domain", "llm-translation"),
            ("route", "messages"),
            ("provider", "anthropic"),
            ("provider", "bedrock"),
            ("model", "claude-haiku-4-5"),
            ("model", "claude-opus-4-7"),
            ("model", "claude-sonnet-4-5"),
            ("capability", "function_calling"),
            ("capability", "vision"),
            ("mode", "stream"),
        )


class TestBareStrIsACollectionError:
    def test_a_str_where_a_tuple_belongs_fails_collection_and_names_the_fix(self, tmp_path: Path) -> None:
        """`models=("gpt-5.5")` raises where the decorator runs, which is import, so
        pytest stops at collection and points at the file. Nothing is run and no
        one-letter `model` properties are ever shipped."""
        write_suite(tmp_path, {"test_bare_str.py": BARE_STR_SUITE})
        child: Final = run_child_pytest(tmp_path)
        assert child.returncode == pytest.ExitCode.INTERRUPTED, child.stdout
        assert "Subject.models must be a tuple, got str: 'gpt-5.5'" in child.stdout
        assert "models=(x,), not models=(x)" in child.stdout
