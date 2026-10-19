from types import ModuleType
from typing import Final

import pytest
from _pytest.pytester import Pytester

from tests._support.native_suite import NativeSuite

pytest_plugins = ("pytester",)

NATIVE: Final = "@pytest.mark.requires_rust_extension\ndef test_native():\n    assert 1 + 1 == 2\n"


def available() -> ModuleType:
    return ModuleType("native")


def unavailable() -> ModuleType:
    raise ModuleNotFoundError("native")


@pytest.mark.parametrize(
    ("source", "arguments", "expected"),
    (
        (NATIVE, (), pytest.ExitCode.OK),
        (NATIVE + "\ndef test_dispatch():\n    assert True\n", (), pytest.ExitCode.OK),
        (NATIVE, ("-k", "absent"), pytest.ExitCode.NO_TESTS_COLLECTED),
        (NATIVE, ("--collect-only",), pytest.ExitCode.OK),
        ("@pytest.mark.skip(reason='unavailable')\n" + NATIVE, (), pytest.ExitCode.TESTS_FAILED),
        (
            "@pytest.mark.skip(reason='unavailable')\n" + NATIVE + "\ndef test_dispatch():\n    assert True\n",
            (),
            pytest.ExitCode.TESTS_FAILED,
        ),
        ("def test_dispatch():\n    assert True\n", (), pytest.ExitCode.TESTS_FAILED),
        (NATIVE.replace("1 + 1 == 2", "1 + 1 == 3"), (), pytest.ExitCode.TESTS_FAILED),
    ),
    ids=("native", "unmarked-included", "empty", "collect-only", "skipped", "native-skipped", "no-native", "failed"),
)
def test_native_suite_requires_execution(
    pytester: Pytester, source: str, arguments: tuple[str, ...], expected: pytest.ExitCode
) -> None:
    pytester.path.joinpath("test_suite.py").write_text("import pytest\n" + source)
    result: Final = pytester.inline_run(  # pyright: ignore[reportUnknownMemberType]  # pytester leaves plugins untyped
        *arguments, plugins=(NativeSuite(pytester.path, available),)
    )

    assert result.ret == expected
    if "def test_dispatch" in source:
        assert any(
            report.nodeid.endswith("::test_dispatch") for report in result.getreports("pytest_runtest_logreport")
        )


def test_missing_extension_fails_before_test_collection(pytester: Pytester) -> None:
    pytester.path.joinpath("test_suite.py").write_text("raise AssertionError('must not collect without native')")
    result: Final = pytester.inline_run(  # pyright: ignore[reportUnknownMemberType]  # pytester leaves plugins untyped
        plugins=(NativeSuite(pytester.path, unavailable),)
    )

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert not result.getreports("pytest_collectreport")


@pytest.mark.parametrize(
    ("source", "inside_root", "expected"),
    (
        (NATIVE, True, pytest.ExitCode.OK),
        ("@pytest.mark.skip(reason='unavailable')\n" + NATIVE, True, pytest.ExitCode.TESTS_FAILED),
        ("def test_dispatch():\n    assert True\n", True, pytest.ExitCode.TESTS_FAILED),
        (NATIVE, False, pytest.ExitCode.TESTS_FAILED),
    ),
    ids=("native", "native-skipped", "unmarked", "outside-root"),
)
def test_controller_requires_a_passing_native_worker_report(
    pytester: Pytester, source: str, inside_root: bool, expected: pytest.ExitCode
) -> None:
    pytester.path.joinpath("test_suite.py").write_text("import pytest\n" + source)
    root: Final = pytester.path if inside_root else pytester.path / "native"
    result: Final = pytester.inline_run(  # pyright: ignore[reportUnknownMemberType]  # pytester leaves plugins untyped
        plugins=(NativeSuite(root, available),)
    )
    controller: Final = NativeSuite(root, available)
    for report in result.getreports("pytest_runtest_logreport"):
        controller.pytest_runtest_logreport(report)
    session: Final = pytest.Session.from_config(pytester.parseconfig())
    session.exitstatus = pytest.ExitCode.OK

    controller.pytest_sessionfinish(session)

    assert session.exitstatus == expected
