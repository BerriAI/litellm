"""Tests for scripts/check_test_quality.py.

Every rule is exercised on a snippet that violates it and on one that does not, so
dropping a rule, widening it, or inverting the suppression check makes a test fail.
The helper-resolution cases are the regression for the false positives the rule
produced against tests/e2e, where the assertions live in a shared helper rather than
in the test body.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "scripts" / "check_test_quality.py"
_spec = importlib.util.spec_from_file_location("check_test_quality", _MODULE_PATH)
checker = importlib.util.module_from_spec(_spec)
# @dataclass(slots=True) rebuilds its class through sys.modules[__module__], so the
# module has to be registered before exec_module runs or Scope fails to construct.
sys.modules[_spec.name] = checker
_spec.loader.exec_module(checker)


def _codes(tmp_path, source):
    snippet = tmp_path / "test_snippet.py"
    snippet.write_text(source, encoding="utf-8")
    return [v.code for v in checker.check_file(snippet)]


def test_zero_assert_test_is_flagged(tmp_path):
    assert _codes(tmp_path, "def test_nothing():\n    compute()\n") == ["TQ001"]


def test_plain_assert_statement_clears_the_rule(tmp_path):
    assert _codes(tmp_path, "def test_value():\n    assert compute() == 3\n") == []


def test_pytest_raises_counts_as_an_assertion(tmp_path):
    source = "import pytest\n\n\ndef test_raises():\n    with pytest.raises(ValueError):\n        compute()\n"
    assert _codes(tmp_path, source) == []


def test_unittest_style_assertion_counts(tmp_path):
    source = "class TestThing:\n    def test_equal(self):\n        self.assertEqual(compute(), 3)\n"
    assert _codes(tmp_path, source) == []


def test_bare_assert_helper_call_counts(tmp_path):
    source = "def test_denied():\n    assert_auth_denied(call(), 'missing header')\n"
    assert _codes(tmp_path, source) == []


def test_assertion_inside_a_module_local_helper_clears_the_rule(tmp_path):
    source = (
        "def _drive_and_check(client):\n"
        "    assert client.status == 429\n"
        "\n"
        "\n"
        "def test_budget_blocks(client):\n"
        "    _drive_and_check(client)\n"
    )
    assert _codes(tmp_path, source) == []


def test_helper_chain_is_followed_transitively(tmp_path):
    source = (
        "def _inner(x):\n"
        "    assert x == 1\n"
        "\n"
        "\n"
        "def _outer(x):\n"
        "    _inner(x)\n"
        "\n"
        "\n"
        "def test_chain():\n"
        "    _outer(1)\n"
    )
    assert _codes(tmp_path, source) == []


def test_a_same_named_helper_in_another_class_does_not_clear_the_rule(tmp_path):
    source = (
        "class TestAsserting:\n"
        "    def _check(self):\n"
        "        assert compute() == 3\n"
        "\n"
        "    def test_ok(self):\n"
        "        self._check()\n"
        "\n"
        "\n"
        "class TestNotAsserting:\n"
        "    def _check(self):\n"
        "        compute()\n"
        "\n"
        "    def test_nothing(self):\n"
        "        self._check()\n"
    )
    assert _codes(tmp_path, source) == ["TQ001"]


def test_self_call_resolves_to_the_enclosing_class(tmp_path):
    source = (
        "class TestOne:\n"
        "    def _check(self):\n"
        "        assert compute() == 3\n"
        "\n"
        "    def test_ok(self):\n"
        "        self._check()\n"
    )
    assert _codes(tmp_path, source) == []


def test_a_method_named_like_a_module_helper_does_not_shadow_it(tmp_path):
    source = (
        "def _check():\n"
        "    assert compute() == 3\n"
        "\n"
        "\n"
        "class TestThing:\n"
        "    def _check(self):\n"
        "        compute()\n"
        "\n"
        "    def test_bare_name_uses_the_module_helper(self):\n"
        "        _check()\n"
        "\n"
        "    def test_self_uses_the_method(self):\n"
        "        self._check()\n"
    )
    assert _codes(tmp_path, source) == ["TQ001"]


def test_helper_without_assertions_does_not_clear_the_rule(tmp_path):
    source = (
        "def _just_calls(client):\n"
        "    client.go()\n"
        "\n"
        "\n"
        "def test_nothing_anywhere(client):\n"
        "    _just_calls(client)\n"
    )
    assert _codes(tmp_path, source) == ["TQ001"]


def test_mutually_recursive_helpers_terminate(tmp_path):
    source = (
        "def _a(x):\n"
        "    _b(x)\n"
        "\n"
        "\n"
        "def _b(x):\n"
        "    _a(x)\n"
        "\n"
        "\n"
        "def test_cycle():\n"
        "    _a(1)\n"
    )
    assert _codes(tmp_path, source) == ["TQ001"]


def test_non_test_function_is_not_collected(tmp_path):
    assert _codes(tmp_path, "def helper_without_asserts():\n    compute()\n") == []


def test_class_with_a_constructor_is_not_collected(tmp_path):
    source = (
        "class TestLegacy:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
        "\n"
        "    def test_nothing(self):\n"
        "        compute()\n"
    )
    assert _codes(tmp_path, source) == []


def test_mock_echo_is_flagged(tmp_path):
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "\n"
        "def test_echo():\n"
        "    with patch('litellm.completion') as mock_completion:\n"
        "        run()\n"
        "    mock_completion.assert_called_once()\n"
    )
    assert _codes(tmp_path, source) == ["TQ002", "TQ008"]


def test_call_args_inspection_is_mock_echo(tmp_path):
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "\n"
        "def test_echo():\n"
        "    with patch('litellm.completion') as mock_completion:\n"
        "        run()\n"
        "    assert mock_completion.call_args[1]['model'] == 'gpt-4o'\n"
    )
    assert _codes(tmp_path, source) == ["TQ002", "TQ008"]


def test_patch_decorator_counts_as_installing_a_patch(tmp_path):
    source = (
        "from unittest import mock\n"
        "\n"
        "\n"
        "@mock.patch('litellm.completion')\n"
        "def test_echo(mock_completion):\n"
        "    run()\n"
        "    mock_completion.assert_called_once()\n"
    )
    assert _codes(tmp_path, source) == ["TQ002", "TQ008"]


def test_patching_but_asserting_the_output_is_not_mock_echo(tmp_path):
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "\n"
        "def test_output():\n"
        "    with patch('litellm.completion') as mock_completion:\n"
        "        result = run()\n"
        "    mock_completion.assert_called_once()\n"
        "    assert result.choices[0].message.content == 'pong'\n"
    )
    assert _codes(tmp_path, source) == ["TQ008"]


def test_asserting_without_patching_is_not_mock_echo(tmp_path):
    source = "def test_plain():\n    m = build()\n    assert m.called\n"
    assert _codes(tmp_path, source) == []


def test_a_test_with_no_assertions_is_tq001_not_tq002(tmp_path):
    source = (
        "from unittest.mock import patch\n"
        "\n"
        "\n"
        "def test_nothing():\n"
        "    with patch('litellm.completion'):\n"
        "        run()\n"
    )
    assert _codes(tmp_path, source) == ["TQ001", "TQ008"]


def test_sys_path_insert_is_flagged(tmp_path):
    assert _codes(tmp_path, "import sys\n\nsys.path.insert(0, '..')\n") == ["TQ003"]


def test_sys_path_read_is_not_flagged(tmp_path):
    assert _codes(tmp_path, "import sys\n\nprint(sys.path)\n") == []


def test_raw_environ_write_is_flagged(tmp_path):
    assert _codes(tmp_path, "import os\n\nos.environ['KEY'] = 'v'\n") == ["TQ004"]


def test_bare_environ_write_is_flagged(tmp_path):
    assert _codes(tmp_path, "from os import environ\n\nenviron['KEY'] = 'v'\n") == ["TQ004"]


def test_environ_read_is_not_flagged(tmp_path):
    assert _codes(tmp_path, "import os\n\nvalue = os.environ.get('KEY')\n") == []


def test_monkeypatch_setenv_is_not_flagged(tmp_path):
    source = "def test_env(monkeypatch):\n    monkeypatch.setenv('KEY', 'v')\n    assert read() == 'v'\n"
    assert _codes(tmp_path, source) == []


def test_litellm_global_write_is_flagged(tmp_path):
    assert _codes(tmp_path, "import litellm\n\nlitellm.drop_params = True\n") == ["TQ005"]


def test_litellm_augmented_global_write_is_flagged(tmp_path):
    assert _codes(tmp_path, "import litellm\n\nlitellm.num_retries += 1\n") == ["TQ005"]


def test_litellm_attribute_read_is_not_flagged(tmp_path):
    assert _codes(tmp_path, "import litellm\n\nvalue = litellm.drop_params\n") == []


def test_unrelated_attribute_write_is_not_flagged(tmp_path):
    assert _codes(tmp_path, "config.drop_params = True\n") == []


def test_suppression_with_a_reason_clears_the_violation(tmp_path):
    source = "import sys\n\nsys.path.insert(0, '..')  # test-quality-ok: vendored path is required here\n"
    assert _codes(tmp_path, source) == []


def test_suppression_without_a_reason_does_not_suppress(tmp_path):
    assert _codes(tmp_path, "import sys\n\nsys.path.insert(0, '..')  # test-quality-ok:\n") == ["TQ003"]


def test_suppression_on_another_line_does_not_suppress(tmp_path):
    source = "import sys  # test-quality-ok: this reason sits on the wrong line\n\nsys.path.insert(0, '..')\n"
    assert _codes(tmp_path, source) == ["TQ003"]


def test_unparseable_source_degrades_to_tq000(tmp_path):
    assert _codes(tmp_path, "def test_broken(:\n    pass\n") == ["TQ000"]


def test_every_violation_renders_as_path_line_code_message():
    rendered = checker.Violation(Path("tests/test_x.py"), 7, "TQ001", "nothing asserted").render()
    assert rendered == "tests/test_x.py:7: TQ001 nothing asserted"


_DIRECT_GATE = """import os
import pytest


def test_live_call():
    if not os.getenv("ACME_API_KEY"):
        pytest.skip("no key")
    assert call() == "ok"
"""

_BOUND_GATE = """import os
import pytest


def test_live_call():
    api_key = os.getenv("ACME_API_KEY")
    if not api_key:
        pytest.skip("no key")
    assert call() == "ok"
"""

_MEMBERSHIP_GATE = """import os
import pytest


def test_live_call():
    if "ACME_API_KEY" not in os.environ:
        pytest.skip("no key")
    assert call() == "ok"
"""


def test_a_skip_gated_on_a_missing_credential_is_flagged(tmp_path):
    assert _codes(tmp_path, _DIRECT_GATE) == ["TQ006"]


def test_the_gate_is_followed_through_the_local_it_was_bound_to(tmp_path):
    assert _codes(tmp_path, _BOUND_GATE) == ["TQ006"]


def test_a_membership_test_against_os_environ_gates_just_the_same(tmp_path):
    assert _codes(tmp_path, _MEMBERSHIP_GATE) == ["TQ006"]


def test_a_skip_gated_on_something_that_is_not_a_credential_is_left_alone(tmp_path):
    source = _DIRECT_GATE.replace("ACME_API_KEY", "CI_RUNNER_OS")
    assert _codes(tmp_path, source) == []


def test_reading_a_credential_without_skipping_on_it_is_left_alone(tmp_path):
    source = 'import os\n\n\ndef test_live_call():\n    assert call(os.getenv("ACME_API_KEY")) == "ok"\n'
    assert _codes(tmp_path, source) == []


def test_a_skip_outside_the_credential_branch_is_left_alone(tmp_path):
    source = (
        "import os\n"
        "import pytest\n"
        "\n"
        "\n"
        "def test_live_call():\n"
        '    if not os.getenv("ACME_API_KEY"):\n'
        "        configure()\n"
        '    pytest.skip("unconditional")\n'
        '    assert call() == "ok"\n'
    )
    assert _codes(tmp_path, source) == []


def test_the_credential_skip_is_suppressible_like_every_other_rule(tmp_path):
    source = _DIRECT_GATE.replace(
        'pytest.skip("no key")',
        'pytest.skip("no key")  # test-quality-ok: the live suite owns this one',
    )
    assert _codes(tmp_path, source) == []


def test_a_skip_taken_when_the_credential_is_present_is_left_alone(tmp_path):
    source = _DIRECT_GATE.replace('if not os.getenv("ACME_API_KEY")', 'if os.getenv("ACME_API_KEY")')
    assert _codes(tmp_path, source) == []


def test_a_none_comparison_reads_as_absence(tmp_path):
    source = _BOUND_GATE.replace("if not api_key:", "if api_key is None:")
    assert _codes(tmp_path, source) == ["TQ006"]


def test_a_membership_test_without_the_negation_is_left_alone(tmp_path):
    source = _MEMBERSHIP_GATE.replace('"ACME_API_KEY" not in os.environ', '"ACME_API_KEY" in os.environ')
    assert _codes(tmp_path, source) == []


_SNAPSHOT_CONFTEST = """import litellm
import pytest


@pytest.fixture(autouse=True)
def restore_globals():
    original_state = {}
    original_state["drop_params"] = litellm.drop_params
    for attr in ("api_base", "num_retries"):
        original_state[attr] = getattr(litellm, attr)
    yield
    for attr, value in original_state.items():
        setattr(litellm, attr, value)
"""


def _conftest_codes(tmp_path, source, name="conftest.py"):
    snippet = tmp_path / name
    snippet.write_text(source, encoding="utf-8")
    return [v.code for v in checker.check_file(snippet)]


def test_every_snapshotted_global_is_counted_once(tmp_path):
    assert _conftest_codes(tmp_path, _SNAPSHOT_CONFTEST) == ["TQ007", "TQ007", "TQ007"]


def test_the_names_come_from_the_loop_tuple_as_well_as_the_direct_keys(tmp_path):
    snippet = tmp_path / "conftest.py"
    snippet.write_text(_SNAPSHOT_CONFTEST, encoding="utf-8")
    reported = [v.message.split("`")[1] for v in checker.check_file(snippet)]
    assert sorted(reported) == ["litellm.api_base", "litellm.drop_params", "litellm.num_retries"]


def test_the_same_global_saved_twice_counts_once(tmp_path):
    source = _SNAPSHOT_CONFTEST.replace(
        '("api_base", "num_retries")', '("api_base", "num_retries", "drop_params")'
    )
    assert _conftest_codes(tmp_path, source) == ["TQ007", "TQ007", "TQ007"]


def test_the_rule_only_looks_at_conftest_files(tmp_path):
    assert _conftest_codes(tmp_path, _SNAPSHOT_CONFTEST, name="test_snapshot.py") == []


def test_a_conftest_that_snapshots_nothing_is_clean(tmp_path):
    source = "import pytest\n\n\n@pytest.fixture\ndef client():\n    return object()\n"
    assert _conftest_codes(tmp_path, source) == []


def test_a_snapshot_entry_is_suppressible_with_a_reason(tmp_path):
    source = _SNAPSHOT_CONFTEST.replace(
        'original_state["drop_params"] = litellm.drop_params',
        'original_state["drop_params"] = litellm.drop_params  # test-quality-ok: owned by the SDK config surface',
    )
    assert _conftest_codes(tmp_path, source) == ["TQ007", "TQ007"]


_NAMED_MAPPING_CONFTEST = """import litellm
import pytest

_SCALAR_DEFAULTS = {
    "num_retries": None,
    "set_verbose": False,
}
_EXTRA_ATTRS = ("api_base", "drop_params")


@pytest.fixture(autouse=True)
def restore_globals():
    original_state = {}
    for attr in _SCALAR_DEFAULTS:
        original_state[attr] = getattr(litellm, attr)
    for attr in _EXTRA_ATTRS:
        original_state[attr] = getattr(litellm, attr)
    yield
    for attr, value in original_state.items():
        setattr(litellm, attr, value)
"""


def test_a_save_loop_over_a_module_level_dict_counts_its_keys(tmp_path):
    # The two largest inventories in the repo name their list instead of spelling it
    # out, so a rule that only reads literal iterables sees neither.
    reported = [v.message.split("`")[1] for v in checker.check_file(_written(tmp_path, _NAMED_MAPPING_CONFTEST))]
    assert sorted(reported) == [
        "litellm.api_base",
        "litellm.drop_params",
        "litellm.num_retries",
        "litellm.set_verbose",
    ]


def test_a_named_iterable_that_is_not_a_module_constant_is_skipped_quietly(tmp_path):
    source = _NAMED_MAPPING_CONFTEST.replace("for attr in _EXTRA_ATTRS:", "for attr in dir(litellm):")
    reported = [v.message.split("`")[1] for v in checker.check_file(_written(tmp_path, source))]
    assert sorted(reported) == ["litellm.num_retries", "litellm.set_verbose"]


def _written(tmp_path, source, name="conftest.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


_HELPER_DICT_CONFTEST = """import litellm
import pytest

_CALLBACK_ATTRS = ("callbacks", "success_callback")


def _copy_litellm_state():
    state = {}
    for attr in _CALLBACK_ATTRS:
        if hasattr(litellm, attr):
            value = getattr(litellm, attr)
            state[attr] = value.copy() if isinstance(value, list) else value
    return state


@pytest.fixture(autouse=True)
def restore_globals():
    saved = _copy_litellm_state()
    yield
    for attr, value in saved.items():
        setattr(litellm, attr, value)
"""


def test_a_snapshot_built_in_a_helper_under_any_dict_name_is_counted(tmp_path):
    # Two conftests build their inventory inside a helper and call the dict `state`,
    # so a rule keyed on blessed dict names sees neither.
    reported = [v.message.split("`")[1] for v in checker.check_file(_written(tmp_path, _HELPER_DICT_CONFTEST))]
    assert sorted(reported) == ["litellm.callbacks", "litellm.success_callback"]


def test_the_read_may_sit_a_statement_above_the_store(tmp_path):
    # `val = getattr(litellm, attr)` then `state[attr] = val.copy()` is the common
    # shape; requiring the store itself to read litellm loses every one of them.
    source = _HELPER_DICT_CONFTEST.replace(
        "            state[attr] = value.copy() if isinstance(value, list) else value",
        "            state[attr] = list(value)",
    )
    reported = [v.message.split("`")[1] for v in checker.check_file(_written(tmp_path, source))]
    assert sorted(reported) == ["litellm.callbacks", "litellm.success_callback"]


def test_a_loop_storing_under_a_key_that_is_not_the_loop_variable_is_not_an_inventory(tmp_path):
    source = _HELPER_DICT_CONFTEST.replace("state[attr] =", 'state["fixed"] =')
    assert [v.code for v in checker.check_file(_written(tmp_path, source))] == []


def test_patching_an_sdk_function_by_string_is_flagged(tmp_path):
    source = 'from unittest.mock import patch\n\n\n@patch("litellm.completion")\ndef test_x(m):\n    assert m\n'
    assert "TQ008" in _codes(tmp_path, source)


def test_patching_a_deep_sdk_path_is_flagged(tmp_path):
    source = (
        "from unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch("litellm.llms.openai.chat.handler.OpenAIChatCompletion.completion"):\n'
        "        assert True\n"
    )
    assert "TQ008" in _codes(tmp_path, source)


def test_patch_object_rooted_at_the_sdk_is_flagged(tmp_path):
    source = (
        "import litellm\nfrom unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch.object(litellm, "api_key", "x"):\n'
        "        assert True\n"
    )
    assert "TQ008" in _codes(tmp_path, source)


def test_patch_object_on_a_from_imported_sdk_module_is_flagged(tmp_path):
    source = (
        "from litellm.llms.openai.chat import handler\nfrom unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch.object(handler.OpenAIChatCompletion, "completion"):\n'
        "        assert True\n"
    )
    assert "TQ008" in _codes(tmp_path, source)


def test_patch_object_on_an_aliased_sdk_module_is_flagged(tmp_path):
    source = (
        "import litellm.llms.openai.chat.handler as oai\nfrom unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch.object(oai.OpenAIChatCompletion, "completion"):\n'
        "        assert True\n"
    )
    assert "TQ008" in _codes(tmp_path, source)


def test_patch_object_on_a_renamed_sdk_symbol_is_flagged(tmp_path):
    source = (
        "from litellm.utils import get_llm_provider as glp\nfrom unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch.object(glp, "__wrapped__"):\n'
        "        assert True\n"
    )
    assert "TQ008" in _codes(tmp_path, source)


def test_the_reported_target_is_the_resolved_sdk_path(tmp_path):
    source = (
        "from litellm.llms.openai.chat import handler\nfrom unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch.object(handler.OpenAIChatCompletion, "completion"):\n'
        "        assert True\n"
    )
    reported = [v.message for v in checker.check_file(_written(tmp_path, source)) if v.code == "TQ008"]
    assert reported
    assert "litellm.llms.openai.chat.handler.OpenAIChatCompletion" in reported[0]


def test_patch_object_on_a_from_imported_third_party_is_not_flagged(tmp_path):
    source = (
        "from openai import OpenAI\nfrom unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch.object(OpenAI, "chat"):\n'
        "        assert True\n"
    )
    assert "TQ008" not in _codes(tmp_path, source)


def test_a_local_name_with_no_sdk_import_behind_it_is_not_flagged(tmp_path):
    source = (
        "from unittest.mock import patch\n\n\n"
        "def test_x(handler):\n"
        '    with patch.object(handler, "completion"):\n'
        "        assert True\n"
    )
    assert "TQ008" not in _codes(tmp_path, source)


def test_mocking_a_third_party_client_is_not_flagged(tmp_path):
    source = (
        "from unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch("openai.OpenAI.chat"):\n'
        "        assert True\n"
    )
    assert "TQ008" not in _codes(tmp_path, source)


def test_mocking_the_http_transport_is_not_flagged(tmp_path):
    source = (
        "from unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch("httpx.AsyncClient.send"):\n'
        "        assert True\n"
    )
    assert "TQ008" not in _codes(tmp_path, source)


def test_a_name_merely_starting_with_litellm_is_not_the_sdk(tmp_path):
    source = (
        "from unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch("litellm_enterprise.thing.go"):\n'
        "        assert True\n"
    )
    assert "TQ008" not in _codes(tmp_path, source)


def test_an_sdk_patch_can_be_suppressed(tmp_path):
    source = (
        "from unittest.mock import patch\n\n\n"
        "def test_x():\n"
        '    with patch("litellm.completion"):  # test-quality-ok: pinning the router seam\n'
        "        assert True\n"
    )
    assert "TQ008" not in _codes(tmp_path, source)


_FANS_OUT = checker._worker_count(checker.PARALLEL_MIN_PATHS) > 1
_SERIAL_ONLY = "one usable core, so scan_paths stays serial and there is no fan-out to compare"


def _corpus(tmp_path: Path, count: int) -> tuple[Path, ...]:
    for index in range(count):
        (tmp_path / f"test_gen_{index}.py").write_text(
            f"def test_flagged_{index}():\n    compute()\n\n\ndef test_clean_{index}():\n    assert compute() == {index}\n",
            encoding="utf-8",
        )
    return tuple(sorted(tmp_path.rglob("*.py")))


def _run_checker(target: Path) -> list[str]:
    completed = subprocess.run(
        [sys.executable, str(_MODULE_PATH), str(target)],
        capture_output=True, text=True, timeout=300,
    )
    return completed.stdout.splitlines()


def test_worker_count_stays_serial_below_the_threshold():
    assert checker._worker_count(checker.PARALLEL_MIN_PATHS - 1) == 1


def test_worker_count_fans_out_at_the_threshold():
    assert checker._worker_count(checker.PARALLEL_MIN_PATHS) == max(
        1, min(os.cpu_count() or 1, checker.MAX_WORKERS)
    )


def test_worker_count_never_exceeds_the_cap():
    assert checker._worker_count(100_000) <= checker.MAX_WORKERS


def test_scan_paths_below_the_threshold_returns_every_violation(tmp_path):
    paths = _corpus(tmp_path, 3)
    assert checker._worker_count(len(paths)) == 1
    assert [v.code for v in checker.scan_paths(paths)] == ["TQ001"] * 3


@pytest.mark.skipif(not _FANS_OUT, reason=_SERIAL_ONLY)
def test_a_fanned_out_run_reports_exactly_what_a_serial_run_reports(tmp_path):
    paths = _corpus(tmp_path, checker.PARALLEL_MIN_PATHS + 5)
    serial = [v.render() for v in sorted(v for path in paths for v in checker.check_file(path))]
    assert serial, "corpus must produce violations or the comparison proves nothing"
    assert _run_checker(tmp_path) == serial


@pytest.mark.skipif(not _FANS_OUT, reason=_SERIAL_ONLY)
def test_a_fanned_out_run_reports_each_generated_file_exactly_once(tmp_path):
    paths = _corpus(tmp_path, checker.PARALLEL_MIN_PATHS + 5)
    reported = _run_checker(tmp_path)
    assert len(reported) == len(paths)
    assert len({line.split(":")[0] for line in reported}) == len(paths)
    assert all(" TQ001 " in line for line in reported)
