"""Tests for scripts/check_test_quality.py.

Every rule is exercised on a snippet that violates it and on one that does not, so
dropping a rule, widening it, or inverting the suppression check makes a test fail.
The helper-resolution cases are the regression for the false positives the rule
produced against tests/e2e, where the assertions live in a shared helper rather than
in the test body.
"""

import importlib.util
import sys
from pathlib import Path

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
    assert _codes(tmp_path, source) == ["TQ002"]


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
    assert _codes(tmp_path, source) == ["TQ002"]


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
    assert _codes(tmp_path, source) == ["TQ002"]


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
    assert _codes(tmp_path, source) == []


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
    assert _codes(tmp_path, source) == ["TQ001"]


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
    # `key = os.getenv(...)` then `if not key: skip` is the shape most of these use,
    # so a rule that only reads the `if` test sees almost none of them.
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
