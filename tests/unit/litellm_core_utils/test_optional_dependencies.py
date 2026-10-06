from pathlib import Path
from typing import Final

import pytest

from litellm._version import get_distribution_name
from litellm.litellm_core_utils.optional_dependencies import MissingOptionalDependencyError, require_optional_dependency


def test_available_dependency_is_usable() -> None:
    require_optional_dependency("json", "integrations", "JSON configuration")
    import json

    assert json.loads('{"enabled": true}') == {"enabled": True}


def test_missing_dependency_explains_capability_and_preserves_cause() -> None:
    with pytest.raises(MissingOptionalDependencyError, match=r'Prompt rendering requires missing_prompt_renderer.*litellm\[prompts\]') as caught:
        require_optional_dependency("missing_prompt_renderer", "prompts", "Prompt rendering")
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == "missing_prompt_renderer"


def test_broken_installed_dependency_is_not_misreported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module: Final = tmp_path / "broken_prompt_renderer.py"
    module.write_text("import missing_renderer_internal_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError) as caught:
        require_optional_dependency("broken_prompt_renderer", "prompts", "Prompt rendering")
    assert caught.value.name == "missing_renderer_internal_dependency"
    assert not isinstance(caught.value, MissingOptionalDependencyError)


@pytest.mark.parametrize(
    "mode, reload, broken_dotenv, succeeds",
    [("DEV", "False", False, True), ("PRODUCTION", "False", False, True), ("DEV", "True", False, False), ("DEV", "False", True, False)],
)
def test_core_startup_without_optional_packages(mode: str, reload: str, broken_dotenv: bool, succeeds: bool) -> None:
    import json
    import os
    import subprocess
    import sys

    import litellm

    source: Final = '''
import importlib.abc, json, os, sys
class MissingFeatures(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dotenv' and os.environ['TEST_BROKEN_DOTENV'] == 'True':
            raise ModuleNotFoundError('Broken dotenv installation', name='dotenv_internal_dependency')
        if fullname.split('.')[0] in ('dotenv', 'pydantic_settings', 'yaml', 'jinja2', 'dateutil', 'packaging'):
            raise ModuleNotFoundError('Optional feature is absent', name=fullname.split('.')[0])
sys.meta_path.insert(0, MissingFeatures())
import litellm
response = litellm.completion(model='openai/test', messages=[{'role':'user','content':'hello'}], mock_response='OK')
print(json.dumps([litellm.__file__, response.choices[0].message.content]))
'''
    environment: Final = {**os.environ, "LITELLM_MODE": mode, "LITELLM_DEV_ENV_HOT_RELOAD": reload, "TEST_BROKEN_DOTENV": str(broken_dotenv), "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    result: Final = subprocess.run([sys.executable, "-I", "-c", source], env=environment, capture_output=True, text=True, timeout=60)
    if succeeds:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout.strip().splitlines()[-1]) == [litellm.__file__, "OK"]
    elif broken_dotenv:
        assert result.returncode != 0
        assert "ModuleNotFoundError: Broken dotenv installation" in result.stderr
        assert f"{get_distribution_name()}[dotenv]" not in result.stderr
    else:
        assert result.returncode != 0
        assert f"{get_distribution_name()}[dotenv]" in result.stderr
        assert "ModuleNotFoundError: Optional feature is absent" in result.stderr
