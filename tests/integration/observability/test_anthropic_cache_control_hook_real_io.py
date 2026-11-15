import json
import os
import subprocess
import sys
import textwrap
from typing import Optional, Tuple

import pytest

import litellm


@pytest.fixture(autouse=True)
def _no_openai_api_base_override(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)


class TestAnthropicPromptCachingEnvVars:
    """Both settings are read from the environment at import, so an admin can enable
    auto-caching without a config file. Each case re-imports litellm in a subprocess
    so the env is read fresh without contaminating this process's module graph.
    """

    @staticmethod
    def _import_litellm_with_env(env_override: dict) -> Tuple[bool, Optional[str]]:
        env = os.environ.copy()
        env.pop("LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING", None)
        env.pop("LITELLM_ANTHROPIC_PROMPT_CACHING_TTL", None)
        env.update(env_override)
        script = textwrap.dedent(
            """
            import json, litellm
            print(json.dumps([litellm.enable_anthropic_prompt_caching, litellm.anthropic_prompt_caching_ttl]))
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300)
        assert result.returncode == 0, result.stderr
        enabled, ttl = json.loads(result.stdout.strip().splitlines()[-1])
        return enabled, ttl

    def test_unset_env_leaves_auto_caching_off(self):
        assert self._import_litellm_with_env({}) == (False, None)

    @pytest.mark.parametrize("value", ["true", "True", "TRUE"])
    def test_env_enables_auto_caching_case_insensitively(self, value):
        enabled, _ = self._import_litellm_with_env({"LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING": value})
        assert enabled is True

    @pytest.mark.parametrize("value", ["false", "0", "yes", ""])
    def test_env_only_enables_on_true(self, value):
        enabled, _ = self._import_litellm_with_env({"LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING": value})
        assert enabled is False

    @pytest.mark.parametrize("value", ["5m", "1h"])
    def test_ttl_env_is_applied(self, value):
        _, ttl = self._import_litellm_with_env({"LITELLM_ANTHROPIC_PROMPT_CACHING_TTL": value})
        assert ttl == value

    @pytest.mark.parametrize("value", ["10m", "1H", "3600", "ephemeral"])
    def test_unsupported_ttl_env_falls_back_to_provider_default(self, value):
        """An unparseable TTL must fall back to Anthropic's 5m default, never reach the provider verbatim."""
        _, ttl = self._import_litellm_with_env({"LITELLM_ANTHROPIC_PROMPT_CACHING_TTL": value})
        assert ttl is None
