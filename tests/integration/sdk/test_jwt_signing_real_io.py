import subprocess
import sys
import textwrap
from typing import Final

from litellm.llms.base_llm.auth.jwt_signing import MISSING_SIGNING_DEPENDENCIES_MESSAGE


class TestBaseSdkImport:
    """A base ``pip install litellm`` has neither PyJWT nor cryptography (both are proxy extras),
    and ``litellm/__init__`` reaches this module through the Anthropic provider, so a
    module-level import of either would break ``import litellm`` for every base SDK user."""

    def test_module_imports_with_pyjwt_and_cryptography_absent(self):
        script: Final = textwrap.dedent(
            """
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in {"jwt", "cryptography"}:
                        raise ModuleNotFoundError(f"No module named {name!r}")

            sys.meta_path.insert(0, Blocker())
            import litellm
            from litellm.llms.base_llm.auth.jwt_signing import jwks_document_json
            try:
                jwks_document_json("not a key")
            except ImportError as e:
                print(e)
            """
        )
        result: Final = subprocess.run(
            [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip() == MISSING_SIGNING_DEPENDENCIES_MESSAGE
