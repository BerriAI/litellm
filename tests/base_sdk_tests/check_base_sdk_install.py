"""Smoke-check that a base ``pip install litellm`` (no extras) is importable and usable.

Run against a virtualenv that has the built wheel installed with no extras, using
that venv's own interpreter and nothing else. Deliberately stdlib-only: pytest would
pull ``packaging``, ``pluggy`` and ``iniconfig`` into the environment and could mask
the very class of undeclared-dependency bug this guards against.
"""

import argparse
import importlib.util
import sys
import traceback
from collections.abc import Callable

EXTRAS_ONLY_MODULES = ("fastapi", "uvicorn", "keyring", "mcp", "mcp_types", "httpx2", "httpcore2")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_environment_is_base_only() -> str:
    present = tuple(name for name in EXTRAS_ONLY_MODULES if importlib.util.find_spec(name) is not None)
    _require(
        not present,
        f"{', '.join(present)} installed, so this environment is not base-only and the run proves nothing",
    )
    return f"no extras-only packages present ({', '.join(EXTRAS_ONLY_MODULES)})"


def check_import() -> str:
    from importlib.metadata import distributions as installed_distributions
    from importlib.metadata import version

    import litellm
    from litellm.llms.brave.search.transformation import BraveSearchConfig

    _require(callable(BraveSearchConfig), "Brave search configuration unavailable")
    _require(bool(litellm.__file__), "litellm has no __file__")
    from litellm._version import version as sdk_version

    distributions = tuple(
        distribution.metadata["Name"]
        for distribution in installed_distributions()
        if distribution.metadata["Name"] in ("litellm", "litellm-core")
    )
    _require(len(distributions) == 1, f"expected one SDK distribution, found {distributions}")
    distribution = distributions[0]
    _require(sdk_version == version(distribution), "SDK version does not match installed metadata")
    _require("litellm.proxy.proxy_cli" not in sys.modules, "SDK import loaded the proxy CLI")
    return f"imported {distribution} {sdk_version}"


def check_completion() -> str:
    import litellm

    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "ping"}],
        mock_response="pong",
    )
    content = response.choices[0].message.content
    _require(content == "pong", f"mock completion returned {content!r}")
    return "mock completion round-trips"


def check_mcp_install_guidance() -> str:
    try:
        import litellm.experimental_mcp_client
    except ImportError as error:
        _require("pip install 'litellm[mcp]'" in str(error), f"missing MCP installation guidance: {error}")
        _require(isinstance(error.__cause__, ModuleNotFoundError), "original missing-dependency cause was lost")
        _require(error.__cause__.name == "mcp", f"unexpected missing dependency: {error.__cause__}")
        return "optional MCP client explains how to install litellm[mcp]"
    raise AssertionError("MCP client imported without the MCP extra")


def check_embedding() -> str:
    import litellm

    response = litellm.embedding(
        model="text-embedding-3-small",
        input=["ping"],
        mock_response=[[0.1, 0.2]],
    )
    _require(len(response.data) == 1, f"mock embedding returned {len(response.data)} rows")
    return "mock embedding round-trips"


def check_bundled_model_metadata() -> str:
    import litellm

    max_input_tokens = litellm.get_model_info("gpt-4o")["max_input_tokens"]
    _require(
        isinstance(max_input_tokens, int) and max_input_tokens > 0,
        f"get_model_info returned max_input_tokens={max_input_tokens!r}",
    )
    prompt_cost, completion_cost = litellm.cost_per_token(model="gpt-4o", prompt_tokens=1000, completion_tokens=1000)
    _require(
        prompt_cost > 0 and completion_cost > 0,
        f"cost_per_token returned ({prompt_cost}, {completion_cost})",
    )
    return f"bundled pricing metadata readable (gpt-4o max_input_tokens={max_input_tokens})"


def check_token_counter() -> str:
    import litellm

    count = litellm.token_counter(model="gpt-4o", text="hello world")
    _require(count > 0, f"token_counter returned {count!r}")
    return f"token_counter returned {count}"


def check_tokenizer_dependencies() -> str:
    import litellm
    from litellm.rust_bridge import tokenizer
    from litellm.litellm_core_utils.tokenizer import HuggingFace, HuggingFaceTokenizer, Tokenizer
    from litellm.utils import claude_json_str

    _require(isinstance(litellm.encoding, Tokenizer), "runtime alias rejects the default encoding")
    native = tokenizer.native_anthropic()
    if native is not None:
        _require(bool(native.encode("hello")), "native tokenizer returned no tokens")
        _require(isinstance(HuggingFaceTokenizer(native), HuggingFace), "runtime alias rejects native tokenizers")
    if importlib.util.find_spec("tokenizers") is not None:
        python_tokenizer = tokenizer._python_tokenizer().from_str(claude_json_str)
        _require(bool(python_tokenizer.encode("hello").ids), "Python tokenizer returned no tokens")
        _require(isinstance(python_tokenizer, HuggingFace), "runtime alias rejects Python tokenizers")
        return "installed Python tokenizer and available native tokenizer work"
    try:
        tokenizer._python_tokenizer()
    except ImportError as error:
        _require("pip install tokenizers" in str(error), f"missing tokenizer guidance: {error}")
    else:
        raise AssertionError("Python tokenizer loaded without tokenizers")
    return "native tokenizer works; Python tokenizer reports its missing dependency"


def check_bedrock_credential_resolution() -> str:
    import os
    from unittest import mock

    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    if importlib.util.find_spec("boto3") is None:
        try:
            BaseAWSLLM()._sign_request(
                service_name="bedrock", headers={}, optional_params={"aws_region_name": "us-east-1"},
                request_data={}, api_base="https://bedrock-runtime.us-east-1.amazonaws.com", api_key="",
            )
        except ImportError as error:
            _require("pip install boto3" in str(error), f"missing installation guidance: {error}")
            return "AWS signing explains how to install boto3"
        raise AssertionError("AWS signing unexpectedly worked without boto3")
    non_aws_environ = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    with mock.patch.dict(os.environ, non_aws_environ, clear=True):
        credentials = BaseAWSLLM().get_credentials(
            aws_access_key_id="AKIA-fake-base-sdk-check",
            aws_secret_access_key="fake-secret",
            aws_region_name="us-east-1",
        )
    _require(
        credentials.access_key == "AKIA-fake-base-sdk-check",
        f"get_credentials returned access_key={credentials.access_key!r}",
    )
    return "bedrock credential resolution works (boto3 ships with the base SDK)"


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("environment is base-only", check_environment_is_base_only),
    ("import litellm", check_import),
    ("optional MCP installation guidance", check_mcp_install_guidance),
    ("chat completion", check_completion),
    ("embedding", check_embedding),
    ("bundled model metadata", check_bundled_model_metadata),
    ("token counter", check_token_counter),
    ("tokenizer dependencies", check_tokenizer_dependencies),
    ("bedrock credential resolution", check_bedrock_credential_resolution),
)


def _run(check: Callable[[], str]) -> tuple[bool, str]:
    try:
        return True, check()
    except Exception:
        return False, traceback.format_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("legacy", "core", "dependencies"), default="legacy")
    profile = parser.parse_args().profile
    for module in ("boto3", "botocore", "tokenizers", "huggingface_hub"):
        present = importlib.util.find_spec(module) is not None
        _require(present == (profile != "core"), f"{profile}: unexpected presence of {module}: {present}")
    _require(importlib.util.find_spec("jsonschema") is not None, "jsonschema must remain mandatory")
    print(f"{profile} SDK smoke check on {sys.executable}")
    for label, check in CHECKS:
        passed, detail = _run(check)
        if not passed:
            print(f"FAIL  {label}:\n{detail}")
            print(f"A base `pip install litellm` is broken at: {label}")
            print("Something needed at import or call time is missing from [project].dependencies")
            print("in pyproject.toml. Declaring it only in an extra is what causes this.")
            return 1
        print(f"PASS  {label}: {detail}")

    print(f"\nall {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
