"""Smoke-check that a base ``pip install litellm`` (no extras) is importable and usable.

Run against a virtualenv that has the built wheel installed with no extras, using
that venv's own interpreter and nothing else. Deliberately stdlib-only: pytest would
pull ``packaging``, ``pluggy`` and ``iniconfig`` into the environment and could mask
the very class of undeclared-dependency bug this guards against.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import traceback
import warnings
import tempfile
from collections.abc import Callable
from functools import partial
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path
from typing import Final
from unittest.mock import patch

from litellm._version import get_distribution_name

EXTRAS_ONLY_MODULES = ("fastapi", "uvicorn", "keyring", "mcp", "mcp_types", "httpx2", "httpcore2")
AWS_MODULES: Final = ("boto3", "botocore", "s3transfer", "jmespath")
TOKENIZER_MODULES: Final = ("tokenizers", "huggingface_hub", "hf_xet", "fsspec")


def check_optional_dependencies(profile: str) -> str:
    if profile == "core":
        for module in ("click", "filelock", "importlib_metadata", "zipp", "jsonschema", "referencing", "rpds", "pydantic_settings", "packaging", "yaml", "jinja2", "markupsafe", "dateutil", "six", "dotenv"):
            _require(importlib.util.find_spec(module) is None, f"core still installs {module}")
    for modules, expected in (
        (AWS_MODULES, profile in ("aws", "aws,tokenizers", "sdk-extras", "proxy")),
        (TOKENIZER_MODULES, profile in ("tokenizers", "aws,tokenizers", "sdk-extras", "proxy")),
    ):
        for name in modules[:2] if expected else modules:
            present: Final = importlib.util.find_spec(name) is not None
            _require(present == expected, f"{name}: installed={present}, expected={expected} for {profile}")
    return f"optional dependencies match {profile}"


def check_validation(profile: str) -> str:
    import litellm
    from litellm.litellm_core_utils.json_validation_rule import validate_schema

    expected: Final = profile in ("validation", "sdk-extras", "proxy")
    _require((importlib.util.find_spec("jsonschema") is not None) == expected, "unexpected validation dependencies")
    if not expected:
        try:
            validate_schema({"type": "object"}, "{}")
        except ImportError as error:
            _require(f"{get_distribution_name()}[validation]" in str(error), f"missing validation guidance: {error}")
            return "requested validation requires its extra"
        raise AssertionError("requested validation silently succeeded without its extra")
    validate_schema({"type": "object"}, "{}")
    for response in ("not json", "[]"):
        try:
            validate_schema({"type": "object"}, response)
        except litellm.JSONSchemaValidationError:
            continue
        raise AssertionError(f"invalid response passed validation: {response}")
    return "validation accepts valid responses and rejects invalid JSON or schema mismatches"


def check_aws_signed_requests() -> str:
    from botocore.credentials import Credentials

    from litellm.llms.aws_polly.text_to_speech.transformation import AWSPollyTextToSpeechConfig
    from litellm.llms.sagemaker.chat.handler import SagemakerChatHandler
    from litellm.llms.sagemaker.completion.handler import SagemakerLLM

    credentials: Final = Credentials("test-key", "test-secret", "test-session")
    data: Final = {"inputs": "hello"}
    prepared: Final = (
        SagemakerChatHandler()._prepare_request(credentials, "test-endpoint", data, {}, "us-east-1"),
        SagemakerLLM()._prepare_request(credentials, "test-endpoint", data, [], {}, {}, "us-east-1"),
    )
    for request in prepared:
        _require(request.headers["Authorization"].startswith("AWS4-HMAC-SHA256"), "SageMaker request was not signed")
        _require(json.loads(request.body) == data, "SageMaker signing changed the body")
    headers, body = AWSPollyTextToSpeechConfig()._sign_polly_request(
        {"Text": "hello", "VoiceId": "Joanna"},
        "https://polly.us-east-1.amazonaws.com/v1/speech",
        {"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret", "aws_region_name": "us-east-1"},
    )
    _require(headers["Authorization"].startswith("AWS4-HMAC-SHA256"), "Polly request was not signed")
    _require(json.loads(body)["Text"] == "hello", "Polly signing changed the body")
    return "installed AWS extra signs SageMaker and Polly requests"


def check_aws_feature_guidance() -> str:
    if importlib.util.find_spec("boto3") is not None:
        return check_aws_signed_requests()
    import litellm
    from litellm.caching.s3_cache import S3Cache
    from litellm.integrations.dynamodb import DyanmoDBLogger
    from litellm.integrations.s3 import S3Logger
    from litellm.integrations.s3_v2 import S3Logger as S3V2Logger
    from litellm.integrations.sqs import SQSLogger
    from litellm.litellm_core_utils.litellm_logging import _init_custom_logger_compatible_class
    from litellm.llms.aws_polly.text_to_speech.transformation import AWSPollyTextToSpeechConfig
    from litellm.llms.sagemaker.chat.handler import SagemakerChatHandler
    from litellm.llms.sagemaker.completion.handler import SagemakerLLM
    from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2

    cases: Final = (
        ("S3 logger", S3Logger),
        ("DynamoDB logger", DyanmoDBLogger),
        ("S3 cache", partial(S3Cache, s3_bucket_name="test-bucket")),
        ("Secrets Manager", partial(AWSSecretsManagerV2()._prepare_request, "GetSecretValue", "test-secret")),
        ("S3 v2 constructor", S3V2Logger),
        ("SQS constructor", SQSLogger),
        ("S3 v2 registration", partial(_init_custom_logger_compatible_class, "s3_v2", None, None)),
        ("SQS registration", partial(_init_custom_logger_compatible_class, "aws_sqs", None, None)),
        ("SageMaker chat", partial(SagemakerChatHandler()._load_credentials, {})),
        ("SageMaker completion", partial(SagemakerLLM()._load_credentials, {})),
        ("Polly signing", partial(AWSPollyTextToSpeechConfig()._sign_polly_request, {}, "https://example.com", {})),
    )
    for label, action in cases:
        try:
            action()
        except ImportError as error:
            _require(f"{get_distribution_name()}[aws]" in str(error), f"{label}: missing guidance: {error}")
        else:
            raise AssertionError(f"{label}: silently accepted missing AWS dependencies")
    for callback in ("s3_v2", "aws_sqs"):
        with patch.object(litellm, "success_callback", [callback]):
            try:
                litellm.completion(model="openai/test", messages=[{"role": "user", "content": "test"}], mock_response="ok")
            except Exception as error:
                _require(f"{get_distribution_name()}[aws]" in str(error), f"{callback}: missing public guidance: {error}")
            else:
                raise AssertionError(f"{callback}: completion silently skipped logging")
    common: Final = {"api_key": "", "aws_access_key_id": "test", "aws_secret_access_key": "test", "aws_region_name": "us-east-1", "num_retries": 0}
    messages: Final = [{"role": "user", "content": "hello"}]
    provider_calls: Final = (
        partial(litellm.completion, model="bedrock/anthropic.claude-sonnet-5-5", messages=messages, **common),
        partial(litellm.completion, model="bedrock/amazon.titan-text-express-v1", messages=messages, **common),
        partial(litellm.embedding, model="bedrock/amazon.titan-embed-text-v2:0", input=["hello"], **common),
        partial(litellm.image_generation, model="bedrock/amazon.nova-canvas-v1:0", prompt="tree", **common),
        partial(litellm.rerank, model="bedrock/cohere.rerank-v3-5:0", query="hello", documents=["hello"], **common),
        partial(litellm.completion, model="sagemaker/test-endpoint", messages=messages, **common),
        partial(litellm.completion, model="sagemaker_chat/test-endpoint", messages=messages, **common),
        partial(litellm.completion, model="bedrock/invoke/anthropic.claude-sonnet-5-5", messages=messages, **common),
    )
    for action in provider_calls:
        try:
            action()
        except litellm.APIConnectionError as error:
            _require(
                f'Install AWS support with pip install "{get_distribution_name()}[aws]"' in str(error),
                "provider wrapper lost consistent installation guidance",
            )
            _require(isinstance(error.__cause__ or error.__context__, ImportError), "provider wrapper lost the import failure")
        else:
            raise AssertionError("AWS provider unexpectedly succeeded without its extra")
    return "AWS logging, cache, secrets, SageMaker and Polly explain the extra"


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
    import litellm
    from litellm._version import get_distribution

    _require(bool(litellm.__file__), "litellm has no __file__")
    from litellm._version import version as reported_version

    _require(reported_version == get_distribution().version, "package version reporting changed")
    return f"imported litellm {reported_version}"


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
    from litellm._version import get_distribution_name

    try:
        import litellm.experimental_mcp_client
    except ImportError as error:
        _require(f"pip install '{get_distribution_name()}[mcp]'" in str(error), f"missing MCP installation guidance: {error}")
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


def check_streaming() -> str:
    import litellm

    stream: Final = litellm.completion(
        model="gpt-4o", messages=[{"role": "user", "content": "ping"}], mock_response="pong", stream=True
    )
    content: Final = "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)
    _require(content == "pong", f"stream returned {content!r}")

    async def async_round_trip() -> None:
        response: Final = await litellm.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "ping"}], mock_response="pong"
        )
        _require(response.choices[0].message.content == "pong", "async completion failed")
        chunks: Final = await litellm.acompletion(
            model="gpt-4o", messages=[{"role": "user", "content": "ping"}], mock_response="pong", stream=True
        )
        parts: Final = tuple([chunk.choices[0].delta.content or "" async for chunk in chunks if chunk.choices])
        _require("".join(parts) == "pong", "async stream failed")

    asyncio.run(async_round_trip())
    return "sync/async completion and streaming round-trip"


def check_retries() -> str:
    import litellm

    outcomes: Final = iter((False, True))

    def attempt(*, max_retries: int, num_retries: int) -> str:
        if not next(outcomes):
            raise RuntimeError("transient failure")
        return "recovered"

    result: Final = litellm.completion_with_retries(original_function=attempt, num_retries=2)
    _require(result == "recovered", "configured retries did not recover")
    return "SDK retry helper recovers from a transient failure"


def check_search_date_parsing() -> str:
    if importlib.util.find_spec("dateutil") is None:
        _expect_extra("search", lambda: __import__("litellm.llms.brave.search.transformation"))
        return "Brave search requests the search extra"
    from litellm.llms.brave.search.transformation import to_yyyy_mm_dd

    for value, expected in (("2026-01-02", "2026-01-02"), ("January 2, 2026", "2026-01-02"), ("1767312000000", "2026-01-02"), ("not a date", None)):
        _require(to_yyyy_mm_dd(value) == expected, "search date normalization changed")
    _require(to_yyyy_mm_dd("02/03/2026", dayfirst=True) == "2026-03-02", "ambiguous date precedence changed")
    return "search date formats and malformed input retain their behavior"


def _expect_extra(extra: str, action: Callable[[], object]) -> None:
    try:
        action()
    except ImportError as error:
        _require(f"litellm[{extra}]" in str(error), f"missing {extra} installation guidance: {error}")
        _require(isinstance(error.__cause__, ModuleNotFoundError), "missing dependency cause was lost")
    else:
        raise AssertionError(f"requested capability silently succeeded without {extra}")


def check_prompt_rendering() -> str:
    import litellm
    from litellm.integrations.dotprompt.prompt_manager import PromptManager
    from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager
    from litellm.litellm_core_utils.prompt_templates.factory import ahf_chat_template, hf_chat_template, prompt_factory
    from litellm.llms.watsonx.chat.transformation import IBMWatsonXChatConfig

    messages: Final = [{"role": "user", "content": "hello"}]
    template: Final = "{{ messages[0].content }}"
    if importlib.util.find_spec("jinja2") is None:
        previous: Final = litellm.known_tokenizer_config
        litellm.known_tokenizer_config = {**previous, "unknown/template-model": {"status": "success", "tokenizer": {"chat_template": template}}, "openai/gpt-oss-120b": {"status": "success", "tokenizer": {"chat_template": template}}}
        for action in (
            PromptManager,
            partial(hf_chat_template, "test", messages, template),
            lambda: asyncio.run(ahf_chat_template("test", messages, template)),
            partial(prompt_factory, model="unknown/template-model", messages=messages),
            partial(IBMWatsonXChatConfig.apply_prompt_template, "openai/gpt-oss-120b", messages),
            lambda: asyncio.run(IBMWatsonXChatConfig.aapply_prompt_template("openai/gpt-oss-120b", messages)),
            partial(DotpromptManager(prompt_data={"hello": {"content": "hi"}}).should_run_prompt_management, "hello", None, {}),
        ):
            _expect_extra("prompts", action)
        litellm.known_tokenizer_config = previous
        return "sync/async templates and prompt selection explain the extra without fallback"
    _require(hf_chat_template("test", messages, template) == "hello", "sync rendering changed")
    _require(asyncio.run(ahf_chat_template("test", messages, template)) == "hello", "async rendering changed")
    manager: Final = PromptManager()
    metadata, content = manager._parse_frontmatter('---\nmodel: openai/test\ntags: [one, two]\n---\nHello {{ name }}')
    _require(metadata["tags"] == ["one", "two"], "nested YAML frontmatter changed")
    _require(manager.jinja_env.from_string(content).render(name="world") == "Hello world", "prompt rendering changed")
    from jinja2.exceptions import SecurityError

    try:
        manager.jinja_env.from_string("{{ value.__class__.__mro__ }}").render(value="test")
    except SecurityError:
        return "YAML, sync/async rendering and sandbox protections retained"
    raise AssertionError("prompt sandbox allowed private attribute traversal")


def check_environment_files() -> str:
    import litellm

    available: Final = importlib.util.find_spec("dotenv") is not None
    with tempfile.TemporaryDirectory() as directory:
        root: Final = Path(directory)
        (root / ".env").write_text("SDK_ENV_FILE_ONLY=file\nSDK_ENV_PRECEDENCE=file\n", encoding="utf-8")
        source: Final = (
            "import os, json; import litellm; "
            "print(json.dumps([os.getenv('SDK_ENV_FILE_ONLY'), os.getenv('SDK_ENV_PRECEDENCE'), litellm.__file__]))"
        )
        for mode, reload, expected in (
            ("DEV", "False", ["file" if available else None, "process"]),
            ("PRODUCTION", "False", [None, "process"]),
            ("DEV", "True", ["file", "file"]),
        ):
            environment: Final = {
                **{key: value for key, value in os.environ.items() if key not in ("SDK_ENV_FILE_ONLY", "PYTHON_DOTENV_DISABLED")},
                "LITELLM_MODE": mode,
                "LITELLM_DEV_ENV_HOT_RELOAD": reload,
                "SDK_ENV_PRECEDENCE": "process",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            }
            result: Final = subprocess.run([sys.executable, "-I", "-c", source], cwd=root, env=environment, capture_output=True, text=True, timeout=60)
            if reload == "True" and not available:
                _require(result.returncode != 0 and "litellm[dotenv]" in result.stderr, "explicit reload lost missing-extra guidance")
            else:
                _require(result.returncode == 0, f"environment loading failed: {result.stderr}")
                _require(json.loads(result.stdout.strip().splitlines()[-1]) == [*expected, litellm.__file__], "file/process environment precedence changed")
    return "DEV/PRODUCTION loading and reload precedence match the installation"


def check_integration_configuration() -> str:
    from litellm.integrations.otel.model.flags import is_otel_v2_enabled
    from litellm.integrations.langfuse.langfuse import raise_if_unsupported_langfuse_version
    from litellm.integrations.lunary import LunaryLogger

    _require(is_otel_v2_enabled() is False, "disabled telemetry must work in core")
    if importlib.util.find_spec("packaging") is None:
        _expect_extra("integrations", partial(raise_if_unsupported_langfuse_version, "4.7.0"))
        _expect_extra("integrations", LunaryLogger)
    else:
        raise_if_unsupported_langfuse_version("4.7.0")
        try:
            raise_if_unsupported_langfuse_version("5.0.0rc1")
        except ImportError as error:
            _require("5.0.0rc1" in str(error), "version rejection lost its explanation")
        else:
            raise AssertionError("unsupported prerelease version was accepted")
    if importlib.util.find_spec("pydantic_settings") is None:
        _expect_extra("integrations", lambda: __import__("litellm.integrations.otel.model.config"))
    else:
        from litellm.integrations.otel import OpenTelemetryV2Config

        _require(OpenTelemetryV2Config().exporters, "telemetry default exporter was lost")
    return "disabled telemetry stays core; optional settings and version checks retain behavior"


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


def check_bedrock_credential_resolution() -> str:
    from unittest import mock

    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

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
    return "bedrock credential resolution works with the AWS extra"


def check_aws_install_guidance() -> str:
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    try:
        BaseAWSLLM()._sign_request(
            service_name="bedrock",
            headers={},
            optional_params={"aws_region_name": "us-east-1"},
            request_data={},
            api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        )
    except ImportError as error:
        _require(f"{get_distribution_name()}[aws]" in str(error), f"missing AWS installation guidance: {error}")
        return f"AWS signing explains how to install {get_distribution_name()}[aws]"
    raise AssertionError("AWS signing succeeded without the AWS extra")


def check_mantle_bearer_authentication() -> str:
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
    from litellm.llms.bedrock_mantle.common_utils import BedrockMantleAuthMixin

    signer: Final = BedrockMantleAuthMixin()
    signer._aws_signer = BaseAWSLLM()
    headers, body = signer.sign_request(
        headers={},
        optional_params={},
        request_data={"model": "example"},
        api_base="https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions",
        api_key="supplied-bearer-token",
    )
    _require(headers["Authorization"] == "Bearer supplied-bearer-token", "bearer token was not preserved")
    _require(body is not None and json.loads(body) == {"model": "example"}, "request body changed")
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    prepared: Final = BaseAWSLLM().get_request_headers(
        credentials=None, aws_region_name="us-east-1", extra_headers=None,
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com/model/example/converse",
        data='{"text":"café"}', headers={"Content-Type": "application/json"}, api_key="supplied-bearer-token",
    )
    _require(prepared.headers["Authorization"] == "Bearer supplied-bearer-token", "Converse bearer token changed")
    _require(prepared.body == '{"text":"café"}'.encode(), "Converse body encoding changed")
    return "Mantle and Converse bearer authentication work without AWS credentials"


def check_tokenizer_fallback() -> str:
    import litellm
    from litellm.litellm_core_utils.tokenizer import HuggingFace, HuggingFaceTokenizer, OpenAIEncoding, Tokenizer
    from litellm.rust_bridge import tokenizer

    _require(isinstance(litellm.encoding, Tokenizer), "core encoding is excluded from runtime Tokenizer alias")
    _require(isinstance(OpenAIEncoding.from_tiktoken("cl100k_base"), Tokenizer), "native encoding is excluded from runtime alias")
    native: Final = HuggingFaceTokenizer.from_str(litellm.utils.claude_json_str)
    _require(isinstance(native, HuggingFace) and isinstance(native, Tokenizer), "native Hugging Face runtime aliases failed")
    if importlib.util.find_spec("tokenizers") is not None:
        custom: Final = litellm.create_tokenizer(litellm.utils.claude_json_str)
        tokens: Final = litellm.encode(text="hello world", custom_tokenizer=custom)
        _require(litellm.decode(tokens=tokens, custom_tokenizer=custom) == "hello world", "tokenizer round trip failed")
        _require(isinstance(custom["tokenizer"], HuggingFace), "runtime HuggingFace alias is incompatible")
        _require(isinstance(custom["tokenizer"], Tokenizer), "runtime Tokenizer alias is incompatible")
        return "custom Hugging Face tokenizer round-trips and runtime aliases resolve"
    try:
        tokenizer._python_huggingface_tokenizer()
    except ImportError as error:
        _require(f"{get_distribution_name()}[tokenizers]" in str(error), f"missing tokenizer installation guidance: {error}")
    else:
        raise AssertionError("Python Hugging Face tokenizer loaded without its extra")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", RuntimeWarning)
        for _ in range(2):
            _require(litellm.token_counter(model="llama-3", text="hello world") > 0, "token counting fallback failed")
    _require(len(caught) == 1 and f"{get_distribution_name()}[tokenizers]" in str(caught[0].message), "fallback must warn once")
    return "missing Python tokenizer explains installation and automatic counting falls back"


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("environment is base-only", check_environment_is_base_only),
    ("import litellm", check_import),
    ("optional MCP installation guidance", check_mcp_install_guidance),
    ("chat completion", check_completion),
    ("embedding", check_embedding),
    ("streaming", check_streaming),
    ("retries", check_retries),
    ("search date parsing", check_search_date_parsing),
    ("prompt rendering", check_prompt_rendering),
    ("integration configuration", check_integration_configuration),
    ("environment files", check_environment_files),
    ("bundled model metadata", check_bundled_model_metadata),
    ("token counter", check_token_counter),
    ("Mantle bearer authentication", check_mantle_bearer_authentication),
    ("tokenizer behavior", check_tokenizer_fallback),
)


def _run(check: Callable[[], str]) -> tuple[bool, str]:
    try:
        return True, check()
    except Exception:
        return False, traceback.format_exc()


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("core", "cli", "validation", "aws", "tokenizers", "aws,tokenizers", "sdk-extras", "proxy", "integrations", "prompts", "search", "dotenv", "prompts,search", "mcp"),
        default="core",
    )
    profile: Final = parser.parse_args().profile
    checks: Final = (
        ("optional dependencies", partial(check_optional_dependencies, profile)),
        ("validation", partial(check_validation, profile)),
        ("AWS feature guidance", check_aws_feature_guidance),
        *(
            check
            for check in CHECKS
            if not (profile in ("cli", "proxy") and check[0] == "environment is base-only")
            and not (profile == "proxy" and check[0] == "optional MCP installation guidance")
        ),
        (
            "AWS behavior",
            check_bedrock_credential_resolution
            if profile in ("aws", "aws,tokenizers", "sdk-extras", "proxy")
            else check_aws_install_guidance,
        ),
    )
    print(f"base SDK smoke check on {sys.executable}")
    for label, check in checks:
        passed, detail = _run(check)
        if not passed:
            print(f"FAIL  {label}:\n{detail}")
            print(f"SDK installation profile {profile} failed at: {label}")
            return 1
        print(f"PASS  {label}: {detail}")

    print(f"\nall {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
