import sys
import types

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import response_schema_prompt
from litellm.llms.anthropic.completion.transformation import AnthropicTextConfig
from litellm.llms.codestral.completion.handler import CodestralTextCompletion
from litellm.llms.ollama.completion.transformation import OllamaConfig
from litellm.llms.petals.completion import handler as petals_handler
from litellm.llms.predibase.chat.transformation import PredibaseConfig
from litellm.llms.vllm.completion import handler as vllm_handler
from litellm.types.utils import ModelResponse, TextCompletionResponse

MODEL = "partial-template-model"
ROLES = {"user": {"pre_message": "[INST] ", "post_message": " [/INST]"}}
MESSAGES = [{"role": "user", "content": "hi"}]
RENDERED_MESSAGES = "[INST] hi [/INST]"

RESPONSE_SCHEMA = {"type": "object"}
RENDERED_SCHEMA = f"[INST] {RESPONSE_SCHEMA} [/INST]"

INITIAL = "<start>"
FINAL = "<end>"

PARTIAL_TEMPLATE = {"roles": ROLES}
FULL_TEMPLATE = {"roles": ROLES, "initial_prompt_value": INITIAL, "final_prompt_value": FINAL}


class _PromptCaptured(Exception):
    def __init__(self, prompt):
        super().__init__(prompt)
        self.prompt = prompt


class _CapturingLogging:
    def pre_call(self, input, api_key, additional_args=None, **kwargs):
        raise _PromptCaptured(input)

    def post_call(self, *args, **kwargs):
        pass


def _captured_prompt(call):
    with pytest.raises(_PromptCaptured) as excinfo:
        call()
    return excinfo.value.prompt


@pytest.fixture(autouse=True)
def stub_vllm_import(monkeypatch):
    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def __init__(self, model):
            self.model = model

        def generate(self, prompts, sampling_params):
            raise _PromptCaptured(prompts)

    stub = types.ModuleType("vllm")
    stub.LLM = _LLM
    stub.SamplingParams = _SamplingParams

    monkeypatch.setitem(sys.modules, "vllm", stub)
    monkeypatch.setattr(vllm_handler, "llm", None)


def _render_ollama(template, monkeypatch):
    request = OllamaConfig().transform_request(
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params={"custom_prompt_dict": {MODEL: template}},
        headers={},
    )
    return request["prompt"]


def _render_predibase(template, monkeypatch):
    request = PredibaseConfig().transform_request(
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params={"custom_prompt_dict": {MODEL: template}},
        headers={},
    )
    return request["inputs"]


def _render_anthropic_text(template, monkeypatch):
    monkeypatch.setattr(litellm, "custom_prompt_dict", {MODEL: template})
    return AnthropicTextConfig()._get_anthropic_text_prompt_from_messages(messages=MESSAGES, model=MODEL)


def _render_response_schema_prompt(template, monkeypatch):
    monkeypatch.setattr(litellm, "custom_prompt_dict", {f"{MODEL}/response_schema_prompt": template})
    return response_schema_prompt(model=MODEL, response_schema=RESPONSE_SCHEMA)


def _render_petals(template, monkeypatch):
    monkeypatch.setattr(litellm, "custom_prompt_dict", {MODEL: template})
    return _captured_prompt(
        lambda: petals_handler.completion(
            model=MODEL,
            messages=MESSAGES,
            api_base="http://petals.invalid",
            model_response=ModelResponse(),
            print_verbose=lambda *args, **kwargs: None,
            encoding=None,
            logging_obj=_CapturingLogging(),
            optional_params={},
        )
    )


def _render_codestral(template, monkeypatch):
    return _captured_prompt(
        lambda: CodestralTextCompletion().completion(
            model=MODEL,
            messages=MESSAGES,
            api_base="http://codestral.invalid",
            custom_prompt_dict={MODEL: template},
            model_response=TextCompletionResponse(),
            print_verbose=lambda *args, **kwargs: None,
            encoding=None,
            api_key="sk-not-used",
            logging_obj=_CapturingLogging(),
            optional_params={},
            timeout=1.0,
        )
    )


def _render_vllm_completion(template, monkeypatch):
    return _captured_prompt(
        lambda: vllm_handler.completion(
            model=MODEL,
            messages=MESSAGES,
            model_response=ModelResponse(),
            print_verbose=lambda *args, **kwargs: None,
            encoding=None,
            logging_obj=_CapturingLogging(),
            optional_params={},
            custom_prompt_dict={MODEL: template},
        )
    )


def _render_vllm_batch(template, monkeypatch):
    prompts = _captured_prompt(
        lambda: vllm_handler.batch_completions(
            model=MODEL,
            messages=[MESSAGES],
            optional_params={},
            custom_prompt_dict={MODEL: template},
        )
    )
    assert len(prompts) == 1
    return prompts[0]


CALL_SITES = [
    pytest.param(_render_ollama, RENDERED_MESSAGES, id="ollama"),
    pytest.param(_render_petals, RENDERED_MESSAGES, id="petals"),
    pytest.param(_render_vllm_completion, RENDERED_MESSAGES, id="vllm-completion"),
    pytest.param(_render_vllm_batch, RENDERED_MESSAGES, id="vllm-batch-completions"),
    pytest.param(_render_predibase, RENDERED_MESSAGES, id="predibase"),
    pytest.param(_render_codestral, RENDERED_MESSAGES, id="codestral"),
    pytest.param(_render_anthropic_text, RENDERED_MESSAGES, id="anthropic-text"),
    pytest.param(_render_response_schema_prompt, RENDERED_SCHEMA, id="response-schema-prompt"),
]


@pytest.mark.parametrize(("render", "rendered_body"), CALL_SITES)
def test_template_without_the_optional_prompt_values_renders(render, rendered_body, monkeypatch):
    assert render(PARTIAL_TEMPLATE, monkeypatch) == rendered_body


@pytest.mark.parametrize(("render", "rendered_body"), CALL_SITES)
def test_template_with_the_optional_prompt_values_still_applies_them(render, rendered_body, monkeypatch):
    assert render(FULL_TEMPLATE, monkeypatch) == INITIAL + rendered_body + FINAL
