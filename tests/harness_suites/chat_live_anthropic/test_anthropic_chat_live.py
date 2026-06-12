# What is this?
## Unit tests for Anthropic Adapter

import os
import sys

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from base_llm_unit_tests import BaseLLMChatTest, BaseAnthropicChatTest
from litellm import completion

@pytest.mark.parametrize(
    "tool_type, tool_config, message_content",
    [
        (
            "computer_20250124",
            {
                "type": "computer_20250124",
                "function": {
                    "name": "computer",
                    "parameters": {
                        "display_height_px": 100,
                        "display_width_px": 100,
                        "display_number": 1,
                    },
                },
            },
            "Save a picture of a cat to my desktop.",
        ),
        (
            "web_fetch_20250910",
            {
                "type": "web_fetch_20250910",
                "name": "web_fetch",
                "max_uses": 5,
            },
            "Please analyze the content at https://example.com/article",
        ),
    ],
)
def test_anthropic_tool_use(tool_type, tool_config, message_content):
    """Test Anthropic tool use with computer use and web fetch tools."""

    litellm._turn_on_debug()

    tools = [tool_config]
    model = "claude-sonnet-4-5-20250929"
    messages = [{"role": "user", "content": message_content}]

    try:
        resp = completion(
            model=model,
            messages=messages,
            tools=tools,
        )
        print(f"Tool type: {tool_type}")
        print(resp)
        assert resp is not None
    except litellm.InternalServerError:
        pass


class TestAnthropicCompletion(BaseLLMChatTest, BaseAnthropicChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "anthropic/claude-sonnet-4-5-20250929"}

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            convert_to_anthropic_tool_invoke,
        )

        result = convert_to_anthropic_tool_invoke([tool_call_no_arguments])
        print(result)

    def test_tool_call_and_json_response_format(self):
        """
        Test that the tool call and JSON response format is supported by the LLM API
        """
        litellm.set_verbose = True
        from pydantic import BaseModel
        from litellm.utils import supports_response_schema

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        class RFormat(BaseModel):
            question: str
            answer: str

        base_completion_call_args = self.get_base_completion_call_args()
        if not supports_response_schema(base_completion_call_args["model"], None):
            pytest.skip("Model does not support response schema")

        try:
            res = litellm.completion(
                **base_completion_call_args,
                messages=[
                    {
                        "role": "system",
                        "content": "response user question with JSON object",
                    },
                    {"role": "user", "content": "Hey! What's the weather in NewYork?"},
                ],
                tool_choice="required",
                response_format=RFormat,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
            )
            assert res is not None

            assert res.choices[0].message.tool_calls is not None
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")


@pytest.mark.asyncio
async def test_anthropic_structured_output():
    """
    Test the _transform_response_for_structured_output

    Relevant Issue: https://github.com/BerriAI/litellm/issues/8291
    """
    from litellm import acompletion

    args = {
        "model": "claude-sonnet-4-5-20250929",
        "seed": 3015206306868917280,
        "stop": None,
        "messages": [
            {
                "role": "system",
                "content": 'You are a hello world agent.\nAlways respond in the following valid JSON format: {\n  "response": "response",\n}\n',
            },
            {"role": "user", "content": "Respond with hello world"},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "drop_params": True,
    }

    response = await acompletion(**args)
    assert response is not None

    print(response)


def test_anthropic_citations_api():
    """
    Test the citations API
    """

    try:
        resp = completion(
            model="claude-sonnet-4-5-20250929",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "The grass is green. The sky is blue.",
                            },
                            "title": "My Document",
                            "context": "This is a trustworthy document.",
                            "citations": {"enabled": True},
                        },
                        {
                            "type": "text",
                            "text": "What color is the grass and sky?",
                        },
                    ],
                }
            ],
        )

    except litellm.InternalServerError:
        pytest.skip("Anthropic overloaded")

    citations = resp.choices[0].message.provider_specific_fields["citations"]

    assert citations is not None
    if citations:
        citation = citations[0][0]
        assert "supported_text" in citation
        assert "cited_text" in citation
        assert "document_index" in citation
        assert "document_title" in citation
        assert "start_char_index" in citation
        assert "end_char_index" in citation


def test_anthropic_citations_api_streaming():

    resp = completion(
        model="claude-sonnet-4-5-20250929",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "The grass is green. The sky is blue.",
                        },
                        "title": "My Document",
                        "context": "This is a trustworthy document.",
                        "citations": {"enabled": True},
                    },
                    {
                        "type": "text",
                        "text": "What color is the grass and sky?",
                    },
                ],
            }
        ],
        stream=True,
    )

    has_citations = False
    for chunk in resp:
        print(f"returned chunk: {chunk}")
        if provider_specific_fields := chunk.choices[0].delta.provider_specific_fields:
            if "citation" in provider_specific_fields:
                has_citations = True

    assert has_citations


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5-20250929",
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
def test_anthropic_thinking_output(model):

    litellm._turn_on_debug()

    resp = completion(
        model=model,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    print(resp)
    assert resp.choices[0].message.reasoning_content is not None
    assert isinstance(resp.choices[0].message.reasoning_content, str)
    assert resp.choices[0].message.thinking_blocks is not None
    assert isinstance(resp.choices[0].message.thinking_blocks, list)
    assert len(resp.choices[0].message.thinking_blocks) > 0

    assert resp.choices[0].message.thinking_blocks[0]["type"] == "thinking"
    assert resp.choices[0].message.thinking_blocks[0]["signature"] is not None


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5-20250929",
        # "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        # "bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
def test_anthropic_thinking_output_stream(model):
    litellm.set_verbose = True
    try:
        # litellm._turn_on_debug()
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            stream=True,
            thinking={"type": "enabled", "budget_tokens": 1024},
            timeout=10,
        )

        reasoning_content_exists = False
        signature_block_exists = False
        for chunk in resp:
            print(f"chunk 2: {chunk}")
            if (
                hasattr(chunk.choices[0].delta, "thinking_blocks")
                and chunk.choices[0].delta.thinking_blocks is not None
                and chunk.choices[0].delta.reasoning_content is not None
                and isinstance(chunk.choices[0].delta.thinking_blocks, list)
                and len(chunk.choices[0].delta.thinking_blocks) > 0
                and isinstance(chunk.choices[0].delta.reasoning_content, str)
            ):
                reasoning_content_exists = True
                print(chunk.choices[0].delta.thinking_blocks[0])
                if chunk.choices[0].delta.thinking_blocks[0].get("signature"):
                    signature_block_exists = True
                    assert (
                        chunk.choices[0].delta.thinking_blocks[0]["type"] == "thinking"
                    )
        assert reasoning_content_exists
        assert signature_block_exists
    except litellm.Timeout:
        pytest.skip("Model is timing out")


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5-20250929",
        # "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
def test_anthropic_thinking_in_assistant_message(model):
    litellm._turn_on_debug()
    params = {
        "model": model,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Alright, let's get started with resolving this issue about implementing `__len__` for the Pipeline class in scikit-learn.\n\n### Step 1: Explore the repository structure to understand it better\n\nLet's first explore the repository structure to get a better understanding of the codebase:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nThis will give us an overview of the top-level directories and files. From there, we can delve deeper into relevant parts of the code.\n\nLet's look at the main organization of the repository first, then we'll locate the Pipeline class, understand how it's currently implemented, and then add the `__len__` method to it.\n\nNext, let's look at the pipeline module, as the issue mentions `pipe[:len(pipe)]`, indicating that there is indexing support for Pipeline, but it lacks a `__len__` method.\n\nSince the issue mentions that Pipeline supports indexing but not `__len__`, I'll check the Pipeline class to understand its implementation.\n\nThe Pipeline class is likely to be in sklearn/pipeline.py or within a pipeline directory. Let's look for it:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nThis will help us locate the file containing the Pipeline class.\n\nOnce we've located the file, we'll examine it to understand:\n1. How indexing is currently implemented\n2. Where we need to add the `__len__` method\n3. How the length of a pipeline should be determined\n\nThen, we'll implement the `__len__` method, test our changes to ensure they fix the issue, and verify that our changes don't break existing functionality.\n\nLet's begin by exploring the repository.\n\n### Step 2: Reproduce the error\n\nAfter exploring the repository, I'll create a script to reproduce the error described in the issue. This script will use the code provided in the \"Steps/Code to Reproduce\" section of the issue description.\n\nThen I'll run the script to confirm the error occurs as described, which will help us understand what needs to be fixed.\n\n### Step 3: Fix the issue\n\nOnce I've understood the problem and located the relevant files, I'll modify the code to add the `__len__` method to the Pipeline class. This method should return the number of steps in the pipeline, which is consistent with what a user would expect when calling `len(pipe)`.\n\n### Step 4: Test the fix\n\nAfter implementing the fix, I'll rerun the reproduction script to verify that the error is resolved and `len(pipe)` works correctly.\n\n### Step 5: Consider edge cases\n\nI'll think about potential edge cases like:\n- Empty pipelines\n- Very long pipelines\n- Any other special cases where the `__len__` implementation might behave unexpectedly\n\nI'll add tests for these cases to ensure our implementation handles them correctly.\n\n### Step 6: Comprehensive testing\n\nI'll run the existing tests to ensure our changes don't break any existing functionality, and I'll also write additional tests to specifically test our new `__len__` method.\n\nLet's start with exploring the repository.\n\nLet's start by exploring the repository structure:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nNow, let's find the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nBased on the repository structure, let's look at the pipeline implementation:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's check if there are tests for the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*test*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nNow, let's create a script to reproduce the error. I'll create a file in the workspace directory and add the code from the issue description:\n\n\ncat > /workspace/reproduce_pipeline_len_error.py << 'EOF'\nfrom sklearn import svm\nfrom sklearn.datasets import samples_generator\nfrom sklearn.feature_selection import SelectKBest\nfrom sklearn.feature_selection import f_regression\nfrom sklearn.pipeline import Pipeline\n\n# generate some data to play with\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\n\nprint(\"Trying to get length of pipeline...\")\ntry:\n    length = len(pipe)\n    print(f\"Length of pipeline: {length}\")\nexcept Exception as e:\n    print(f\"Error: {type(e).__name__}: {e}\")\nEOF\n\n\nLet's run this to confirm the error:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nNow, let's implement the `__len__` method in the Pipeline class. First, I'll examine the Pipeline class to understand how it's structured and where to add the `__len__` method.\n\nFrom the error we saw above and looking at the pipeline.py file, we should add a `__len__` method to the Pipeline class.\n\nLet's modify the pipeline.py file:\n\n\ncp /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py /workspace/pipeline.py.backup\n\n\nNow, let's edit the file to add the `__len__` method:\n\n\nsed -n -e '/class Pipeline/,/^class /p' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nThis will show us the Pipeline class definition so we can understand where to add the `__len__` method.\n\nBased on what we see, I'll add the `__len__` method to the Pipeline class:\n\n\nsed -i -e '/def __getitem__.*:/a\\\n    def __len__(self):\\\n        \"\"\"Returns the length of the Pipeline.\\\n\\\n        Returns\\\n        -------\\\n        length : int\\\n            The number of steps in the pipeline.\\\n        \"\"\"\\\n        return len(self.steps)\n' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's verify that the changes were applied correctly:\n\n\ngrep -A 10 \"__len__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's run our script again to see if the error is fixed:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nIf the error is fixed, let's test some edge cases by modifying our script:\n\n\ncat > /workspace/test_pipeline_len.py << 'EOF'\nfrom sklearn.pipeline import Pipeline\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\n\n# Test case 1: Normal pipeline\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\n\n# Test case 2: Empty pipeline\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\n\n# Test case 3: Long pipeline\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\n\n# Test case 4: Slicing with len\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nsliced_pipe = pipe[:len(pipe)]\nprint(f\"Original pipe steps: {[name for name, _ in pipe.steps]}\")\nprint(f\"Sliced pipe steps: {[name for name, _ in sliced_pipe.steps]}\")\nEOF\n\n\nLet's run this test script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also run the tests for the pipeline module to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/test_pipeline.py -v\n\n\nAfter testing the changes, let's update our script once more to test the edge case where we use `len(pipe)` in slicing:\n\n\ncat >> /workspace/test_pipeline_len.py << 'EOF'\n\n# Test case 5: Various slicing operations with len\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"\\nOriginal pipe length: {len(pipe)}\")\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\n\n# Slice a subset in the middle\nif len(pipe) >= 3:\n    slice4 = pipe[1:len(pipe)-1]\n    print(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nEOF\n\n\nLet's run this updated script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also check all the tests in the sklearn directory to ensure our changes don't break anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nBefore we conclude, let's verify that our changes to scikit-learn match the expected functionality:\n\n1. We have implemented the `__len__` method in the Pipeline class\n2. The method returns the number of steps in the pipeline\n3. We have tested that it works for normal, empty, and long pipelines\n4. We have also tested that slicing operations using `len(pipe)` work correctly\n\nSince the issue description mentioned that the repository already supports indexing but not `__len__`, we didn't have to implement indexing. We only needed to add the `__len__` method to make `len(pipe)` work, which we've done.\n\nOur changes are minimal and focused on solving the specific issue described.\n\nFurthermore, to ensure we've done a thorough job, let's run all sklearn tests to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/\n\n\nLet's check the full implementation of the Pipeline to understand where `__len__` fits in compared to other methods like `__getitem__`:\n\n\nhead -n 300 /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's more specifically check the surrounding methods to ensure our addition fits well with the rest of the class:\n\n\ngrep -A 20 \"__getitem__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's also test our changes against the base commit a62775e99f2a5ea3d51db7160fad783f6cd8a4c5, as mentioned in the requirements:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngit status\n\n\nHmm, we don't have access to the git repository directly. Let's focus on thoroughly testing our changes then.\n\nLet's run the full test suite for the sklearn module to make sure our changes don't break any existing functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nGreat, now let's create a more comprehensive test script to ensure our `__len__` implementation works in various scenarios:\n\n\ncat > /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\"\"\"\nComprehensive tests for the __len__ method of Pipeline.\n\"\"\"\nfrom sklearn.pipeline import Pipeline, FeatureUnion\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Test case 1: Normal pipeline\nprint(\"Test case 1: Normal pipeline\")\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\nassert len(pipe) == 2, \"Length of normal pipeline should be 2\"\n\n# Test case 2: Empty pipeline\nprint(\"\\nTest case 2: Empty pipeline\")\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\nassert len(empty_pipe) == 0, \"Length of empty pipeline should be 0\"\n\n# Test case 3: Long pipeline\nprint(\"\\nTest case 3: Long pipeline\")\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\nassert len(long_pipe) == 10, \"Length of long pipeline should be 10\"\n\n# Test case 4: Pipeline with FeatureUnion\nprint(\"\\nTest case 4: Pipeline with FeatureUnion\")\nunion = FeatureUnion([\n    ('pca', PCA(n_components=1)),\n    ('select', SelectKBest(k=1))\n])\npipe_with_union = Pipeline([\n    ('scaler', StandardScaler()),\n    ('union', union),\n    ('svc', svm.SVC(kernel='linear'))\n])\nprint(f\"Length of pipeline with FeatureUnion: {len(pipe_with_union)}\")\nassert len(pipe_with_union) == 3, \"Length of pipeline with FeatureUnion should be 3\"\n\n# Test case 5: Various slicing operations with len\nprint(\"\\nTest case 5: Various slicing operations with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"Original pipe length: {len(pipe)}\")\nassert len(pipe) == 3, \"Original pipe length should be 3\"\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\nassert len(slice1) == 3, \"Length of pipe[:len(pipe)] should be 3\"\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\nassert len(slice2) == 1, \"Length of pipe[:len(pipe)//2] should be 1\"\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\nassert len(slice3) == 2, \"Length of pipe[len(pipe)//2:] should be 2\"\n\n# Slice a subset in the middle\nslice4 = pipe[1:len(pipe)-1]\nprint(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nassert len(slice4) == 1, \"Length of pipe[1:len(pipe)-1] should be 1\"\n\nprint(\"\\nAll tests passed!\")\nEOF\n\n\nLet's run this comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also run specific tests from the scikit-learn test suite that relate to Pipeline indexing to ensure our changes integrate well:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing\n\n\nOh, there might not be a specific test named `test_pipeline_indexing`. Let's look at the available tests in the test_pipeline.py file:\n\n\ngrep -r \"def test_\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py\n\n\nLet's run the tests that seem related to pipeline indexing or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"index\"\n\n\nSince we might not have direct access to specific test names, let's examine the pipeline test file to identify relevant tests:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -i \"index\\|slicing\\|getitem\"\n\n\nBased on our findings, let's run specific tests that are relevant to our changes:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nLet's check if the test_pipeline.py file contains any tests for slicing functionality:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -A 20 -B 20 \"slicing\\|__getitem__\"\n\n\nLet's modify our comprehensive test script to include more edge cases and ensure it aligns with any existing tests:\n\n\ncat >> /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\n# Test case 6: Testing on pipeline with make_pipeline\nprint(\"\\nTest case 6: Testing on pipeline with make_pipeline\")\nfrom sklearn.pipeline import make_pipeline\n\npipe = make_pipeline(StandardScaler(), PCA(n_components=2), SelectKBest(k=1))\nprint(f\"Length of make_pipeline: {len(pipe)}\")\nassert len(pipe) == 3, \"Length of make_pipeline should be 3\"\n\n# Test case 7: Testing on nested pipelines\nprint(\"\\nTest case 7: Testing on nested pipelines\")\ninner_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))])\nouter_pipe = Pipeline([('inner', inner_pipe), ('svc', svm.SVC())])\nprint(f\"Length of outer pipeline: {len(outer_pipe)}\")\nassert len(outer_pipe) == 2, \"Length of outer pipeline should be 2\"\n\n# Test case 8: Testing __len__ with negative indexing\nprint(\"\\nTest case 8: Testing __len__ with negative indexing\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1))\n])\nneg_slice = pipe[-len(pipe):]\nprint(f\"pipe[-len(pipe):] steps: {[name for name, _ in neg_slice.steps]}\")\nassert len(neg_slice) == 3, \"Length of pipe[-len(pipe):] should be 3\"\n\nneg_slice2 = pipe[:-len(pipe)]\nprint(f\"pipe[:-len(pipe)] steps: {[name for name, _ in neg_slice2.steps]}\")\nassert len(neg_slice2) == 0, \"Length of pipe[:-len(pipe)] should be 0\"\n\n# Test case 9: Testing complex slicing expressions with len\nprint(\"\\nTest case 9: Testing complex slicing expressions with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1)),\n    ('step4', svm.SVC())\n])\ncomplex_slice = pipe[len(pipe)//4:len(pipe)//4*3]\nprint(f\"pipe[len(pipe)//4:len(pipe)//4*3] steps: {[name for name, _ in complex_slice.steps]}\")\nassert len(complex_slice) == 2, \"Length of complex slice should be 2\"\n\nprint(\"\\nAll additional tests passed!\")\nEOF\n\n\nLet's run the updated comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also verify that our implementation doesn't cause any regressions in the scikit-learn test suite:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nNow, let's also run another specific test related to pipeline functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_metaestimators.py | grep -i \"pipeline\"\n\n\nLet's get a deeper understanding of how the Pipeline class is used in tests by examining test files containing pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"Pipeline\" --include=\"*.py\" sklearn/tests/ | head -n 20\n\n\nLet's also look for any tests related to pipeline length or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"len(\" --include=\"*.py\" sklearn/tests/ | grep -i \"pipeline\" | head -n 20\n\n\nWith all our tests passing, it appears we've successfully implemented the `__len__` method for the Pipeline class, and it behaves as expected in various scenarios.\n\nLet's summarize what we've done:\n\n1. We located the Pipeline class in sklearn/pipeline.py\n2. We identified that it was missing a `__len__` method, which was causing the error described in the issue\n3. We implemented the `__len__` method to return the length of the pipeline's steps list\n4. We verified our fix with a reproduction script and comprehensive tests\n5. We confirmed that existing scikit-learn tests still pass with our changes\n\nOur implementation is minimal and focused on the specific issue, adding only the `__len__` method without changing any existing functionality.\n\nAt this point we can conclude that the issue has been successfully resolved. The Pipeline class now supports `len(pipe)`, which allows expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's make sure we've run the specific test for pipeline indexing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"indexing\"\n\n\nLet's also check if there are any tests related to `__len__` or length in the pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"__len__\\|len(\" --include=\"*.py\" sklearn/tests/test_pipeline.py\n\n\nWith all our checks and tests, we're confident that our implementation of the `__len__` method in the Pipeline class correctly resolves the issue described. The method returns the number of steps in the pipeline, which is the expected behavior for `len(pipe)`.\n\nI believe our solution is complete and correct. The changes we made are minimal and focused on the specific issue, and we've thoroughly tested them to ensure they work as expected in various scenarios.\n\nLet's complete the task by running our comprehensive test one more time:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nI think we're good to go. Our implementation of the `__len__` method in the Pipeline class resolves the issue described, allowing expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's do a final check on the changes we made. The diff should show just the addition of the `__len__` method to the Pipeline class:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ndiff /workspace/pipeline.py.backup sklearn/pipeline.py\n\n\nWith all our checks and tests passing, we've successfully resolved the issue described in the problem statement.\n\nAlright, let's summarize our solution once more:\n\nThe issue was that Pipeline in scikit-learn didn't implement the `__len__` method, which was causing `pipe[:len(pipe)]` to raise an error, even though Pipeline supported indexing.\n\nTo resolve this, we added a `__len__` method to the Pipeline class that returns the number of steps in the pipeline. This allows expressions like `len(pipe)` and `pipe[:len(pipe)]` to work correctly, providing a more consistent and intuitive API for users.\n\nWe thoroughly tested our solution with:\n1. A script to reproduce the original error and confirm our fix\n2. Comprehensive tests for various scenarios, including edge cases\n3. Running the existing scikit-learn test suite to ensure no regressions\n\nThese tests confirm that our solution correctly resolves the issue while maintaining compatibility with the rest of the codebase.\n\nAnd that concludes our implementation.",
                        "signature": "ErUBCkYIARgCIkDnKEqW20HOxB9jS85ReLznKuKSV88wnAJf/M+y9ge7U+GOWAHd/trgmvpgHpJmOBkeUWUftR5g+81TNmAsDd2EEgxQHv0uMQBMvelCR+saDFvnSSfaNOj3NTVYpyIwG1ddKj5AVnXe382HERyVchmOTdn1ebogEFpXlRE66Mckq57xLSygusq6MwAV/6RvKh28HEuwcXNQn+BPIxQDVCwGGUh5g4fpZ6IgGk3mRg==",
                    },
                    {
                        "type": "text",
                        "text": "I'll help you implement the `__len__` method for Pipeline in scikit-learn to fix the issue described. Let's start by exploring the repository and understanding the problem better.\n\n## Step 1: Explore the repository structure\n\nLet's first look at the repository structure:",
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Who do you know?"}]},
        ],
        "max_tokens": 32768,
        "thinking": {"type": "enabled", "budget_tokens": 30720},
    }

    response = litellm.completion(**params)

    assert response is not None


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5-20250929",
        # "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ],
)
def test_anthropic_redacted_thinking_in_assistant_message(model):
    litellm._turn_on_debug()
    params = {
        "model": model,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "redacted_thinking",
                        "data": "EqkBCkYIARgCKkAflgFkky5bvpaXt2GnDYgbA8QOCr+BF53t+UmiRA22Z7Ply9z2xfTGYSqvjlhIEsV6WDPdVoXndztvhKCzE2PUEgxwXpRD1hBLUSajVWoaDEftxmhqdg0mRwPUGCIwcht1EH91+gznPoaMNquU4sGeaOLFaeyNeG4dJXsYT/Jc4OG3453LN5ra4uVxC/GgKhGMQ1A9aO2Ac0O5M+bOdp1RFw==Eo0CCkYIARgCKkCcHATldbjR0vfU1DlNaQr3J2GKem6OjFybQyshp4C9XnysT/6y1CNcI+VGsbX99GfKLGqcsGYr81WlM+d7NscJEgxzkyZuwL3QnnxFiUUaDIA3nZpQa15D5XD72yIwyIGpJwhdavzXvE1bQLZj43aNtznG6Uwsxx4ZlLv83SUqH7GqzMxvm3stLj3cYmKMKnUqqhpeluvoxODUY/fhhF6Bjsj9C1MIRL+9urDH2EtAmZ+BrvLoXjRlbEH9+DtzLE57I1ShMDbUqLJXxXTcjhPkmu3JscBYf0waXfUgrQl2Pnv5dAxM2S3ZASk8di7ak0XcRknVBhhaR2ykdDbVyxzFzyZo8Fc=EtcBCkYIARgCKkCl6nQeKqHIBgdZ1EByLfEwnlZxsZWoDwablEKqRAIrKvB10ccs6RZqrTMZgcMLaW3QpWwnI4fC/WiOe811B94JEgyvTK4+E/zB+a42bYcaDOPesimKdlIPLT7VQiIwplWjvDcbe16vZSJ0OezjHCHEvML4QJPyvGE3NRHcLzC9UiGYriFys5zgv0O7qKr5Kj/56IL1BbaFqSANA7vjGoW+GSlv294L4LzqNWCD0ANzDnEjlXlVeibNM74v+KKXRVwn/IInHPog4hJA0/3GQyA=EtwBCkYIARgCKkBda4XEzq+PTfE7niGdYVzvAXRTb+3ujsDVGhVNtFnPx6K/I6ORfxOWmwEuk7iXygehQA18p0CVYLsCU4AHFvtjEgzYH2JNCxa8F07pGioaDOA635mdHKbyiecBJSIwshUavES7HZBnA4l3k8l92LAhuJQV1C5tUgKkk0pHRT+/OzDfXvxsZSx7AmR7J3QXKkQwHL6K9yZEWdeh/B22ft/GxyRViO7nZrT95PAAux31u++rYQyeFJ+rv0Yrs/KoBnlNUg9YFOpDMo1bMWV9n4CGwq92bw==EtEBCkYIARgCKkCZdn2NBzxiOEJt/E8VOs6YLbYjRaCkvhEdz5apcEZlBQJpulvgv1JvamrMZD0FCJZVTwxd/65M9Ady/LbtYTh7EgwtL7W9DXSFjxPErCIaDGk0e/bXY8yJdjk3CSIwYS0TtiaFK8tJrREBFA9IOp+q+tnE8Wl338CbbskRvF5topYmtofuBIG4GQkHvbQjKjn2BmwrEic/CdSEVbvEix7AWEsw92DabVmseTQhUbbuYRa4Ou6jXMW2pMJFUBjMr95gF6BlVFr4iEA=EsUBCkYIARgCKkAsEmKjMN9TVYLyBdo1+0uopommcjQx8Fu65+mje5Ft05KOnyKAzuUyORtk5r73glan8L+WlygaOOrZ1hi81219EgwpdTA6qbcaggIWeTIaDDrJ0eTbsqku4VSY8CIw3mJfRyv7ISHih4mpAVioGuuduXbaie5eKn5a+WgQiOmm22uZ4Gv72uluCSGGriHnKi28bHMomrytYLvKNvhL51yf5/Tgm/lIgQ9gyTJLqVzVjGn6ng1sN8vUti/tuGw=EsoBCkYIARgCKkB+jJBrxqqpzyGt5RXDKTBVxTnE8IrYRysAL2U/H171INDMCxrDHxfts3M0wuQirXN/2fZXwmQJIZRzzumA+I2sEgw0ySDeyTfHgTiafo8aDKOTl485koQiPwXipyIwG9n/zWUZ+tgfFELW2rV5/yo6Pq/r9bJdrd2b25qCATwX2gd54gsjWhSvLDkD7pLJKjL6ZuiW4N6hVo6JIR4UL8LxcsP9tET0ElIgQZ/h8HOIi18fQKsEdtseWCFnuXse21KIeg==EtwBCkYIARgCKkDWMlgTA+iKsScbpNtZab6dgMKRZYpQSoJ274+n0TqvLAqHL8GxLm1sMVom81LcVWCZZeIVQFbkmbJxyBovvLoUEgxy6YGb0EeJW10P8XEaDKowL3qI/z000pgR2SIwZIczlDKkqw75UYcEOC6Cx9yc0CdYjJnmQOa4Ezni20SANA8YnBMIYJqW4osO/KalKkTLmgvJRQE1Hk8Bn3af9fIYt+vITYEY4Wr7/UVNBtSXBOMP0YoSgNyzjX/pu2N3oy2Blv/YAgtHIJ3Xwd43clN5F2wU+Q==EtQBCkYIARgCKkD3vxW2GsLyEGtmBpI6NdNyh4i/ea7E9rp5puSHdk/dSCpW5G1wI3nrFIS2bUqZsvsDu3YgcDixG8eeDnzacC/qEgzilh/V8vaE1X9lRlIaDAa17eq6kSgaRrsAfSIwFAXgLu5BUKldMeQdcomRqgmY9hDzkDlRnBrbO9GxXsrmpGTU9iqVZQ7z9OVW522bKjyB/GeuNlv4V8a8uricx1InN8q94coWGCRPvAJVAvhP/YMCcNlvrgoN8C2RGc13e88uDq01r6gpkWTlVDY=EssBCkYIARgCKkAOhKBpvfqIElQ1mlG7NiCiolHnqagXryuwNsODnttLBeVMGBsZ8DgpSGWonVE/22MQgciWLY7WaaeoDcpL3X/pEgx4xuL/KqOgxrBnau4aDH3pQ/Sqr1aHa68YiiIwR6+w9QOWFfut8ZG8z+QkAO/kZVePcELKabHp7ikY+DOjvOt4FfnaChwQFTSGzZhaKjPK4MwQukuZIT1PFGFIh20Hi6wMQlHvsChIF88nUV2EAz4Sgb/vWPiQBbWP3gT3hJBehQY=EtMBCkYIARgCKkCT0yD5m4Rvs3KBNkAC2g7aprLTzKRqF+vdHAeYte9KngJZhThexj65o+q9HOGhIIAsboRhz70xkAybdQdsrg8OEgzQm1M980FeZMCi1XsaDJSFOpIuOhUOkPIs+iIw62jO5yY9ZETmrYtEb+pYN5Cyf467YVOOv7FBo44gIFgUvFklU5+y09k3MGzrBNViKjvkopPoFbpYI9ilB3dN6pAzrzhDzOum+Rsx1N25+UYvdT+yYBilrIPW1XmLmzT+ZMs4eV5caG35ZsNsjQ==EtwBCkYIARgCKkCOShz0/2ZO3u0WH8PBN63fAwKo4TcNFM3axUJL9dK9JJDLtC0XwP9Ee4vqPZyLBao4RyAefbYmY3TJ1As/AbuvEgxbYiyN4UcjaJU9mwkaDP9L3FACdMRQ+UFOSSIwQ0btU6cKIRsSNzvBsP8Fa4Ab7vOnlo4YSAv2lD7ZdDKVcQaWQZHYsQb/QQDfIGKGKkRXhNoET9KyQkb/x8lVpUR1d2u/sHTdgKEjkUdQop88SUFHvkGcJrMUTvnuvUdO4MdHwKnN0IINbDHTEUjUXSQPkpfTTA==EtwBCkYIARgCKkCIwQCFJUrhd1aT8hGMNcPIl+CaSZWsqerPDUGzZnS2tt2+tAs+TAPcKVHC07BdEXj6aKSbrOb8b7OQ/KFbrWJ4Egz980omEnE4djm8t5UaDDXrDJWgFSuZ+LWFmSIw/RzMo5ncKnqvf0TZ1krxMi4/DpAZb0Lgmc1XxGT2JPA4At9EEHNVPrWLXwGM3vUYKkQltG8EJFOWL1In5541dca1pnRDyBg4JVRQ5CuvA/pUCI2e9ARiODI7D+ydZorcnWQ7j2Qc1DguMQVHMbPLyGbQx9vqgQ==EtsBCkYIARgCKkDiH+ww5G0OgaW7zSQD7ZKYdViZfi+KO+TkA/k4rlTKsIwpUILZZ/53ppu93xaEazsD92GXKKSG3B/jBCqjQRg7EgzR3K/BJFTt359xPOgaDEHyoGVloiLS71ufAiIwO77B26VivdVgd2Dmv3DOtUAFs/jDwLM9EmNCBeoivwJPD2hYEKNm6TUWTinGfO2jKkNbrYgpA5esB0y1iXA0qGwRAmnD8ykZc0DT40vvd9EDvb5gHCd7RyjEU9BKnXBPWpGdTi4U+LZKYQ9LEE6sJ8vBm8w3EtUBCkYIARgCKkBbxQIjnTzzKf8Qhfcu+so91+MMbpJNyga27D9tZBtTexYLMJtzDWux4urfCc5TjjX0MvK62lKkhcPLuJE7KiI8EgzFF+TlNgPNp6RoyQgaDBAUDEAsqBMj7z4kciIwUWEZMGkG8ZnjltVpuffHxw5Rqyc+Smh1MnqnWxo0JlCOC43W5JH5KoJ/4RDxX7IjKj2fs5F6eiRMEi+L4KyjDBIvoPoE/wrdC+Fo6c8lMJiYw0MJ/lXgJQv6p0GRe251X+pcfN+2lx067/GLP6qjEtsBCkYIARgCKkCItf9nN0FKJsetom0ZoZvccwboNM2erGP7tIAYsOzsA9lmh7rFI2mFbOOC2WZ1v+QkvxppQ2wO+N35t29LC7RPEgzyJgiM1GHTVN+VPPwaDOXyzSg9BQ85oi58DCIwu/JxKJwVECkbru1d05yhwMYDsJrSJW1BO2ZBrg8Tb48S+dpD6hEPd1itq8cSM3ChKkNv83rGY8Gjg2DiTWDsIqUCD0pb2drrwnjkherr5/EQWdhHC7MijF8zyvqU4tBZrxP+64GcII7P87ja8B4YxGUIw9J7Et0BCkYIARgCKkCInOjYRgGSjcV/WHJ6HjB983rvz/nrOZ9xZMdrTYdHURtXN4zMAjZYQ8ZBk31n4aFGv5PAtDfbjqcytZUaCKicEgwXQrjgS0FHWq/2PwAaDKjYgoXuPPq+RNJUvCIwh1VmSiLGu+3pl7RcCBxnH/ue38EUDZAIRYiDI59h8CVdZpDSqaH8yJvFlR5Jxc8xKkXcEPduWcuONY+vatnIo5AQeSh9HM4oM4DoDma1OvVfdPUpbvaTP3ZhEv4iOMjvwzHBBkvc8b9jV2oTb8Xe50COLFJvURk=EtcBCkYIARgCKkDM4CyfgVBHhusU4C0tg/RwXiAbNtjOoYfcufGUnFlQKcpuJnekvb61EAerBrELguIrvNJIbyqy0Kcd/r64hu1UEgyITWjG3/cVsm/o0JkaDKm1/y0HF1YpqoiFoCIwqImOpk6SngP99aXE4p5c7y9rOvVo3lmKidTUdi1lmtoEZ9sXdY49nLsGeCuCjPJKKj976uFmgrZWIEZIL+HQGVjDOJ7mK8NzAxjX3m0AELsWN5FgbGOHus/S4o2EKi43/MLaRervgaFdrxK9BKGE6LY=EtMBCkYIARgCKkDvEoH/lv1fRxN+JaknzdY53WmQrEGJ7yupv22X2TdxN2+GmY8l1KYONWboOxalfoSbSlp3+zVJXdvTCa60CYnnEgyUslgNTFL5iGt+aq0aDESsIoNRuPYqDc5fbCIw9gHGejHXKw9GMR0sw1RnIF2FBI5Zo5/4EK2AFZ8BU5yAYgJw0wTc16ZVEFEraKS+KjtqVPmiodedFzc+f4kr+U8dy+xQtcsmTe9KcvAYmskvZ6Kl6iCitm/PZdjl/7COePcTVu32QnxZuG4Mpw==EtEBCkYIARgCKkB/SdSv2Jo8DJ4pOOK4mYXhSsPrnf6/ESHL7voj6FbdYPsgg2f3XQByQV93Menel5tgcx0jvNfY7Z9nx4Rz3iTvEgxN/mWUwb6Lb/1BfkAaDBONEsjWD1fKeK8H/iIwy+yJUFPTde2wxI/j6em5uS8HWGsfX9pUB4u/K4QHAd85bn63rrXSxbe2DHIG620UKjk+C6q3aXztOAGAyvhjiN9lnNAFPv93GTnwj+14n07c/xPdHBQyXXi742UBjFdQkmwp3m6RWf5psYU=EuQBCkYIARgCKkBxavD9zRmeX22ltvtCNzZzXTpsAHmNwSuejX7ibJueaDQaSOykBjNJavdMn6yQ8mAxCpNrNmhtBhGxHBGZE668EgzFNqHVE2WctK5ZiN0aDGNFTI5T3/0vDCtFXiIwRDXV5+9nWYGzuih8cG8h4dCs+n90rcL/Tz78QKsfpZeLNpr4aZSU8KHO2OmcmFoOKkxdgzKPy/gOfcCELsudlawbVyobU4CIhOYacIPhi+0XvgjXpqP0JIANaOdawb2zWrKhBKNA4VCHzbFkDm9cV1WrGIw0cEJ3oRU7idRgEsEBCkYIARgCKkDJUpJz2Ct4ZZJlWkAGg1Lc/rVqCd/V5rq01yehv9GkTIaq9H2jgjVKnUV1e4o9F1cUxmMk6fn4XK01sp/szP2GEgyvuemo2Di0USGKingaDCAMXK1kWRk6KofoyyIwxr/Jdwz2RrUytRWMGjrs4MkcQ2rhrVL/00Ktebga9cwrqeDOq+7nN8L64V+XEwsJKimHdmpCQPqYz8rIX25+v2XqcBDXzoBW8+eqdJKRhKcYooLbBXK3DUgRVQ==",
                    },
                    {
                        "type": "text",
                        "text": "I'm not able to respond to special commands or trigger phrases like the one you've shared. Those types of strings don't activate any special modes or features in my system. Is there something specific I can help you with today? I'm happy to assist with questions, have a conversation, provide information, or help with various tasks within my normal capabilities.",
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Who do you know?"}]},
        ],
        "max_tokens": 32768,
        "thinking": {"type": "enabled", "budget_tokens": 30720},
    }

    response = litellm.completion(**params)

    assert response is not None


def test_just_system_message():
    litellm._turn_on_debug()
    litellm.modify_params = True
    params = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [{"role": "system", "content": "You are a helpful assistant."}],
    }

    response = litellm.completion(**params)

    assert response is not None


@pytest.mark.parametrize(
    "optional_params",
    [
        # {
        #     "tools": [{
        #         "type": "web_search_20250305",
        #         "name": "web_search",
        #         "max_uses": 5
        #     }]
        # },
        {"web_search_options": {}}
    ],
)
def test_anthropic_websearch(optional_params: dict):
    litellm._turn_on_debug()
    params = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [
            {
                "role": "user",
                "content": "What is the current weather in Tokyo right now?. Make sure to search the web for an answer",
            }
        ],
        **optional_params,
    }

    try:
        response = litellm.completion(**params)
    except litellm.InternalServerError as e:
        print(e)

    assert response is not None

    print(f"response: {response}\n")
    # When web search is requested and used, server_tool_use should be present
    assert response.usage.server_tool_use is not None
    assert response.usage.server_tool_use.web_search_requests >= 1


def test_anthropic_text_editor():
    litellm._turn_on_debug()
    params = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [
            {
                "role": "user",
                "content": "There'''s a syntax error in my primes.py file. Can you help me fix it?",
            }
        ],
        "tools": [
            {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
        ],
    }

    try:
        response = litellm.completion(**params)
    except litellm.InternalServerError as e:
        print(e)

    assert response is not None


@pytest.mark.parametrize("spec", ["anthropic", "openai"])
@pytest.mark.skipif(
    os.getenv("ZAPIER_CI_CD_MCP_TOKEN") is None, reason="ZAPIER_CI_CD_MCP_TOKEN not set"
)
def test_anthropic_mcp_server_tool_use(spec: str):
    litellm._turn_on_debug()

    if spec == "anthropic":
        tools = [
            {
                "type": "url",
                "url": "https://mcp.zapier.com/api/mcp/mcp",
                "name": "zapier-mcp",
                "authorization_token": os.getenv("ZAPIER_CI_CD_MCP_TOKEN"),
            }
        ]
    elif spec == "openai":
        tools = [
            {
                "type": "mcp",
                "server_label": "zapier",
                "server_url": "https://mcp.zapier.com/api/mcp/mcp",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('ZAPIER_CI_CD_MCP_TOKEN')}"
                },
                "require_approval": "never",
            },
        ]

    params = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
        "tools": tools,
    }

    try:
        response = litellm.completion(**params)
        assert response is not None
    except litellm.InternalServerError as e:
        pytest.skip(f"Skipping test due to internal server error: {e}")


@pytest.mark.parametrize(
    "model", ["openai/gpt-4.1", "anthropic/claude-sonnet-4-5-20250929"]
)
@pytest.mark.skipif(
    os.getenv("ZAPIER_CI_CD_MCP_TOKEN") is None, reason="ZAPIER_CI_CD_MCP_TOKEN not set"
)
def test_anthropic_mcp_server_responses_api(model: str):

    litellm._turn_on_debug()
    tools = [
        {
            "type": "mcp",
            "server_label": "zapier",
            "server_url": "https://mcp.zapier.com/api/mcp/mcp",
            "require_approval": "never",
            "headers": {
                "Authorization": f"Bearer {os.getenv('ZAPIER_CI_CD_MCP_TOKEN')}"
            },
        },
    ]

    response = litellm.responses(
        model=model,
        input="Who won the World Cup in 2022?",
        max_output_tokens=100,
        tools=tools,
    )

    assert response is not None


def test_anthropic_prefix_prompt():
    params = {
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "messages": [
            {"role": "user", "content": "Who won the World Cup in 2022?"},
            {"role": "assistant", "content": "Argentina", "prefix": True},
        ],
    }

    response = litellm.completion(**params)
    print(f"response: {response}")
    assert response is not None
    assert response.choices[0].message.content.startswith("Argentina")


@pytest.mark.asyncio
async def test_claude_tool_use_with_anthropic_acreate():
    response = await litellm.anthropic.messages.acreate(
        messages=[
            {"role": "user", "content": "Hello, can you tell me the weather in Boston?"}
        ],
        model="anthropic/claude-sonnet-4-5-20250929",
        stream=True,
        max_tokens=100,
        tools=[
            {
                "name": "get_weather",
                "description": "Get current weather information for a specific location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ],
    )

    async for chunk in response:
        print(chunk)


def test_anthropic_streaming():

    request_data = {
        "messages": [
            {
                "role": "system",
                "content": "Call the tool, please, but tell me what you are doing before you do it.",  # (so we get some pre-tool streaming output)
            },
            {
                "role": "user",
                "content": "Do what you are told to do in the system prompt",
            },
        ],
        "model": "anthropic/claude-sonnet-4-5-20250929",
        "max_tokens": 7000,
        "parallel_tool_calls": False,
        "stream": True,
        "temperature": 0,
        "tool_choice": "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "call_me_please",
                    "strict": True,
                    "parameters": {
                        "properties": {
                            "a_number": {
                                "description": "String that is text version of a number, e.g. sixty-five. At least a 5 digit number.",
                                "type": "string",
                                "title": "A Number Function",
                            }
                        },
                        "title": "call_me_please",
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["a_number"],
                    },
                    "description": "Call this tool with a number to get a random number back",
                },
            }
        ],
    }

    response = completion(**request_data)

    role_set_count = 0
    for chunk in response:
        if chunk.choices[0].delta.role is not None:
            print(f"role: {chunk.choices[0].delta.role}")
            role_set_count += 1

    assert role_set_count == 1


def test_anthropic_via_responses_api():
    from litellm.types.llms.openai import ResponsesAPIStreamEvents

    response = litellm.responses(
        model="anthropic/claude-sonnet-4-5",
        input="Who won the World Cup in 2022?",
        max_output_tokens=100,
        stream=True,
    )

    assert response is not None

    # Expected event sequence
    expected_events = [
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,  # Can occur multiple times
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    ]

    events_seen = []
    text_delta_count = 0

    for chunk in response:
        print(f"chunk: {chunk}")

        # Each chunk should have a type attribute
        assert hasattr(chunk, "type"), f"Chunk missing 'type' attribute: {chunk}"

        event_type = chunk.type

        # Track events seen
        if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
            text_delta_count += 1
            if ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA not in events_seen:
                events_seen.append(event_type)
        else:
            events_seen.append(event_type)

        # Assert specific structures for each event type
        if event_type == ResponsesAPIStreamEvents.RESPONSE_CREATED:
            assert chunk.type == ResponsesAPIStreamEvents.RESPONSE_CREATED
            assert hasattr(chunk, "response")
            assert chunk.response.status == "in_progress"
            assert hasattr(chunk.response, "id")
            assert hasattr(chunk.response, "model")

        elif event_type == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS:
            assert chunk.type == ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS
            assert hasattr(chunk, "response")
            assert chunk.response.status == "in_progress"

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "item")
            assert chunk.item.type == "message"
            assert chunk.item.role == "assistant"

        elif event_type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED:
            assert chunk.type == ResponsesAPIStreamEvents.CONTENT_PART_ADDED
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "part")
            assert chunk.part.type == "output_text"

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "delta")
            assert isinstance(chunk.delta, str)

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "text")

        elif event_type == ResponsesAPIStreamEvents.CONTENT_PART_DONE:
            assert chunk.type == ResponsesAPIStreamEvents.CONTENT_PART_DONE
            assert hasattr(chunk, "item_id")
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "content_index")
            assert hasattr(chunk, "part")
            assert chunk.part.type == "output_text"

        elif event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
            assert chunk.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE
            assert hasattr(chunk, "output_index")
            assert hasattr(chunk, "item")
            assert chunk.item.status == "completed"

        elif event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
            assert chunk.type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
            assert hasattr(chunk, "response")
            assert chunk.response.status == "completed"
            assert hasattr(chunk.response, "usage")
            assert hasattr(chunk.response, "output")

    # Assert we saw all expected events
    print(f"Events seen: {events_seen}")
    assert (
        events_seen == expected_events
    ), f"Event sequence mismatch. Expected: {expected_events}, Got: {events_seen}"

    # Assert we saw at least one text delta
    assert (
        text_delta_count > 0
    ), f"Expected at least one response.output_text.delta event, got {text_delta_count}"

    print(f"✓ All {len(events_seen)} events matched expected structure")
    print(f"✓ Received {text_delta_count} text delta chunks")


def test_anthropic_structured_output_chat_completion_api():
    response = litellm.completion(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "strict": True,
                "schema": {
                    "description": 'Progress report for the thinking process\n\nThis model represents a snapshot of the agent\'s current progress during\nthe thinking process, providing a brief description of the current activity.\n\nAttributes:\n    agent_doing: Brief description of what the agent is currently doing.\n                Should be kept under 10 words. Example: "Learning about home automation"',
                    "properties": {
                        "agent_doing": {"title": "Agent Doing", "type": "string"}
                    },
                    "required": ["agent_doing"],
                    "title": "ThinkingStep",
                    "type": "object",
                    "additionalProperties": False,
                },
            },
        },
    )
    assert response is not None
    print(f"response: {response}")


def test_anthropic_basic_completion_replay():
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert response is not None
    content = response.choices[0].message.content
    assert isinstance(content, str) and content.strip(), content
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.choices[0].finish_reason in {"stop", "length"}


def test_anthropic_streaming_completion_replay():
    stream = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

    collected_text = ""
    finish_reason = None
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            collected_text += delta.content
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    assert chunk_count > 1, "expected multiple SSE chunks from streaming response"
    assert collected_text.strip(), collected_text
    assert finish_reason in {"stop", "length"}
