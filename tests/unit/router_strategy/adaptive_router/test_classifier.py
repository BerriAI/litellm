import pytest

from litellm.router_strategy.adaptive_router.classifier import classify_prompt
from litellm.types.router import RequestType


@pytest.mark.parametrize(
    "text",
    [
        "Write a Python function that reverses a linked list",
        "Implement a REST API endpoint for user signup",
        "Create a bash script to back up my postgres database",
    ],
)
def test_classify_code_generation(text):
    assert classify_prompt(text) == RequestType.CODE_GENERATION


@pytest.mark.parametrize(
    "text",
    [
        "Explain what this function does: def foo(): ...",
        "Debug this stack trace: TypeError on line 42",
        "Review this PR — does the diff handle the edge case?",
    ],
)
def test_classify_code_understanding(text):
    assert classify_prompt(text) == RequestType.CODE_UNDERSTANDING


@pytest.mark.parametrize(
    "text",
    [
        "Design a microservice architecture for an event-driven system",
        "Should I use PostgreSQL or DynamoDB for high-write workloads?",
        "How should I structure my Django app for multi-tenancy?",
    ],
)
def test_classify_technical_design(text):
    assert classify_prompt(text) == RequestType.TECHNICAL_DESIGN


@pytest.mark.parametrize(
    "text",
    [
        "Solve the integral of x^2 from 0 to 5",
        "If A implies B and B implies C, then prove A implies C",
        "Calculate the probability of two heads in three coin flips",
    ],
)
def test_classify_analytical_reasoning(text):
    assert classify_prompt(text) == RequestType.ANALYTICAL_REASONING


@pytest.mark.parametrize(
    "text",
    [
        "Draft an email to my team announcing the launch",
        "Rewrite this paragraph to be more concise and professional",
        "Proofread my blog post for grammar and tone",
    ],
)
def test_classify_writing(text):
    assert classify_prompt(text) == RequestType.WRITING


@pytest.mark.parametrize(
    "text",
    [
        "Who is the current president of France?",
        "What is the capital of Australia?",
        "Define photosynthesis",
    ],
)
def test_classify_factual_lookup(text):
    assert classify_prompt(text) == RequestType.FACTUAL_LOOKUP


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "tell me about your day",
        "interesting",
    ],
)
def test_classify_general_fallback(text):
    assert classify_prompt(text) == RequestType.GENERAL


def test_classify_empty_string():
    assert classify_prompt("") == RequestType.GENERAL


def test_classify_whitespace_only():
    assert classify_prompt("   \n\t  ") == RequestType.GENERAL


def test_classify_truncates_very_long_input():
    text = (
        "Who is the current president of France? "
        + "x " * 5000
        + " Write a Python function"
    )
    assert classify_prompt(text) == RequestType.FACTUAL_LOOKUP


def test_classify_is_deterministic():
    text = "Implement a REST API endpoint for user signup"
    results = {classify_prompt(text) for _ in range(10)}
    assert len(results) == 1


def test_classify_returns_request_type_enum():
    result = classify_prompt("hello")
    assert isinstance(result, RequestType)
