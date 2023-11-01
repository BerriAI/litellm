import litellm
from litellm.utils import (
    process_system_message,
    process_messages,
    attempt_message_addition,
    can_add_message,
    _get_last_sentences,
    shorten_message_content,
    shorten_message_to_fit_limit,
    trim_messages,
)

from dotenv import load_dotenv
import pytest
from unittest.mock import patch, Mock

load_dotenv()


# Test for shorten_message_to_fit_limit
@pytest.mark.parametrize(
    "input_message, tokens_needed, expected_output",
    [
        # Assuming the function 'shorten_message_content' removes 10 tokens each call
        (
            {"role": "user", "content": "a" * 100},
            10,
            {
                "role": "user",
                "content": "aaaaaaaaaaaaaaaaaaaaaaaaaaaa[...]aaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            },
        ),
        # Test case with already fitting content
        ({"content": "Short content"}, 20, {"content": "Short content"}),
        # Test case with empty content
        ({"content": ""}, 20, {"content": ""}),
        # More cases...
    ],
)
def test_shorten_message_to_fit_limit(
    input_message, tokens_needed, expected_output, monkeypatch
):
    # Setup the mocks to return values that simulate the actual function's behavior

    # Now, the test can call shorten_message_to_fit_limit without depending on the actual implementations
    model = "gpt-3.5-turbo-0613"
    result = shorten_message_to_fit_limit(input_message, tokens_needed, model)

    # Check that the result matches the expected output
    assert result == expected_output
    # You can also assert that the mocks were called with the expected parameters if necessary


@pytest.mark.parametrize(
    "content, number, expected",
    [
        (
            "This is the first sentence. This is the second sentence.",
            35,
            "This is the second sentence.",
        ),
        ("No delimiter here but a newline\n", 10, ""),
        ("One. Two. Three.", 50, "One. Two. Three."),
        ("", 10, ""),  # Empty content
        # Add more test cases as needed
    ],
)
def test_get_last_sentences(content, number, expected):
    result = _get_last_sentences(content, number)
    assert result == expected


@pytest.mark.parametrize(
    "content, tokens_needed, total_tokens, strategy, expected",
    [
        # Filler strategy test cases
        ("The content that will be shortened", 5, 10, "filler", "The co[...]rtened"),
        # Last sentences strategy test cases
        ("First. Second. Third.", 2, 3, "last_sentences", "Third."),
        # Content is already short enough
        ("Short", 5, 4, "filler", "Short"),
        # Content that can't be shortened should raise an error
        ("Short", 10, 5, "unknown_strategy", ValueError),
        # Add more test cases as needed
    ],
)
def test_shorten_message_content(
    content, tokens_needed, total_tokens, strategy, expected
):
    if issubclass(expected, Exception):
        with pytest.raises(expected):
            shorten_message_content(content, tokens_needed, total_tokens, strategy)
    else:
        result = shorten_message_content(content, tokens_needed, total_tokens, strategy)
        assert result == expected


@pytest.fixture
def mock_get_token_count(monkeypatch):  # noqa: F811
    def fake_get_token_count(messages, model):
        return sum(
            len(msg["content"]) for msg in messages
        )  # Simplified token count based on content length

    monkeypatch.setattr(litellm.utils, "get_token_count", fake_get_token_count)


@pytest.mark.usefixtures("mock_get_token_count")
@pytest.mark.parametrize(
    "message, messages, max_tokens, expected",
    [
        # Can add when there's enough space
        ({"content": "Hello"}, [{"content": "Existing message"}], 50, True),
        # Cannot add when it would exceed the limit
        (
            {"content": "This is a long message"},
            [{"content": "Existing message"}],
            30,
            False,
        ),
        # Edge case: exactly at the limit
        ({"content": "Exact"}, [{"content": "Existing message"}], 25, True),
        # Edge case: empty message list
        ({"content": "New message"}, [], 20, True),
        # Edge case: empty message content
        ({"content": ""}, [{"content": "Existing message"}], 20, True),
        # Add more test cases as needed
    ],
)
def test_can_add_message(message, messages, max_tokens, expected):
    result = can_add_message(
        message, messages, max_tokens, "dummy_model"
    )  # 'dummy_model' since it's not used in the mock
    assert result is expected


@pytest.fixture
def mock_shorten_message_to_fit_limit():
    with patch("litellm.utils.mock_shorten_message_to_fit_limit") as mock:
        yield mock


@pytest.mark.parametrize(
    "current_tokens, max_tokens, can_shorten, expected_result",
    [
        (100, 200, False, True),  # Adding message is under the token limit
        (300, 200, False, False),  # Message can't be shortened
        (
            300,
            200,
            True,
            True,
        ),  # Message is over the limit but can be shortened and added
    ],
)
def test_attempt_message_addition(
    monkeypatch, current_tokens, max_tokens, can_shorten, expected_result
):
    mock_message = {"content": "Test message", "role": "user"}
    final_messages = [{"content": "Existing messages", "role": "user"}]

    # Mock get_token_count to return our predefined token count
    monkeypatch.setattr(
        "litellm.utils.get_token_count", Mock(return_value=current_tokens)
    )

    # Mock shorten_message_to_fit_limit to return the message itself or not
    mock_shorten_message_to_fit_limit = Mock(
        return_value=mock_message if can_shorten else None
    )
    monkeypatch.setattr(
        "litellm.utils.shorten_message_to_fit_limit",
        mock_shorten_message_to_fit_limit,
    )

    # Mock can_add_message based on whether shortening was successful
    monkeypatch.setattr("litellm.utils.can_add_message", Mock(return_value=can_shorten))

    result = attempt_message_addition(
        final_messages, mock_message, max_tokens, "model_name"
    )

    # If expected to succeed, check if the mock_message is in the result
    if expected_result:
        assert mock_message in result
    else:
        # Otherwise, ensure the result list is unchanged
        assert result == final_messages


@pytest.mark.parametrize(
    "system_message, max_tokens, expected_trim",
    [
        (
            " ".join(["System message content."] * 2),
            30,
            False,
        ),  # Under limit, no trim expected
        (
            " ".join(["System message content."] * 30),
            30,
            True,
        ),  # Over limit, trim expected
    ],
)
def test_process_system_message(system_message, max_tokens, expected_trim):
    result_message, remaining_tokens = process_system_message(
        system_message, max_tokens, "model_name"
    )

    assert {"content", "role"} == set(result_message.keys())
    assert result_message["role"] == "system"
    if not expected_trim:
        assert result_message["content"] == system_message
    else:
        assert len(result_message["content"]) < len(system_message)
        assert remaining_tokens == 0


@pytest.mark.parametrize(
    "messages_tokens, max_tokens, expected_final_count",
    [
        ([30, 40, 50], 200, 3),  # All messages fit under the limit
        ([100, 150, 200], 300, 1),  # Only one message fits after trimming
        ([], 300, 0),  # No messages to process
    ],
)
def test_process_messages(
    monkeypatch, messages_tokens, max_tokens, expected_final_count
):
    messages = [
        {"content": f"Message {i}", "role": "user"} for i in range(len(messages_tokens))
    ]

    # Mock get_token_count to return a list of token counts corresponding to the messages.
    monkeypatch.setattr(
        "litellm.utils.get_token_count", Mock(side_effect=messages_tokens)
    )

    # Mock attempt_message_addition to return messages one by one if they fit
    monkeypatch.setattr(
        "litellm.utils.attempt_message_addition",
        Mock(
            side_effect=lambda final_msgs, msg, max_tkns, mdl: final_msgs + [msg]
            if len(final_msgs) < expected_final_count
            else final_msgs
        ),
    )

    final_messages = process_messages(messages, max_tokens, "model_name")

    assert len(final_messages) == expected_final_count


@pytest.fixture
def mock_litellm_model_cost():
    # Mock for the LiteLLM model costs
    return {"gpt3": {"max_tokens": 200}}


@pytest.mark.parametrize(
    "provided_max_tokens, model_name, expected_max_tokens",
    [
        (
            None,
            "gpt3",
            150,
        ),  # `max_tokens` computed from the model's max tokens with `trim_ratio` of 0.75
        (100, None, 100),  # `max_tokens` is directly provided
    ],
)
def test_trim_messages_no_trimming_required(
    monkeypatch,
    mock_litellm_model_cost,
    provided_max_tokens,
    model_name,
    expected_max_tokens,
):
    messages = [{"role": "user", "content": "Hello"}]

    monkeypatch.setattr("litellm.model_cost", mock_litellm_model_cost)

    # Mock get_token_count to return a token count below the max_tokens
    monkeypatch.setattr("litellm.utils.get_token_count", Mock(return_value=80))

    result = trim_messages(messages, model=model_name, max_tokens=provided_max_tokens)
    assert result == messages  # No change because no trimming was required


def test_trim_messages_invalid_model():
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError):
        trim_messages(messages)  # Neither `max_tokens` nor valid `model` is provided


@pytest.mark.parametrize("return_response_tokens", [True, False])
def test_trim_messages_trimming_required(monkeypatch, return_response_tokens):
    messages = [
        {"role": "user", "content": "A" * 100},
        {"role": "system", "content": "B" * 50},
    ]

    monkeypatch.setattr("litellm.utils.get_token_count", Mock(return_value=200))

    # Mock process_system_message to handle system message
    mock_system_message_event = {"role": "system", "content": "Trimmed system message"}
    monkeypatch.setattr(
        "litellm.utils.process_system_message",
        Mock(return_value=(mock_system_message_event, 50)),
    )

    # Mock process_messages to simulate the trimming process
    trimmed_messages = [{"role": "user", "content": "A" * 50}]
    monkeypatch.setattr(
        "litellm.utils.process_messages", Mock(return_value=trimmed_messages)
    )

    result = trim_messages(
        messages, max_tokens=100, return_response_tokens=return_response_tokens
    )
    if return_response_tokens:
        assert isinstance(result, tuple)
        assert result[0] == trimmed_messages
        assert isinstance(result[1], int)
    else:
        assert result == trimmed_messages
