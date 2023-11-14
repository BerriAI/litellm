import sys, os
from dotenv import load_dotenv
import copy

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.utils import trim_messages, get_token_count, get_valid_models, check_valid_key, validate_environment, function_to_dict, token_counter

# Assuming your trim_messages, shorten_message_to_fit_limit, and get_token_count functions are all in a module named 'message_utils'

# Test 1: Check trimming of normal message
def test_basic_trimming():
    messages = [{"role": "user", "content": "This is a long message that definitely exceeds the token limit."}]
    trimmed_messages = trim_messages(messages, model="claude-2", max_tokens=8)
    print("trimmed messages")
    print(trimmed_messages)
    # print(get_token_count(messages=trimmed_messages, model="claude-2"))
    assert (get_token_count(messages=trimmed_messages, model="claude-2")) <= 8
# test_basic_trimming()

def test_basic_trimming_no_max_tokens_specified():
    messages = [{"role": "user", "content": "This is a long message that is definitely under the token limit."}]
    trimmed_messages = trim_messages(messages, model="gpt-4")
    print("trimmed messages for gpt-4")
    print(trimmed_messages)
    # print(get_token_count(messages=trimmed_messages, model="claude-2"))
    assert (get_token_count(messages=trimmed_messages, model="gpt-4")) <= litellm.model_cost['gpt-4']['max_tokens']
# test_basic_trimming_no_max_tokens_specified()

def test_multiple_messages_trimming():
    messages = [
        {"role": "user", "content": "This is a long message that will exceed the token limit."},
        {"role": "user", "content": "This is another long message that will also exceed the limit."}
    ]
    trimmed_messages = trim_messages(messages=messages, model="gpt-3.5-turbo", max_tokens=20)
    # print(get_token_count(messages=trimmed_messages, model="gpt-3.5-turbo"))
    assert(get_token_count(messages=trimmed_messages, model="gpt-3.5-turbo")) <= 20
# test_multiple_messages_trimming()

def test_multiple_messages_no_trimming():
    messages = [
        {"role": "user", "content": "This is a long message that will exceed the token limit."},
        {"role": "user", "content": "This is another long message that will also exceed the limit."}
    ]
    trimmed_messages = trim_messages(messages=messages, model="gpt-3.5-turbo", max_tokens=100)
    print("Trimmed messages")
    print(trimmed_messages)
    assert(messages==trimmed_messages)

# test_multiple_messages_no_trimming()


def test_large_trimming_multiple_messages():
    messages = [{"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."}, {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},{"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},{"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},{"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."}]
    trimmed_messages = trim_messages(messages, max_tokens=20, model="gpt-4-0613")
    print("trimmed messages")
    print(trimmed_messages)
    assert(get_token_count(messages=trimmed_messages, model="gpt-4-0613")) <= 20
# test_large_trimming()

def test_large_trimming_single_message():
    messages = [{"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."}]
    trimmed_messages = trim_messages(messages, max_tokens=5, model="gpt-4-0613")
    assert(get_token_count(messages=trimmed_messages, model="gpt-4-0613")) <= 5
    assert(get_token_count(messages=trimmed_messages, model="gpt-4-0613")) > 0


def test_trimming_with_system_message_within_max_tokens():
    # This message is 33 tokens long
    messages = [{"role": "system", "content": "This is a short system message"}, {"role": "user", "content": "This is a medium normal message, let's say litellm is awesome."}]
    trimmed_messages = trim_messages(messages, max_tokens=30, model="gpt-4-0613") # The system message should fit within the token limit
    assert len(trimmed_messages) == 2
    assert trimmed_messages[0]["content"] == "This is a short system message"


def test_trimming_with_system_message_exceeding_max_tokens():
    # This message is 33 tokens long. The system message is 13 tokens long.
    messages = [{"role": "system", "content": "This is a short system message"}, {"role": "user", "content": "This is a medium normal message, let's say litellm is awesome."}]
    trimmed_messages = trim_messages(messages, max_tokens=12, model="gpt-4-0613")
    assert len(trimmed_messages) == 1
    assert '..' in trimmed_messages[0]["content"]

def test_trimming_should_not_change_original_messages():
    messages = [{"role": "system", "content": "This is a short system message"}, {"role": "user", "content": "This is a medium normal message, let's say litellm is awesome."}]
    messages_copy = copy.deepcopy(messages)
    trimmed_messages = trim_messages(messages, max_tokens=12, model="gpt-4-0613")
    assert(messages==messages_copy)

def test_get_valid_models():
    old_environ = os.environ
    os.environ = {'OPENAI_API_KEY': 'temp'} # mock set only openai key in environ

    valid_models = get_valid_models()
    print(valid_models)

    # list of openai supported llms on litellm
    expected_models = litellm.open_ai_chat_completion_models + litellm.open_ai_text_completion_models
    
    assert(valid_models == expected_models)

    # reset replicate env key
    os.environ = old_environ

# test_get_valid_models()

def test_bad_key():
    key = "bad-key"
    response = check_valid_key(model="gpt-3.5-turbo", api_key=key)
    print(response, key)
    assert(response == False)

def test_good_key():
    key = os.environ['OPENAI_API_KEY']
    response = check_valid_key(model="gpt-3.5-turbo", api_key=key)
    assert(response == True)

# test validate environment 

def test_validate_environment_empty_model():
    api_key = validate_environment()
    if api_key is None:
        raise Exception() 

# test_validate_environment_empty_model()

def test_function_to_dict():
    print("testing function to dict for get current weather")
    def get_current_weather(location: str, unit: str):
        """Get the current weather in a given location

        Parameters
        ----------
        location : str
            The city and state, e.g. San Francisco, CA
        unit : {'celsius', 'fahrenheit'}
            Temperature unit

        Returns
        -------
        str
            a sentence indicating the weather
        """
        if location == "Boston, MA":
            return "The weather is 12F"
    function_json = litellm.utils.function_to_dict(get_current_weather)
    print(function_json)

    expected_output = {
        'name': 'get_current_weather', 
        'description': 'Get the current weather in a given location', 
        'parameters': {
            'type': 'object', 
            'properties': {
                'location': {'type': 'string', 'description': 'The city and state, e.g. San Francisco, CA'}, 
                'unit': {'type': 'string', 'description': 'Temperature unit', 'enum': "['fahrenheit', 'celsius']"}
            }, 
            'required': ['location', 'unit']
        }
    }
    print(expected_output)
    
    assert function_json['name'] == expected_output["name"]
    assert function_json["description"] == expected_output["description"]
    assert function_json["parameters"]["type"] == expected_output["parameters"]["type"]
    assert function_json["parameters"]["properties"]["location"] == expected_output["parameters"]["properties"]["location"]

    # the enum can change it can be - which is why we don't assert on unit
    # {'type': 'string', 'description': 'Temperature unit', 'enum': "['fahrenheit', 'celsius']"}
    # {'type': 'string', 'description': 'Temperature unit', 'enum': "['celsius', 'fahrenheit']"}

    assert function_json["parameters"]["required"] == expected_output["parameters"]["required"]

    print("passed")
# test_function_to_dict()


def test_token_counter():
    try:
        messages = [
            {
                "role": "user",
                "content": "hi how are you what time is it"
            }
        ]
        tokens = token_counter(
            model = "gpt-3.5-turbo",
            messages=messages
        )
        print("gpt-35-turbo")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(
            model = "claude-2",
            messages=messages
        )
        print("claude-2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(
            model = "palm/chat-bison",
            messages=messages
        )
        print("palm/chat-bison")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(
            model = "ollama/llama2",
            messages=messages
        )
        print("ollama/llama2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(
            model = "anthropic.claude-instant-v1",
            messages=messages
        )
        print("anthropic.claude-instant-v1")
        print(tokens)
        assert tokens > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
test_token_counter()





