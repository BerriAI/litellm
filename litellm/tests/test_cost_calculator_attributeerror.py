import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from litellm.cost_calculator import completion_cost
from litellm.types.utils import Usage, ServerToolUse, ModelResponse

def test_usage_coercion_from_dict():
    """
    Verify that Usage correctly coerces server_tool_use from a dict to a ServerToolUse object.
    """
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=20,
        server_tool_use={"web_search_requests": 5}
    )
    
    assert isinstance(usage.server_tool_use, ServerToolUse)
    assert usage.server_tool_use.web_search_requests == 5

def test_completion_cost_with_dict_usage():
    """
    Verify that completion_cost handles a response where server_tool_use is a dict.
    This simulates the bug reported in #26153.
    """
    # Create a usage object and manually set server_tool_use to a dict to simulate the state after stream assembly
    usage = Usage(prompt_tokens=10, completion_tokens=20)
    usage.server_tool_use = {"web_search_requests": 5} # Manually bypass the __init__ coercion for testing defensive checks
    
    response = ModelResponse(
        id="test-id",
        choices=[{"message": {"role": "assistant", "content": "hello"}}],
        usage=usage
    )
    
    # This should not raise AttributeError
    cost = completion_cost(completion_response=response, model="gpt-3.5-turbo")
    assert cost is not None

if __name__ == "__main__":
    test_usage_coercion_from_dict()
    test_completion_cost_with_dict_usage()
