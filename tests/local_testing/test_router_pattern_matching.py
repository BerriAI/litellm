"""
This tests the pattern matching router

Pattern matching router is used to match patterns like openai/*, vertex_ai/*, anthropic/* etc. (wildcard matching)
"""


from dotenv import load_dotenv

from litellm import Router

load_dotenv()


# Add this test to check for exception handling

def test_pattern_matching_router_with_default_wildcard():
    """
    Tests that the router returns the default wildcard model when the pattern is not found

    Make sure generic '*' allows all models to be passed through.
    """
    router = Router(
        model_list=[
            {
                "model_name": "*",
                "litellm_params": {"model": "*"},
                "model_info": {"access_groups": ["default"]},
            },
            {
                "model_name": "anthropic-claude",
                "litellm_params": {"model": "anthropic/claude-3-5-sonnet"},
            },
        ]
    )

    assert len(router.pattern_router.patterns) > 0

    router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )
