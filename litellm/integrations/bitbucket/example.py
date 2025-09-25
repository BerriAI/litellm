"""
Example usage of BitBucket prompt management with LiteLLM.

This example demonstrates how to use BitBucket repositories for team-based prompt management.
"""

import litellm
from litellm.integrations.bitbucket import (
    BitBucketPromptManager,
    set_global_bitbucket_config,
)


def example_basic_usage():
    """Basic example of using BitBucket prompt management."""

    # Configure BitBucket access
    bitbucket_config = {
        "workspace": "your-workspace",
        "repository": "your-prompt-repo",
        "access_token": "your-access-token",
        "branch": "main",  # optional, defaults to main
    }

    # Set global BitBucket configuration
    set_global_bitbucket_config(bitbucket_config)

    # Use with LiteLLM completion
    response = litellm.completion(
        model="bitbucket/gpt-4",  # The actual model comes from the .prompt file
        prompt_id="chat_assistant",  # Name of the .prompt file without extension
        prompt_variables={
            "user_message": "What is machine learning?",
            "system_context": "You are a helpful AI tutor.",
        },
    )

    print(response.choices[0].message.content)


def example_with_custom_config():
    """Example using custom BitBucket configuration per request."""

    # Use with custom BitBucket configuration
    response = litellm.completion(
        model="bitbucket/gpt-4",
        prompt_id="code_reviewer",
        prompt_variables={
            "code": "def hello():\n    print('Hello, World!')",
            "language": "Python",
        },
        bitbucket_config={
            "workspace": "team-a",
            "repository": "prompts",
            "access_token": "team-a-token",
            "branch": "production",
        },
    )

    print(response.choices[0].message.content)


def example_direct_manager_usage():
    """Example using BitBucketPromptManager directly."""

    # Initialize the manager
    config = {
        "workspace": "your-workspace",
        "repository": "your-repo",
        "access_token": "your-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="my_prompt")

    # Get available prompts
    prompts = manager.get_available_prompts()
    print(f"Available prompts: {prompts}")

    # Render a template directly
    rendered = manager.render_template("my_prompt", {"variable": "value"})
    print(f"Rendered prompt: {rendered}")


def example_proxy_configuration():
    """Example configuration for LiteLLM proxy server."""

    # config.yaml for proxy server
    config_yaml = """
model_list:
  - model_name: my-bitbucket-model
    litellm_params:
      model: bitbucket/gpt-4
      prompt_id: "hello"
      bitbucket_config:
        workspace: "your-workspace"
        repository: "your-repo"
        access_token: "your-access-token"
        branch: "main"
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  global_bitbucket_config:
    workspace: "your-workspace"
    repository: "your-repo"
    access_token: "your-access-token"
    branch: "main"
"""

    print("Proxy configuration:")
    print(config_yaml)


def example_team_based_access():
    """Example of team-based access control."""

    # Team A configuration
    team_a_config = {
        "workspace": "team-a",
        "repository": "team-a-prompts",
        "access_token": "team-a-token",
    }

    # Team B configuration
    team_b_config = {
        "workspace": "team-b",
        "repository": "team-b-prompts",
        "access_token": "team-b-token",
    }

    # Use different configurations for different teams
    team_a_response = litellm.completion(
        model="bitbucket/gpt-4",
        prompt_id="team_a_prompt",
        bitbucket_config=team_a_config,
        prompt_variables={"message": "Hello from Team A"},
    )

    team_b_response = litellm.completion(
        model="bitbucket/gpt-4",
        prompt_id="team_b_prompt",
        bitbucket_config=team_b_config,
        prompt_variables={"message": "Hello from Team B"},
    )

    print(f"Team A response: {team_a_response.choices[0].message.content}")
    print(f"Team B response: {team_b_response.choices[0].message.content}")


def example_error_handling():
    """Example of error handling with BitBucket integration."""

    try:
        response = litellm.completion(
            model="bitbucket/gpt-4",
            prompt_id="nonexistent_prompt",
            bitbucket_config={
                "workspace": "test-workspace",
                "repository": "test-repo",
                "access_token": "test-token",
            },
        )
    except Exception as e:
        print(f"Error: {e}")
        # Handle different types of errors
        if "Access denied" in str(e):
            print("Check your BitBucket permissions")
        elif "Authentication failed" in str(e):
            print("Check your access token")
        elif "not found" in str(e):
            print("Prompt file not found in repository")


if __name__ == "__main__":
    print("BitBucket Prompt Management Examples")
    print("=" * 40)

    # Note: These examples require actual BitBucket repositories and access tokens
    # Uncomment and modify the examples below to test with your setup

    # example_basic_usage()
    # example_with_custom_config()
    # example_direct_manager_usage()
    # example_proxy_configuration()
    # example_team_based_access()
    # example_error_handling()

    print("\nTo use these examples:")
    print("1. Create a BitBucket repository with .prompt files")
    print("2. Set up access tokens with appropriate permissions")
    print("3. Update the configuration with your workspace and repository details")
    print("4. Uncomment and run the desired examples")
