"""
Test Bedrock AgentCore integration
"""

from dotenv import load_dotenv

load_dotenv()


import litellm
import pytest


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_13sf6-cALnp38iZD",  # non-streaming invocation
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",  # streaming invocation
    ],
)
def test_bedrock_agentcore_basic(model):
    """
    Test AgentCore invocation parameterized by model
    """
    litellm.turn_on_debug()
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "user", "content": "Explain machine learning in simple terms"}
        ],
    )
    print("response from agentcore=", response.model_dump_json(indent=4))
    # Assert that the message content has a response with some length
    assert response.choices[0].message.content
    assert len(response.choices[0].message.content) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_13sf6-cALnp38iZD",  # streaming invocation
    ],
)
async def test_bedrock_agentcore_with_streaming(model):
    """
    Test AgentCore with streaming
    """
    print("running streming test for model=", model)
    # litellm.turn_on_debug()
    response = await litellm.acompletion(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
        messages=[
            {
                "role": "user",
                "content": "Explain machine learning in simple terms",
            }
        ],
        stream=True,
    )

    async for chunk in response:
        print("chunk=", chunk)
