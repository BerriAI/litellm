"""Live e2e: Bedrock reached with the credentials the gateway's environment gives it.

A deployment whose litellm_params name no `aws_access_key_id` and no
`aws_secret_access_key` must still reach Bedrock, by resolving credentials
through boto3's ambient chain (`base_aws_llm.py` falls through to
`boto3.Session()`). That is how the gateway is deployed on AWS: an EKS Pod
Identity association, an IRSA role, or an EC2 instance profile supplies the
identity, and the config carries a region and nothing secret.

Nothing else in the suite covers that. Every other Bedrock test names
`os.environ/AWS_ACCESS_KEY_ID` and `os.environ/AWS_SECRET_ACCESS_KEY` in its
litellm_params, so all of them keep passing if the pod's own identity is
revoked, misassociated, or stripped of its Bedrock grant. This test is the one
that goes red, and it is deliberately the cheapest possible shape of that
signal: one short completion, no tools, no streaming.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import Success
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

BEDROCK_MODEL = "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"


class TestAmbientAwsCredentials:
    @pytest.mark.covers(
        "other.auth.aws_ambient_credentials.resolves_without_static_keys",
        exercised_on=["chat_completions"],
    )
    def test_bedrock_deployment_without_static_keys_completes(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        """A key-less Bedrock deployment answers a real completion.

        `aws_region_name` is the only AWS field set. Adding either static
        credential here would defeat the test: boto3 prefers what it is handed,
        so the call would pass without the ambient identity ever being consulted.
        """
        model = f"e2e-bedrock-ambient-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model=BEDROCK_MODEL, aws_region_name="os.environ/AWS_REGION"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        result = proxy.chat(
            resources.key(),
            ChatBody(
                model=model,
                messages=[
                    ChatMessage(role="user", content=f"Reply with one word. {unique_marker()}")
                ],
                max_tokens=16,
            ),
        )

        match result:
            case Success(data=response):
                assert response.choices, (
                    f"Bedrock answered without a completion, so the request reached AWS but "
                    f"came back empty: {response}"
                )
            case failure:
                pytest.fail(
                    "Bedrock was unreachable using only the gateway's ambient AWS identity. "
                    "A credentials error here means the pod lost that identity (association "
                    "removed or repointed, or its role lost Bedrock); any other error means "
                    f"the deployment itself is wrong. Got: {failure}"
                )
