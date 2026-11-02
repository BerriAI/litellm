import json
import os
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

MODEL: Final = "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
ACCESS_KEY: Final = "AKIAINTEGRATION000002"
FORWARDED_FOR: Final = "203.0.113.5"
RESPONSE: Final = json.dumps(
    {"results": [{"index": 1, "relevanceScore": 0.9}, {"index": 0, "relevanceScore": 0.1}]}
).encode()


def signed_headers(authorization: str) -> tuple[str, ...]:
    return tuple(authorization.split("SignedHeaders=")[1].split(",")[0].split(";"))


def rerank_peer(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/rerank"
    assert request.headers["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/")
    assert signed_headers(request.headers["authorization"]) == ("content-type", "host", "x-amz-date"), request.headers[
        "authorization"
    ]
    assert request.headers["x-forwarded-for"] == FORWARDED_FOR
    body: Final = json.loads(request.body)
    assert body["queries"] == [{"textQuery": {"text": "synthetic rerank query"}, "type": "TEXT"}]
    assert body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"] == {
        "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    }
    assert body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["numberOfResults"] == 2
    return Reply(body=RESPONSE)


@pytest.mark.covers("providers.bedrock_rerank.forwarded_client_headers_are_sent_unsigned")
def test_forwarded_client_header_on_rerank_is_excluded_from_the_sigv4_signature(
    gateway: Gateway, tmp_path: Path
) -> None:
    empty: Final = tmp_path / "empty-aws-config"
    empty.write_text("")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["general_settings"]["forward_client_headers_to_llm_api"] = True
    path: Final = tmp_path / "forwarding.yaml"
    path.write_text(yaml.safe_dump(configuration))
    overrides: Final = {
        "AWS_CONFIG_FILE": str(empty),
        "AWS_SHARED_CREDENTIALS_FILE": str(empty),
        "AWS_EC2_METADATA_DISABLED": "true",
        "LITELLM_RUST": "false",
    }
    with wire_server(rerank_peer) as wire:
        with (
            owned_proxy(
                gateway,
                tmp_path,
                overrides,
                config=path,
                remove_environment=tuple(name for name in os.environ if name.startswith("AWS_")),
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=MODEL,
                api_key=None,
                api_base=None,
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire.url,
                aws_access_key_id=ACCESS_KEY,
                aws_secret_access_key="synthetic-rerank-secret-key-for-testing",
            )
            response: Final = candidate.request(
                "POST",
                "/v1/rerank",
                {
                    "model": model,
                    "query": "synthetic rerank query",
                    "documents": ["first synthetic document", "second synthetic document"],
                    "top_n": 2,
                },
                headers={"x-forwarded-for": FORWARDED_FOR},
            )
            assert response.status_code == 200, response.text
            assert response.json()["results"] == [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ], response.text
            assert len(wire.drain()) == 1
