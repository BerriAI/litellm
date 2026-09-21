import json
import os
import uuid
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import pytest
import yaml

from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from integration.providers.test_bedrock_auth_wire import MODEL, RESPONSE


@pytest.mark.covers("other.provider_wire.bedrock.db_yaml_role_reference_reaches_sts_and_signed_request")
def test_role_reference_from_db_and_yaml_reaches_real_sts_http_and_bedrock(gateway: Gateway, tmp_path: Path) -> None:
    role: Final = "arn:aws:iam::123456789012:role/integration-" + uuid.uuid4().hex
    assumed_key: Final = "ASIAINTEGRATION000001"
    assumed_token: Final = "synthetic-assumed-session-token"

    def sts(request: Request) -> Reply:
        parameters: Final = parse_qs(request.body.decode())
        action: Final = parameters["Action"][0]
        assert request.method == "POST" and action in {"GetCallerIdentity", "AssumeRole"}
        if action == "GetCallerIdentity":
            result = "<GetCallerIdentityResult><Arn>arn:aws:iam::123456789012:user/integration-source</Arn><UserId>integration-source</UserId><Account>123456789012</Account></GetCallerIdentityResult>"
        else:
            assert parameters["RoleArn"] == [role]
            assert parameters["RoleSessionName"][0] in {"integration-yaml-session", "integration-db-session"}
            result = f"<AssumeRoleResult><Credentials><AccessKeyId>{assumed_key}</AccessKeyId><SecretAccessKey>synthetic-assumed-secret-key-for-testing</SecretAccessKey><SessionToken>{assumed_token}</SessionToken><Expiration>2035-01-01T00:00:00Z</Expiration></Credentials><AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/integration/session</Arn><AssumedRoleId>integration:session</AssumedRoleId></AssumedRoleUser><PackedPolicySize>0</PackedPolicySize></AssumeRoleResult>"
        return Reply(content_type="text/xml", body=f'<{action}Response xmlns="https://sts.amazonaws.com/doc/2011-06-15/">{result}<ResponseMetadata><RequestId>synthetic-sts-request</RequestId></ResponseMetadata></{action}Response>'.encode())

    def bedrock(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
        assert f"Credential={assumed_key}/" in request.headers["authorization"]
        assert request.headers["x-amz-security-token"] == assumed_token
        assert json.loads(request.body)["messages"][0]["content"][0]["text"] == "synthetic role request"
        return Reply(body=RESPONSE)

    with wire_server(sts) as authority, wire_server(bedrock) as provider:
        parameters: Final = {
            "model": MODEL, "aws_region_name": "us-east-1", "aws_role_name": "os.environ/INTEGRATION_ROLE_ARN",
            "aws_session_name": "integration-yaml-session", "aws_bedrock_runtime_endpoint": provider.url,
            "aws_sts_endpoint": authority.url,
        }
        alias: Final = "integration-role-yaml-" + uuid.uuid4().hex
        configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        configuration["model_list"] = [{"model_name": alias, "litellm_params": parameters, "model_info": {"id": alias}}]
        path: Final = tmp_path / "roles.yaml"
        path.write_text(yaml.safe_dump(configuration))
        empty: Final = tmp_path / "empty-aws-config"
        empty.write_text("")
        overrides: Final = {
            "INTEGRATION_ROLE_ARN": role, "AWS_ACCESS_KEY_ID": "AKIAINTEGRATION000001", "AWS_SECRET_ACCESS_KEY": "synthetic-source-secret-key-for-testing",
            "AWS_CONFIG_FILE": str(empty), "AWS_SHARED_CREDENTIALS_FILE": str(empty), "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_ENDPOINT_URL_STS": authority.url, "AWS_DEFAULT_REGION": "us-east-1", "LITELLM_RUST": "false",
        }
        with owned_proxy(gateway, tmp_path, overrides, config=path, remove_environment=tuple(name for name in os.environ if name.startswith("AWS_"))) as candidate, candidate.scenario() as scenario:
            database_model: Final = scenario.model(**{**parameters, "api_key": None, "aws_session_name": "integration-db-session"})
            for generation in range(2):
                for model in (alias, database_model):
                    response: Final = candidate.request("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "synthetic role request"}], "cache": {"no-cache": True}})
                    assert response.status_code == 200, response.text
                    assert response.json()["choices"][0]["message"]["content"] == "bedrock wire control"
                    assert response.json()["usage"]["total_tokens"] == 15
                    assert len(provider.drain()) == 1
                if generation == 0:
                    target: Final = next(entry for entry in candidate.get("/model/info")["data"] if entry["model_name"] == database_model)
                    response: Final = candidate.request("PATCH", f"/model/{target['model_info']['id']}/update", {"model_info": {"description": "role reload"}})
                    assert response.status_code == 200, response.text
            assumed: Final = tuple(parse_qs(request.body.decode()) for request in authority.drain() if parse_qs(request.body.decode())["Action"] == ["AssumeRole"])
            assert {entry["RoleSessionName"][0] for entry in assumed} == {"integration-yaml-session", "integration-db-session"}
            assert all(entry["RoleArn"] == [role] for entry in assumed)
