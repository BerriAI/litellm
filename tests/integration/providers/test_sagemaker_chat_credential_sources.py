import os
import signal
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._sagemaker_chat_container import (
    ACCESS_KEY,
    ENDPOINT,
    JSON_OBJECT,
    NO_CACHE,
    REGION,
    SECRET_KEY,
    answered_choices,
    assert_signed,
    by_marker,
    container_body,
    container_reply,
    marked_prompt,
    marker,
)
from pydantic import JsonValue

_ROLE: Final = "arn:aws:iam::123456789012:role/integration-sagemaker-chat"
_SOURCE_KEY: Final = "AKIAINTEGRATIONSRC001"
_ENV_KEY: Final = "AKIAINTEGRATIONENV001"
_PROFILE_KEY: Final = "AKIAINTEGRATIONPRO001"
_ASSUMED_KEY: Final = "ASIAINTEGRATIONROLE01"
_WEB_IDENTITY_KEY: Final = "ASIAINTEGRATIONWEB001"
_PROFILE: Final = "integration-sagemaker-profile"
_WEB_IDENTITY_VARIABLE: Final = "INTEGRATION_WIF_IDENTITY_TOKEN"
_STARTED_WORKER: Final = "Started server process ["
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
_SIGNERS: Final[Mapping[str, tuple[str, str | None]]] = MappingProxyType(
    {
        "static": (ACCESS_KEY, None),
        "env": (_ENV_KEY, None),
        "profile": (_PROFILE_KEY, None),
        "role": (_ASSUMED_KEY, f"synthetic-session-{_ASSUMED_KEY}"),
        "web-identity": (_WEB_IDENTITY_KEY, f"synthetic-session-{_WEB_IDENTITY_KEY}"),
    }
)


def _sts_reply(action: str, result: str) -> Reply:
    return Reply(
        content_type="text/xml",
        body=(
            f'<{action}Response xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><{action}Result>{result}'
            f"</{action}Result><ResponseMetadata><RequestId>synthetic-sts-request</RequestId></ResponseMetadata>"
            f"</{action}Response>"
        ).encode(),
    )


def _assumed(action: str, key: str) -> Reply:
    return _sts_reply(
        action,
        f"<Credentials><AccessKeyId>{key}</AccessKeyId><SecretAccessKey>synthetic-assumed-secret-for-testing"
        f"</SecretAccessKey><SessionToken>synthetic-session-{key}</SessionToken><Expiration>2035-01-01T00:00:00Z"
        "</Expiration></Credentials><AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/integration/session"
        "</Arn><AssumedRoleId>integration:session</AssumedRoleId></AssumedRoleUser><PackedPolicySize>0"
        "</PackedPolicySize>",
    )


def _sts(request: Request) -> Reply:
    parameters: Final = parse_qs(request.body.decode())
    action: Final = parameters["Action"][0]
    assert request.method == "POST", request
    match action:
        case "GetCallerIdentity":
            return _sts_reply(
                action,
                "<Arn>arn:aws:iam::123456789012:user/integration-source</Arn><UserId>integration-source</UserId>"
                "<Account>123456789012</Account>",
            )
        case "AssumeRole":
            assert parameters["RoleArn"] == [_ROLE], parameters
            return _assumed(action, _ASSUMED_KEY)
        case "AssumeRoleWithWebIdentity":
            assert parameters["WebIdentityToken"] == ["synthetic.web-identity.token"], parameters
            return _assumed(action, _WEB_IDENTITY_KEY)
        case _:
            raise AssertionError(parameters)


def _container(request: Request) -> Reply:
    return container_reply(f"sagemaker-chat-{marker(request.body.decode())}", False)


def _config(container: Wire, authority: Wire, tmp_path: Path) -> Path:
    common: Final = {
        "model": f"sagemaker_chat/{ENDPOINT}",
        "aws_region_name": REGION,
        "sagemaker_base_url": container.url,
    }
    role: Final = {"aws_role_name": _ROLE, "aws_session_name": "integration-session", "aws_sts_endpoint": authority.url}
    sources: Final[Mapping[str, Mapping[str, str]]] = {
        "static": {"aws_access_key_id": ACCESS_KEY, "aws_secret_access_key": SECRET_KEY},
        "env": {},
        "profile": {"aws_profile_name": _PROFILE},
        "role": {"aws_access_key_id": _SOURCE_KEY, "aws_secret_access_key": SECRET_KEY, **role},
        "web-identity": {"aws_web_identity_token": f"oidc/env/{_WEB_IDENTITY_VARIABLE}", **role},
    }
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = tmp_path / "sagemaker-chat-credential-sources.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "model_list": [
                    {"model_name": f"sagemaker-chat-{source}", "litellm_params": {**common, **extra}}
                    for source, extra in sources.items()
                ],
            }
        )
    )
    return path


def _environment(authority: Wire, tmp_path: Path) -> Mapping[str, str]:
    credentials: Final = tmp_path / "aws-credentials"
    credentials.write_text(f"[{_PROFILE}]\naws_access_key_id = {_PROFILE_KEY}\naws_secret_access_key = {SECRET_KEY}\n")
    empty: Final = tmp_path / "aws-config"
    empty.write_text("")
    return {
        "AWS_ACCESS_KEY_ID": _ENV_KEY,
        "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
        "AWS_SHARED_CREDENTIALS_FILE": str(credentials),
        "AWS_CONFIG_FILE": str(empty),
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_ENDPOINT_URL_STS": authority.url,
        "AWS_DEFAULT_REGION": REGION,
        _WEB_IDENTITY_VARIABLE: "synthetic.web-identity.token",
    }


def _ask(owned: OwnedProxy, source: str) -> str:
    prompt: Final = marked_prompt()
    body: Final[Mapping[str, JsonValue]] = {
        "model": f"sagemaker-chat-{source}",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        **NO_CACHE,
    }
    with httpx.Client(base_url=owned.gateway.client.base_url, timeout=60, trust_env=False) as client:
        response: Final = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {owned.gateway.key}"}
        )
    assert response.status_code == 200, (source, response.text)
    assert JSON_OBJECT.validate_json(response.content)["choices"] == answered_choices(), (source, response.text)
    return prompt


def _assert_each_source_signed_its_own_clean_body(container: Wire, prompts: Mapping[str, str]) -> None:
    forwarded: Final = by_marker(container.drain())
    assert set(forwarded) == {marker(prompt) for prompt in prompts.values()}, sorted(forwarded)
    for source, prompt in prompts.items():
        _assert_forwarded(forwarded[marker(prompt)], container, source, prompt)


def _assert_forwarded(request: Request, container: Wire, source: str, prompt: str) -> None:
    assert JSON_OBJECT.validate_json(request.body) == container_body(container.url, prompt, stream=False), (
        source,
        request.body,
    )
    signer, session_token = _SIGNERS[source]
    assert_signed(request, signer, REGION, session_token)


def _worker_pids(log: str) -> tuple[int, ...]:
    return tuple(
        int(line.partition(_STARTED_WORKER)[2].partition("]")[0])
        for line in log.splitlines()
        if _STARTED_WORKER in line
    )


def _dead(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_every_aws_credential_source_keeps_its_credentials_out_of_the_body_when_a_worker_is_killed(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_container) as container, wire_server(_sts) as authority:
        with owned_proxy_process(
            gateway,
            tmp_path,
            _environment(authority, tmp_path),
            config=_config(container, authority, tmp_path),
            remove_environment=tuple(name for name in os.environ if name.startswith("AWS_")),
            workers=2,
        ) as owned:
            workers: Final = eventually(
                lambda: _worker_pids(owned.log.read_text()), lambda pids: len(pids) == 2, seconds=30
            )
            _assert_each_source_signed_its_own_clean_body(
                container, {source: _ask(owned, source) for source in _SIGNERS}
            )
            victim: Final = psutil.Process(workers[0])
            victim.send_signal(signal.SIGKILL)
            eventually(lambda: _dead(victim), bool, seconds=graceful_stop_seconds())
            _assert_each_source_signed_its_own_clean_body(
                container, {source: _ask(owned, source) for source in _SIGNERS}
            )
            assert psutil.Process(workers[1]).is_running(), workers
            assert owned.process.poll() is None, owned.log.read_text()
