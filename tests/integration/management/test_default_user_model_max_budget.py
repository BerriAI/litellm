import json
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment, object_value
from tests.integration._support.process import OwnedState, owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

DEFAULT_BUDGET: Final[Mapping[str, JsonValue]] = {"gpt-4o": {"budget_limit": 5.0, "time_period": "30d"}}
EXPLICIT_BUDGET: Final[Mapping[str, JsonValue]] = {"gpt-4o-mini": {"budget_limit": 1.0, "time_period": "1d"}}
LICENSE_ERROR: Final = "You must have an enterprise license to set model_max_budget"


@contextmanager
def _settings_gateway(
    directory: Path,
    default_internal_user_params: Mapping[str, JsonValue],
    *,
    licensed: bool,
    general_settings: Mapping[str, JsonValue] = MappingProxyType({}),
    environment: Mapping[str, str] = MappingProxyType({}),
) -> Iterator[Gateway]:
    config: Final = directory / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    **general_settings,
                },
                "litellm_settings": (
                    {"default_internal_user_params": dict(default_internal_user_params)}
                    if default_internal_user_params
                    else {}
                ),
            }
        )
    )
    with (
        gateway_from_environment() as rig,
        owned_proxy(
            rig,
            directory,
            {"STORE_MODEL_IN_DB": "True", **environment},
            config=config,
            remove_environment=() if licensed else ("LITELLM_LICENSE",),
            state=OwnedState.PRIVATE,
        ) as gateway,
    ):
        yield gateway


def _new_user(gateway: Gateway, **fields: JsonValue) -> str:
    user_id: Final = f"default-budget-{uuid.uuid4().hex[:12]}"
    created: Final = gateway.request("POST", "/user/new", {"user_id": user_id, "auto_create_key": False, **fields})
    assert created.status_code == 200, created.text
    return user_id


def _stored_model_max_budget(gateway: Gateway, user_id: str) -> JsonValue:
    rows: Final = gateway.rows('SELECT model_max_budget FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    assert len(rows) == 1, rows
    return rows[0]["model_max_budget"]


def _internal_user_settings(gateway: Gateway) -> dict[str, JsonValue]:
    return gateway.get("/get/internal_user_settings")


def test_a_default_model_max_budget_set_through_the_settings_api_is_returned_and_given_to_new_users(
    tmp_path: Path,
) -> None:
    with _settings_gateway(tmp_path, {}, licensed=True) as gateway:
        updated: Final = gateway.request(
            "PATCH",
            "/update/internal_user_settings",
            {"user_role": "internal_user", "model_max_budget": DEFAULT_BUDGET},
        )
        assert updated.status_code == 200, updated.text
        assert object_value(updated.json()["settings"])["model_max_budget"] == DEFAULT_BUDGET, updated.text

        settings: Final = _internal_user_settings(gateway)
        assert object_value(settings["values"])["model_max_budget"] == DEFAULT_BUDGET, settings
        assert "model_max_budget" in object_value(object_value(settings["field_schema"])["properties"]), settings

        assert _stored_model_max_budget(gateway, _new_user(gateway, user_role="internal_user")) == DEFAULT_BUDGET
        assert (
            _stored_model_max_budget(
                gateway, _new_user(gateway, user_role="internal_user", model_max_budget=EXPLICIT_BUDGET)
            )
            == EXPLICIT_BUDGET
        )
        assert _stored_model_max_budget(gateway, _new_user(gateway, user_role="proxy_admin")) == {}


def test_a_config_file_default_model_max_budget_is_given_to_a_user_created_through_user_new(tmp_path: Path) -> None:
    with _settings_gateway(
        tmp_path, {"user_role": "internal_user", "model_max_budget": DEFAULT_BUDGET}, licensed=True
    ) as gateway:
        assert _stored_model_max_budget(gateway, _new_user(gateway, user_role="internal_user")) == DEFAULT_BUDGET
        assert (
            _stored_model_max_budget(
                gateway, _new_user(gateway, user_role="internal_user", model_max_budget=EXPLICIT_BUDGET)
            )
            == EXPLICIT_BUDGET
        )
        assert _stored_model_max_budget(gateway, _new_user(gateway, user_role="proxy_admin")) == {}


def test_a_default_model_max_budget_without_an_enterprise_license_is_rejected_and_not_saved(tmp_path: Path) -> None:
    with _settings_gateway(tmp_path, {}, licensed=False) as gateway:
        on_user_new: Final = gateway.request(
            "POST", "/user/new", {"user_role": "internal_user", "model_max_budget": DEFAULT_BUDGET}
        )
        assert LICENSE_ERROR in on_user_new.text, f"{on_user_new.status_code} {on_user_new.text}"

        rejected: Final = gateway.request(
            "PATCH",
            "/update/internal_user_settings",
            {"user_role": "internal_user", "model_max_budget": DEFAULT_BUDGET},
        )
        assert rejected.status_code == 400, f"{rejected.status_code} {rejected.text}"
        assert LICENSE_ERROR in rejected.text, rejected.text

        values: Final = object_value(_internal_user_settings(gateway)["values"])
        assert values.get("model_max_budget") is None, values
        assert values["user_role"] == "internal_user_viewer", values

        accepted: Final = gateway.request(
            "PATCH", "/update/internal_user_settings", {"user_role": "internal_user", "model_max_budget": {}}
        )
        assert accepted.status_code == 200, accepted.text


def test_a_malformed_default_model_max_budget_is_rejected_and_not_saved(tmp_path: Path) -> None:
    malformed: Final[Mapping[str, JsonValue]] = {"gpt-4o": {"budget_limit": "not-a-number"}}
    with _settings_gateway(tmp_path, {}, licensed=True) as gateway:
        on_user_new: Final = gateway.request(
            "POST", "/user/new", {"user_role": "internal_user", "model_max_budget": malformed}
        )
        assert "Invalid model_max_budget" in on_user_new.text, f"{on_user_new.status_code} {on_user_new.text}"

        rejected: Final = gateway.request(
            "PATCH", "/update/internal_user_settings", {"user_role": "internal_user", "model_max_budget": malformed}
        )
        assert rejected.status_code == 400, f"{rejected.status_code} {rejected.text}"
        assert "Invalid model_max_budget" in rejected.text, rejected.text

        values: Final = object_value(_internal_user_settings(gateway)["values"])
        assert values.get("model_max_budget") is None, values
        assert values["user_role"] == "internal_user_viewer", values


def test_a_config_file_default_model_max_budget_without_an_enterprise_license_does_not_block_user_new(
    tmp_path: Path,
) -> None:
    with _settings_gateway(
        tmp_path, {"user_role": "internal_user", "model_max_budget": DEFAULT_BUDGET}, licensed=False
    ) as gateway:
        assert _stored_model_max_budget(gateway, _new_user(gateway, user_role="internal_user")) == {}


@pytest.mark.parametrize(
    ("licensed", "status", "stored"),
    [
        (True, 200, [{"user_role": "internal_user", "max_budget": 7.0, "model_max_budget": DEFAULT_BUDGET}]),
        (False, 403, []),
    ],
    ids=["licensed", "no-enterprise-license"],
)
def test_a_jwt_sign_in_stores_the_default_model_max_budget_with_a_license_and_creates_no_user_without_one(
    tmp_path: Path, licensed: bool, status: int, stored: Sequence[Mapping[str, JsonValue]]
) -> None:
    key_id: Final = "integration-jwt-signing-key"
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    subject: Final = f"default-budget-jwt-{uuid.uuid4().hex[:12]}"
    token: Final = jwt.encode(
        {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
        private_key,
        algorithm="RS256",
        headers={"kid": key_id},
    )

    def jwks(_: Request) -> Reply:
        return Reply(body=json.dumps({"keys": [{**public_jwk, "kid": key_id}]}).encode())

    with (
        wire_server(jwks) as identity_provider,
        _settings_gateway(
            tmp_path,
            {"user_role": "internal_user", "max_budget": 7.0, "model_max_budget": DEFAULT_BUDGET},
            licensed=licensed,
            general_settings={
                "enable_jwt_auth": True,
                "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True},
            },
            environment={"JWT_PUBLIC_KEY_URL": identity_provider.url},
        ) as gateway,
    ):
        signed_in: Final = gateway.request("GET", "/v1/models", key=token)

        assert signed_in.status_code == status, signed_in.text
        assert (
            gateway.rows(
                'SELECT user_role, max_budget, model_max_budget FROM "LiteLLM_UserTable" WHERE user_id = %s',
                (subject,),
            )
            == stored
        )
