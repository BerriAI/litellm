import uuid
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.database import read_rows


def _credential_with_alias(gateway: Gateway, scenario: Scenario, alias: str) -> str:
    name: Final = f"credential-{uuid.uuid4().hex}"
    gateway.post(
        "/credentials",
        {
            "credential_name": name,
            "credential_alias": alias,
            "credential_values": {"api_key": "synthetic-credential"},
            "credential_info": {"custom_llm_provider": "openai"},
        },
    )
    scenario.cleanups.callback(_delete_credential_if_present, gateway, name)
    return name


def _delete_credential_if_present(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"/credentials/{name}")
    assert response.status_code in (200, 404), response.text


def _stored_alias(name: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT credential_name, credential_alias FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s',
        (name,),
    )


def _served_credential(gateway: Gateway, name: str) -> dict[str, JsonValue]:
    by_name: Final = gateway.get(f"/credentials/by_name/{name}")
    listed: Final = [entry for entry in gateway.get("/credentials")["credentials"] if isinstance(entry, dict)]
    from_list: Final = [entry for entry in listed if entry["credential_name"] == name]
    assert len(from_list) == 1, listed
    assert from_list[0]["credential_alias"] == by_name["credential_alias"], (from_list[0], by_name)
    return by_name


def _model_using(gateway: Gateway, scenario: Scenario, credential: str) -> str:
    return scenario.model(
        model="openai/gpt-4o-mini",
        api_base=f"{gateway.upstream_url}/v1",
        litellm_credential_name=credential,
    )


def test_alias_round_trips_and_patch_keeps_clears_and_rejects_blank(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = _credential_with_alias(gateway, scenario, "Prod OpenAI")
        assert _stored_alias(name) == [{"credential_name": name, "credential_alias": "Prod OpenAI"}]
        assert _served_credential(gateway, name)["credential_alias"] == "Prod OpenAI"

        kept: Final = gateway.request("PATCH", f"/credentials/{name}", {"credential_info": {"description": "d"}})
        assert kept.status_code == 200, kept.text
        assert _served_credential(gateway, name)["credential_alias"] == "Prod OpenAI"

        renamed_alias: Final = gateway.request(
            "PATCH", f"/credentials/{name}", {"credential_alias": "Staging OpenAI", "credential_info": {}}
        )
        assert renamed_alias.status_code == 200, renamed_alias.text
        assert _stored_alias(name) == [{"credential_name": name, "credential_alias": "Staging OpenAI"}]
        assert _served_credential(gateway, name)["credential_alias"] == "Staging OpenAI"

        blank: Final = gateway.request(
            "PATCH", f"/credentials/{name}", {"credential_alias": "  ", "credential_info": {}}
        )
        assert blank.status_code == 400, blank.text
        assert _stored_alias(name) == [{"credential_name": name, "credential_alias": "Staging OpenAI"}]

        cleared: Final = gateway.request(
            "PATCH", f"/credentials/{name}", {"credential_alias": None, "credential_info": {}}
        )
        assert cleared.status_code == 200, cleared.text
        assert _stored_alias(name) == [{"credential_name": name, "credential_alias": None}]
        assert _served_credential(gateway, name)["credential_alias"] is None


def test_blank_alias_on_create_is_rejected(gateway: Gateway) -> None:
    name: Final = f"credential-{uuid.uuid4().hex}"
    rejected: Final = gateway.request(
        "POST",
        "/credentials",
        {"credential_name": name, "credential_alias": "", "credential_values": {"api_key": "k"}, "credential_info": {}},
    )
    assert rejected.status_code == 400, rejected.text
    assert _stored_alias(name) == []


def test_patch_rename_is_rejected_and_model_keeps_working(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = _credential_with_alias(gateway, scenario, "Prod OpenAI")
        model: Final = _model_using(gateway, scenario, name)
        assert gateway.chat(model)["object"] == "chat.completion"

        other: Final = f"credential-{uuid.uuid4().hex}"
        renamed: Final = gateway.request(
            "PATCH", f"/credentials/{name}", {"credential_name": other, "credential_info": {"description": "x"}}
        )
        assert renamed.status_code == 400, renamed.text
        assert "credential_alias" in renamed.text, renamed.text
        assert _stored_alias(name) == [{"credential_name": name, "credential_alias": "Prod OpenAI"}]
        assert _stored_alias(other) == []
        assert gateway.request("GET", f"/credentials/by_name/{other}").status_code == 404

        same_name: Final = gateway.request(
            "PATCH", f"/credentials/{name}", {"credential_name": name, "credential_info": {"description": "y"}}
        )
        assert same_name.status_code == 200, same_name.text
        assert _served_credential(gateway, name)["credential_info"] == {
            "custom_llm_provider": "openai",
            "description": "y",
        }
        assert gateway.chat(model)["object"] == "chat.completion"
