import json
import pytest
import litellm.llms.sap.credentials as sap_credentials

mock_sap_service_key_dict = {
    "serviceurls":
        {
            "AI_API_URL":"https://testurl.hana.ondemand.com/"
        },
    "clientid":"mockclientid",
    "clientsecret":"mockclientsecret",
    "url":"https://test.sap.hana.ondemand.com/"
}

mock_wrapped_sap_service_key_dict = {
    "credentials": {
        "serviceurls":
            {
                "AI_API_URL":"https://testurl.hana.ondemand.com/"
            },
        "clientid":"mockclientid",
        "clientsecret":"mockclientsecret",
        "url":"https://test.sap.hana.ondemand.com/"
    }
}

expected_creds = {'client_id': "mockclientid",
                  'client_secret': "mockclientsecret",
                  'auth_url': 'https://test.sap.hana.ondemand.com/oauth/token',
                  'base_url': 'https://testurl.hana.ondemand.com/v2',
                  'resource_group': 'default'}

mock_sap_vcap_service_key_dict = {
    'aicore': [{
        'label': 'aicore',
        'name': 'aicore-instance',
        'instance_guid': '53ad5b47-a49a-4fec-9f0b-cd921c00b828',
        'credentials': {
            'serviceurls': {
                'AI_API_URL': 'vcap-api-url'
            },
            'url': 'vcap-auth-url',
            'clientid': 'vcap-clientid',
            'clientsecret': 'vcap-clientsecret'
        }
    }]
}
def _prep_env(monkeypatch):
    for var in ("AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET", "AICORE_AUTH_URL", "AICORE_RESOURCE_GROUP",
                "AICORE_BASE_URL", "AICORE_CERT_URL", "AICORE_SERVICE_KEY", "VCAP_SERVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AICORE_HOME", 'notexist')
    monkeypatch.setattr('litellm.sap_service_key', None)

def test_sap_fetch_creds_from_env_service_key(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("AICORE_SERVICE_KEY", json.dumps(mock_sap_service_key_dict))
    creds = sap_credentials.fetch_credentials()
    assert creds == expected_creds

def test_sap_fetch_creds_from_env_wrapped_service_key(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("AICORE_SERVICE_KEY", json.dumps(mock_wrapped_sap_service_key_dict))
    creds = sap_credentials.fetch_credentials()
    assert creds == expected_creds

def test_sap_fetch_creds_from_arg_service_key(monkeypatch):
    _prep_env(monkeypatch)
    creds = sap_credentials.fetch_credentials(service_key=json.dumps(mock_sap_service_key_dict))
    assert creds == expected_creds

def test_fetch_creds_from_env_vcap_service(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("VCAP_SERVICES", json.dumps(mock_sap_vcap_service_key_dict))
    creds = sap_credentials.fetch_credentials()
    assert creds['client_id'] == "vcap-clientid"
    assert creds['client_secret'] == "vcap-clientsecret"
    assert creds['auth_url'] == "vcap-auth-url/oauth/token"
    assert creds['base_url'] == "vcap-api-url/v2"
    assert creds['resource_group'] == "default"

def test_fetch_creds_from_env(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("AICORE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("AICORE_AUTH_URL", "env-auth-url")
    monkeypatch.setenv("AICORE_BASE_URL", "env-base-url")
    monkeypatch.setenv("AICORE_RESOURCE_GROUP", "env-resource-group")

    creds = sap_credentials.fetch_credentials()

    assert creds['client_id'] == "env-client-id"
    assert creds['client_secret'] == "env-client-secret"
    assert creds['auth_url'] == "env-auth-url/oauth/token"
    assert creds['base_url'] == "env-base-url/v2"
    assert creds['resource_group'] == "env-resource-group"

def test_creds_priority_order(monkeypatch):
    _prep_env(monkeypatch)
    monkeypatch.setenv("AICORE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("AICORE_AUTH_URL", "env-auth-url")
    monkeypatch.setenv("AICORE_BASE_URL", "env-base-url")
    monkeypatch.setenv("AICORE_RESOURCE_GROUP", "env-resource-group")
    creds = sap_credentials.fetch_credentials(service_key=json.dumps(mock_sap_service_key_dict))
    assert creds['client_id'] == "mockclientid"
    assert creds['resource_group'] == "env-resource-group"

def test_no_credentials_configured(monkeypatch):
    _prep_env(monkeypatch)
    with pytest.raises(ValueError, match="No credentials found in any source"):
        sap_credentials.fetch_credentials()


def test_partial_credentials_missing_auth_url(monkeypatch):
    _prep_env(monkeypatch)

    # Set only client_id and base_url, missing auth_url
    monkeypatch.setenv("AICORE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AICORE_BASE_URL", "test-base-url")

    # fetch_credentials should succeed (it returns whatever it finds)
    creds = sap_credentials.fetch_credentials()
    creds.pop('resource_group')

    with pytest.raises(ValueError, match="SAP AI Core credentials not found"):
        sap_credentials.validate_credentials(**creds)

def test_credentials_without_authentication_mode(monkeypatch):
    _prep_env(monkeypatch)

    # Set all required fields but no authentication mode (no client_secret, no certs)
    monkeypatch.setenv("AICORE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AICORE_AUTH_URL", "test-auth-url")
    monkeypatch.setenv("AICORE_BASE_URL", "test-base-url")

    creds = sap_credentials.fetch_credentials()
    creds.pop('resource_group')

    # validate_credentials should raise because no authentication mode is provided
    with pytest.raises(ValueError, match="SAP AI Core credentials are incomplete"):
        sap_credentials.validate_credentials(**creds)
