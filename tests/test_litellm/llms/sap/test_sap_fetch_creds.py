import json
from litellm.llms.sap.credentials import fetch_credentials
mock_sap_service_key_dict = {
    "serviceurls":
        {
            "AI_API_URL":"https://testurl.hana.ondemand.com/"
        },
    "clientid":"mockclientid",
    "clientsecret":"mockclientsecret",
    "url":"https://test.sap.hana.ondemand.com/"
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

def test_sap_fetch_creds_from_env_service_key(monkeypatch):
    monkeypatch.setenv("AICORE_HOME", 'notexist')
    monkeypatch.setenv("AICORE_SERVICE_KEY", json.dumps(mock_sap_service_key_dict))
    creds = fetch_credentials()
    assert creds == expected_creds

def test_sap_fetch_creds_from_api_key_service_key(monkeypatch):
    monkeypatch.setenv("AICORE_HOME", 'notexist')
    creds = fetch_credentials(service_key=json.dumps(mock_sap_service_key_dict))
    assert creds == expected_creds

def test_fetch_creds_from_env_vcap_service(monkeypatch):
    monkeypatch.setenv("AICORE_HOME", 'notexist')
    monkeypatch.setenv("VCAP_SERVICES", json.dumps(mock_sap_vcap_service_key_dict))
    creds = fetch_credentials()
    assert creds['client_id'] == "vcap-clientid"
    assert creds['client_secret'] == "vcap-clientsecret"

def test_fetch_creds_from_env(monkeypatch):
    monkeypatch.setenv("AICORE_HOME", 'notexist')
    monkeypatch.setenv("AICORE_CLIENT_ID", "env-client-id")
    monkeypatch.setenv("AICORE_CLIENT_SECRET", "env-client-secret")
    monkeypatch.setenv("AICORE_AUTH_URL", "env-auth-url")
    monkeypatch.setenv("AICORE_BASE_URL", "env-base-url")
    monkeypatch.setenv("AICORE_RESOURCE_GROUP", "env-resource-group")

    creds = fetch_credentials()

    assert creds['client_id'] == "env-client-id"
    assert creds['client_secret'] == "env-client-secret"
    assert creds['auth_url'] == "env-auth-url/oauth/token"
    assert creds['base_url'] == "env-base-url/v2"
    assert creds['resource_group'] == "env-resource-group"
