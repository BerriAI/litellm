from types import SimpleNamespace
from unittest.mock import MagicMock

from litellm.proxy.auth.azure_postgres_token import (
    AZURE_POSTGRES_SCOPE,
    generate_azure_postgres_auth_token,
)


def test_generate_azure_postgres_auth_token_uses_database_scope_and_url_encodes_token():
    credential = MagicMock()
    credential.get_token.return_value = SimpleNamespace(token="header.payload/signature+padding=", expires_on=0)

    token = generate_azure_postgres_auth_token(credential=credential)

    assert token == "header.payload%2Fsignature%2Bpadding%3D"
    credential.get_token.assert_called_once_with(AZURE_POSTGRES_SCOPE)
