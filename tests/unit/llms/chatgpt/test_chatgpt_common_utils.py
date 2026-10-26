import base64
import json

import pytest

from litellm.llms.chatgpt.common_utils import (
    extract_chatgpt_account_id,
    get_chatgpt_client_credential,
    is_chatgpt_oauth_key,
    select_chatgpt_client_credential_headers,
)


def _make_jwt(payload: dict) -> str:
    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(payload)}."


CHATGPT_JWT = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-alice"}})
CHATGPT_JWT_WITHOUT_ACCOUNT = _make_jwt({"https://api.openai.com/auth": {"user_id": "user-1"}})
NON_CHATGPT_JWT = _make_jwt({"sub": "someone", "https://api.openai.com/auth": "not-an-object"})


class TestExtractChatGPTAccountId:
    def test_reads_account_id_from_auth_claim(self):
        assert extract_chatgpt_account_id(CHATGPT_JWT) == "acct-alice"

    @pytest.mark.parametrize(
        "token",
        [None, "", "sk-proj-not-a-jwt", "header.only", NON_CHATGPT_JWT, CHATGPT_JWT_WITHOUT_ACCOUNT],
    )
    def test_returns_none_without_a_usable_account_claim(self, token):
        assert extract_chatgpt_account_id(token) is None


class TestIsChatGPTOAuthKey:
    def test_recognizes_bearer_and_raw_chatgpt_jwt(self):
        assert is_chatgpt_oauth_key(CHATGPT_JWT)
        assert is_chatgpt_oauth_key(f"Bearer {CHATGPT_JWT}")
        assert is_chatgpt_oauth_key(f"Bearer {CHATGPT_JWT_WITHOUT_ACCOUNT}")

    @pytest.mark.parametrize(
        "value", [None, "", "Bearer sk-proj-platform-key", "Bearer not.a.jwt", _make_jwt({"sub": "x"})]
    )
    def test_rejects_non_chatgpt_credentials(self, value):
        assert not is_chatgpt_oauth_key(value)


class TestSelectChatGPTClientCredentialHeaders:
    def test_account_header_alone_is_not_forwarded(self):
        selected = select_chatgpt_client_credential_headers(
            {"chatgpt-account-id": "acct-alice", "authorization": "Bearer sk-proj-platform-key"}
        )
        assert dict(selected) == {}
        assert get_chatgpt_client_credential({"chatgpt-account-id": "acct-alice"}) is None

    def test_forwards_only_bearer_and_account_context(self):
        selected = select_chatgpt_client_credential_headers(
            {
                "Authorization": f"Bearer {CHATGPT_JWT}",
                "ChatGPT-Account-Id": "acct-explicit",
                "x-litellm-api-key": "sk-gateway",
                "user-agent": "codex",
            }
        )
        assert dict(selected) == {"Authorization": f"Bearer {CHATGPT_JWT}", "ChatGPT-Account-Id": "acct-explicit"}
        credential = get_chatgpt_client_credential(selected)
        assert credential is not None
        assert credential.access_token == CHATGPT_JWT
        assert credential.account_id == "acct-explicit"

    def test_account_id_falls_back_to_jwt_claim(self):
        credential = get_chatgpt_client_credential({"authorization": f"Bearer {CHATGPT_JWT}"})
        assert credential is not None
        assert credential.account_id == "acct-alice"
