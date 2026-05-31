import os
import time
from typing import Any, Dict

from litellm.llms.server_oauth_base import RefreshTokenOAuthAuthenticator


class ClaudeMaxAuthenticator(RefreshTokenOAuthAuthenticator):
    provider = "claude-max"
    env_prefix = "CLAUDE_MAX"
    token_url = os.getenv("CLAUDE_MAX_TOKEN_URL", "https://platform.claude.com/v1/oauth/token")
    client_id = os.getenv("CLAUDE_MAX_CLIENT_ID", "9d1c250a-e61b-44d9-88ed-5944d1962f5e")

    def refresh_tokens(self, data: Dict[str, Any], refresh_token: str) -> Dict[str, Any]:
        response = self._post_refresh_json(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            },
            headers={"content-type": "application/json"},
        )
        return {
            "access_token": response.get("access_token"),
            "refresh_token": response.get("refresh_token") or refresh_token,
            "expires_at": int(time.time()) + int(response.get("expires_in", 3600)),
            "id_token": response.get("id_token") or data.get("id_token"),
            "scope": response.get("scope") or data.get("scope"),
        }
