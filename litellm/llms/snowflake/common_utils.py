from typing import Optional


class SnowflakeBase:
    def validate_environment(
        self,
        headers: dict,
        JWT: Optional[str] = None,
    ) -> dict:
        """
        Return headers to use for Snowflake completion request

        Snowflake REST API Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#api-reference
        Expected headers:
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + <JWT or PAT>,
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT" or "PROGRAMMATIC_ACCESS_TOKEN"
        }
        """

        if JWT is None:
            raise ValueError("Missing Snowflake JWT or PAT key")

        # Detect if using PAT token (prefixed with "pat/")
        token_type = "KEYPAIR_JWT"
        token = JWT

        if JWT.startswith("pat/"):
            token_type = "PROGRAMMATIC_ACCESS_TOKEN"
            token = JWT[4:]  # Strip "pat/" prefix

        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + token,
                "X-Snowflake-Authorization-Token-Type": token_type,
            }
        )
        return headers
