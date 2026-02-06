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
            "Authorization": "Bearer " + <JWT>,
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT"
        }
        """

        if JWT is None:
            raise ValueError("Missing Snowflake JWT key")

        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + JWT,
                "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            }
        )
        return headers
