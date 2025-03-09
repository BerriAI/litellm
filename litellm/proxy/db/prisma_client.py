"""
This file contains the PrismaWrapper class, which is used to wrap the Prisma client and handle the RDS IAM token.
"""

import asyncio
import os
import urllib
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from litellm.secret_managers.main import str_to_bool


class PrismaWrapper:
    def __init__(self, original_prisma: Any, iam_token_db_auth: bool):
        self._original_prisma = original_prisma
        self.iam_token_db_auth = iam_token_db_auth

    def is_token_expired(self, token_url: Optional[str]) -> bool:
        if token_url is None:
            return True
        # Decode the token URL to handle URL-encoded characters
        decoded_url = urllib.parse.unquote(token_url)

        # Parse the token URL
        parsed_url = urllib.parse.urlparse(decoded_url)

        # Parse the query parameters from the path component (if they exist there)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Get expiration time from the query parameters
        expires = query_params.get("X-Amz-Expires", [None])[0]
        if expires is None:
            raise ValueError("X-Amz-Expires parameter is missing or invalid.")

        expires_int = int(expires)

        # Get the token's creation time from the X-Amz-Date parameter
        token_time_str = query_params.get("X-Amz-Date", [""])[0]
        if not token_time_str:
            raise ValueError("X-Amz-Date parameter is missing or invalid.")

        # Ensure the token time string is parsed correctly
        try:
            token_time = datetime.strptime(token_time_str, "%Y%m%dT%H%M%SZ")
        except ValueError as e:
            raise ValueError(f"Invalid X-Amz-Date format: {e}")

        # Calculate the expiration time
        expiration_time = token_time + timedelta(seconds=expires_int)

        # Current time in UTC
        current_time = datetime.utcnow()

        # Check if the token is expired
        return current_time > expiration_time

    def get_rds_iam_token(self) -> Optional[str]:
        if self.iam_token_db_auth:
            from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token

            db_host = os.getenv("DATABASE_HOST")
            db_port = os.getenv("DATABASE_PORT")
            db_user = os.getenv("DATABASE_USER")
            db_name = os.getenv("DATABASE_NAME")
            db_schema = os.getenv("DATABASE_SCHEMA")

            token = generate_iam_auth_token(
                db_host=db_host, db_port=db_port, db_user=db_user
            )

            # print(f"token: {token}")
            _db_url = f"postgresql://{db_user}:{token}@{db_host}:{db_port}/{db_name}"
            if db_schema:
                _db_url += f"?schema={db_schema}"

            os.environ["DATABASE_URL"] = _db_url
            return _db_url
        return None

    async def recreate_prisma_client(
        self, new_db_url: str, http_client: Optional[Any] = None
    ):
        from prisma import Prisma  # type: ignore

        if http_client is not None:
            self._original_prisma = Prisma(http=http_client)
        else:
            self._original_prisma = Prisma()

        await self._original_prisma.connect()

    def __getattr__(self, name: str):
        original_attr = getattr(self._original_prisma, name)
        if self.iam_token_db_auth:
            db_url = os.getenv("DATABASE_URL")
            if self.is_token_expired(db_url):
                db_url = self.get_rds_iam_token()
                loop = asyncio.get_event_loop()

                if db_url:
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.recreate_prisma_client(db_url), loop
                        )
                    else:
                        asyncio.run(self.recreate_prisma_client(db_url))
                else:
                    raise ValueError("Failed to get RDS IAM token")

        return original_attr


def should_update_prisma_schema(
    disable_updates: Optional[Union[bool, str]] = None
) -> bool:
    """
    Determines if Prisma Schema updates should be applied during startup.

    Args:
        disable_updates: Controls whether schema updates are disabled.
            Accepts boolean or string ('true'/'false'). Defaults to checking DISABLE_SCHEMA_UPDATE env var.

    Returns:
        bool: True if schema updates should be applied, False if updates are disabled.

    Examples:
        >>> should_update_prisma_schema()  # Checks DISABLE_SCHEMA_UPDATE env var
        >>> should_update_prisma_schema(True)  # Explicitly disable updates
        >>> should_update_prisma_schema("false")  # Enable updates using string
    """
    if disable_updates is None:
        disable_updates = os.getenv("DISABLE_SCHEMA_UPDATE", "false")

    if isinstance(disable_updates, str):
        disable_updates = str_to_bool(disable_updates)

    return not bool(disable_updates)
