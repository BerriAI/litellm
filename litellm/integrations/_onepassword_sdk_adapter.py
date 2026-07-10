"""
1Password SDK adapter -- the only module that touches the untyped `onepassword-sdk`.

Kept deliberately small and excluded from basedpyright (see pyrightconfig.json)
because the third-party SDK ships no type information. All typed logic lives in
`litellm.integrations.onepassword`.
"""

from typing import Awaitable, Callable, List, Optional

import litellm
from litellm.integrations.onepassword import (
    OnePasswordConfig,
    OnePasswordShareError,
    OnePasswordShareResult,
)

_INTEGRATION_NAME = "LiteLLM"

Authenticator = Callable[[OnePasswordConfig], Awaitable[object]]


async def _authenticate(config: OnePasswordConfig) -> object:
    try:
        from onepassword.client import Client
    except ImportError as e:
        raise OnePasswordShareError(
            "The 'onepassword-sdk' package is required for 1Password sharing. "
            "Install it with `pip install onepassword-sdk`."
        ) from e

    return await Client.authenticate(
        auth=config.service_account_token,
        integration_name=_INTEGRATION_NAME,
        integration_version=getattr(litellm, "__version__", "unknown"),
    )


async def share_via_sdk(
    config: OnePasswordConfig,
    title: str,
    secret_value: str,
    recipients: Optional[List[str]],
    expire_after: Optional[str],
    one_time_only: bool,
    authenticate: Authenticator = _authenticate,
) -> OnePasswordShareResult:
    """
    Create an API Credential item holding ``secret_value`` and return a
    secure-share link for it, using the official 1Password SDK.

    ``authenticate`` is injectable so tests can supply a fake SDK client and
    exercise the create + share sequence without the native SDK.
    """
    from onepassword.types import (
        ItemCategory,
        ItemCreateParams,
        ItemField,
        ItemFieldType,
        ItemShareParams,
    )

    client = await authenticate(config)

    try:
        item = await client.items.create(
            ItemCreateParams(
                category=ItemCategory.APICREDENTIALS,
                vault_id=config.vault_id,
                title=title,
                fields=[
                    ItemField(
                        id="credential",
                        title="credential",
                        field_type=ItemFieldType.CONCEALED,
                        value=secret_value,
                    )
                ],
            )
        )

        policy = await client.items.shares.get_account_policy(config.vault_id, item.id)

        valid_recipients = None
        if recipients:
            valid_recipients = await client.items.shares.validate_recipients(policy, recipients)

        share_link = await client.items.shares.create(
            item,
            policy,
            ItemShareParams(
                recipients=valid_recipients,
                expire_after=expire_after,
                one_time_only=one_time_only,
            ),
        )
    except OnePasswordShareError:
        raise
    except Exception as e:
        raise OnePasswordShareError(f"1Password share failed: {e}") from e

    return OnePasswordShareResult(
        share_link=share_link,
        item_id=item.id,
        item_title=title,
        expire_after=expire_after,
        one_time_only=one_time_only,
    )
