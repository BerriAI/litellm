import hmac
import secrets
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse, post_callback
from litellm.proxy.mcp_server.events.webhooks.signing import new_webhook_id, signed_headers
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Result
from litellm.types.mcp_server.events import CallbackFailure, VerificationEnvelope

Poster: TypeAlias = Callable[[str, bytes, Mapping[str, str]], Awaitable[Result[CallbackResponse, CallbackFailure]]]
_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def echoed_challenge(response: CallbackResponse, challenge: str) -> bool:
    if not 200 <= response.status < 300:
        return False
    try:
        echoed: Final = _JSON_OBJECT.validate_json(response.body).get("challenge")
    except ValidationError:
        return False
    return isinstance(echoed, str) and hmac.compare_digest(echoed.encode(), challenge.encode())


async def verify_callback(
    url: str, secret: bytes, subscription: str, now: int, post: Poster = post_callback
) -> CallbackFailure | None:
    challenge: Final = secrets.token_urlsafe(24)
    body: Final = VerificationEnvelope(challenge=challenge).model_dump_json().encode()
    response: Final = await post(
        url, body, signed_headers(secret, new_webhook_id("verification"), now, body, subscription)
    )
    if isinstance(response, Error):
        return response.error
    return None if echoed_challenge(response.ok, challenge) else "challenge_failed"
