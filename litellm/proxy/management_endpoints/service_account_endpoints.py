"""
Service Account key management endpoints.

A service account is a LiteLLM user with a row in LiteLLM_ServiceAccountTable.
Its lifecycle is driven by two boolean columns:

    is_active                   — True once a key has been issued (approved).
    is_key_rotation_requested   — True when an owner has asked for a new key.

The pending states map onto the list filters:

    is_active=False, is_key_rotation_requested=False → creation request
    is_active=True,  is_key_rotation_requested=False → active ("my keys")
    is_active=True,  is_key_rotation_requested=True  → rotation request

User-facing (owners):
    GET    /service-account/list?status=...     — list SAs (my_keys filter)
    POST   /service-account/request-rotation     — file a rotation request

Approver-facing (Xyne team — gated by the grid proxy):
    POST   /service-account/approve             — approve creation OR rotation
    POST   /service-account/reject              — reject creation OR rotation

"Request a new service account" reuses the existing /user/new endpoint with
is_service_account=True, auto_create_key=False (creates an inactive SA row).

Approve works for both creation and rotation because /key/generate's helper
_activate_service_account_and_block_previous_keys flips is_active→True and
is_key_rotation_requested→False and blocks the SA's prior keys — regardless of
which pending state the SA was in. Reject branches on the SA's current state:
creation-pending → delete the SA (+user) row; rotation-pending → just clear
the flag.

GPG key delivery:
    A creation request carries the requester's ASCII-armored OpenPGP public key
    (stored on the SA row as `public_key`) and the requester's user_id (`requester`).
    At creation-approve the freshly minted key is GPG-encrypted to that public key
    with pgpy, so the plaintext key never leaves litellm — only the ASCII-armored
    ciphertext is returned to the approver UI and DM'd to the requester.

Slack notifications:
    Service-account lifecycle events fire a best-effort Slack DM (non-blocking) to the
    requester + owners: creation requested, creation approved (with the encrypted
    key), creation rejected, rotation requested, rotation approved/rejected. DMs
    use the Slack Web API and require SLACK_BOT_TOKEN (xoxb-..., scopes
    users:read, users:read.email, im:write, chat:write). Creation-approve key
    delivery sends a .gpg file and also requires files:write; without a bot token
    lifecycle notices fall back to the opt-in AlertType.service_account_request
    channel webhook.
"""

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import (
    ApproveServiceAccountRequest,
    ApproveServiceAccountResponse,
    RejectServiceAccountRequest,
    RequestRotationRequest,
    ServiceAccountKeyInfo,
    ServiceAccountListItem,
    ServiceAccountListResponse,
    SpecialModelNames,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _activate_service_account_and_block_previous_keys,
    generate_key_helper_fn,
)
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.utils import handle_exception_on_proxy

router = APIRouter()
_TAGS = ["service account management"]
_DEPS = [Depends(user_api_key_auth)]


# ─── helpers ─────────────────────────────────────────────────────────────────

# Domain used for the synthetic service-account user_email. A service account
# is a non-human LiteLLM user; its email is derived from its name so owners
# can be resolved/looked up and Slack can address them: <name>-service-account@juspay.in
_SERVICE_ACCOUNT_EMAIL_DOMAIN = "juspay.in"
_SERVICE_ACCOUNT_ALLOWED_ROUTES = ["llm_api_routes"]


def _sanitize_sa_slug(name: Optional[str], fallback: str) -> str:
    """Normalize a service-account name into an alias/email-safe slug.

    lowercase, trim, collapse any run of non-[a-z0-9-] chars to a single '-',
    and strip leading/trailing '-'. Falls back to `fallback` (the SA user_id)
    when name is None/empty or produces an empty slug — so a key_alias / email
    is always produced.

    Keep in sync with the twin in grid-ai-onboarding/backend/app/routes/service_accounts.py.
    """
    if not name:
        return fallback
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or fallback


def _sa_key_alias(name: Optional[str], fallback: str) -> str:
    """The VerificationToken key_alias for a service account.

    Always ends with the `-service-account` suffix, e.g. `payments-batch-runner-service-account`.
    """
    return f"{_sanitize_sa_slug(name, fallback)}-service-account"


def _sa_user_email(name: Optional[str], fallback: str) -> str:
    """The synthetic user_email for a service account user row.

    `<key_alias>@juspay.in`, e.g. `payments-batch-runner-service-account@juspay.in`.
    """
    return f"{_sa_key_alias(name, fallback)}@{_SERVICE_ACCOUNT_EMAIL_DOMAIN}"


def _expires_from_duration(duration: str) -> Optional[datetime]:
    """Compute a key's new `expires` from an approver-chosen duration.

    Mirrors the duration handling in prepare_key_update_data:
        - "-1" or empty → None (the key never expires)
        - otherwise    → now(UTC) + duration_in_seconds(duration)

    Used on rotation approval, where we extend the existing key's expiry in
    place rather than minting a new key.
    """
    if not duration or duration == "-1":
        return None
    seconds = duration_in_seconds(duration=duration)
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


# ─── GPG (OpenPGP) public-key validation + encryption ────────────────────────
# The requester's ASCII-armored OpenPGP public key is collected on the creation
# form and stored on the SA row. At creation-approve time the freshly minted
# service-account key is GPG-encrypted to that public key with pgpy (pure-Python
# OpenPGP — no `gpg` binary required) so the plaintext key never leaves litellm:
# only the ASCII-armored ciphertext is relayed to the approver UI and DM'd to the
# requester. Keep the validation logic in sync with the twin in
# grid-ai-onboarding/backend/app/routes/service_accounts.py.

_PGP_PUBLIC_KEY_HEADER = "-----BEGIN PGP PUBLIC KEY BLOCK-----"
_PGP_PUBLIC_KEY_FOOTER = "-----END PGP PUBLIC KEY BLOCK-----"


def _validate_openpgp_public_key(key: Optional[str]) -> str:
    """Validate that `key` is an ASCII-armored OpenPGP PUBLIC key block.

    Returns the stripped key on success. Raises HTTPException(400) if missing,
    not a string, or not wrapped in the expected armor header/footer. This is a
    fast structural check (no parse) — the real parse happens at encrypt time
    in _gpg_encrypt, which raises HTTPException(500) on a malformed key.
    """
    if not key or not isinstance(key, str):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=(
                "A GPG public key is required to request a service account. "
                "Paste your ASCII-armored OpenPGP public key "
                "(it must begin with '-----BEGIN PGP PUBLIC KEY BLOCK-----')."
            ),
        )
    stripped = key.strip()
    if not stripped.startswith(_PGP_PUBLIC_KEY_HEADER) or _PGP_PUBLIC_KEY_FOOTER not in stripped:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=(
                "The GPG public key must be an ASCII-armored OpenPGP public key "
                "block beginning with '-----BEGIN PGP PUBLIC KEY BLOCK-----' "
                "and ending with '-----END PGP PUBLIC KEY BLOCK-----'."
            ),
        )
    return stripped


def _gpg_encrypt(plaintext: str, public_key_armored: str) -> str:
    """GPG-encrypt `plaintext` to the given ASCII-armored public key.

    Returns the ASCII-armored ciphertext (`-----BEGIN PGP MESSAGE-----` block).
    Uses pgpy (pure-Python OpenPGP). Raises HTTPException(500) on any crypto
    error so the approve fails loudly rather than silently leaking a plaintext
    key back to the caller.
    """
    try:
        import pgpy
    except ImportError as e:
        missing_name = getattr(e, "name", None) or "pgpy"
        dependency_hint = (
            "Install standard-imghdr alongside pgpy for Python 3.13."
            if missing_name == "imghdr"
            else "Install the pgpy package on the proxy."
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Cannot GPG-encrypt the service account key: required Python "
                f"module '{missing_name}' is not installed on the proxy. "
                f"{dependency_hint}"
            ),
        ) from e
    try:
        pub_key, _ = pgpy.PGPKey.from_blob(public_key_armored)
        message = pgpy.PGPMessage.new(plaintext)
        encrypted = pub_key.encrypt(message)
        return str(encrypted)
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Failed to GPG-encrypt the service account key with the stored "
                "public key. The requester must re-export a valid public key."
            ),
        ) from e


async def _cleanup_failed_creation_key(
    prisma_client: Any, user_id: str, key_token: Optional[Any]
) -> None:
    """Best-effort cleanup when creation approval minted a key but cannot finish.

    The plaintext key cannot be safely returned unless it was encrypted to the
    requester's public key. If encryption/activation fails after key generation,
    block the just-minted VerificationToken and keep the service account pending
    so the request can be retried after the underlying issue is fixed.
    """
    if not key_token:
        return
    try:
        await prisma_client.db.litellm_verificationtoken.update(
            where={"token": key_token},
            data={"blocked": True},
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            "service_account_endpoints: failed to block generated key after "
            f"creation approval failure for {user_id} (non-blocking): {e}"
        )
    try:
        await prisma_client.db.litellm_serviceaccounttable.update(
            where={"user_id": user_id},
            data={"is_active": False, "is_key_rotation_requested": False},
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            "service_account_endpoints: failed to keep service account pending "
            f"after creation approval failure for {user_id} (non-blocking): {e}"
        )


def _build_sa_list_item(
    sa: Any, keys: List[Any]
) -> ServiceAccountListItem:
    """Adapt a LiteLLM_ServiceAccountTable row + its VerificationToken keys
    into the ServiceAccountListItem response model."""
    key_infos: List[ServiceAccountKeyInfo] = []
    for k in keys:
        key_infos.append(
            ServiceAccountKeyInfo(
                token=getattr(k, "token", None) or "",
                key_alias=getattr(k, "key_alias", None),
                key_name=getattr(k, "key_name", None),
                expires=getattr(k, "expires", None),
                blocked=getattr(k, "blocked", None),
                spend=getattr(k, "spend", None),
                created_at=getattr(k, "created_at", None),
            )
        )
    return ServiceAccountListItem(
        user_id=getattr(sa, "user_id", "") or "",
        owner_ids=list(getattr(sa, "owner_ids", []) or []),
        name=getattr(sa, "name", None),
        requested_models=list(getattr(sa, "requested_models", []) or []),
        use_case=getattr(sa, "use_case", None),
        requested_rpm_limit=getattr(sa, "requested_rpm_limit", None),
        requested_parallel_requests_limit=getattr(
            sa, "requested_parallel_requests_limit", None
        ),
        is_active=bool(getattr(sa, "is_active", False)),
        is_key_rotation_requested=bool(
            getattr(sa, "is_key_rotation_requested", False)
        ),
        public_key=getattr(sa, "public_key", None),
        requester=getattr(sa, "requester", None),
        created_at=getattr(sa, "created_at", None),
        keys=key_infos,
    )


async def _fetch_sa_keys(prisma_client: Any, user_id: str) -> List[Any]:
    """Fetch all VerificationToken keys for a service account user."""
    return await prisma_client.db.litellm_verificationtoken.find_many(
        where={"user_id": user_id}
    )


async def _get_sa_or_404(prisma_client: Any, user_id: str) -> Any:
    sa = await prisma_client.db.litellm_serviceaccounttable.find_unique(
        where={"user_id": user_id}
    )
    if sa is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Service account not found: {user_id}",
        )
    return sa


async def _resolve_user_email(prisma_client: Any, user_id: Optional[str]) -> Optional[str]:
    """Resolve a litellm user_id to its user_email, or None if not found."""
    if not user_id:
        return None
    try:
        row = await prisma_client.db.litellm_usertable.find_unique(
            where={"user_id": user_id}
        )
        return getattr(row, "user_email", None) if row else None
    except Exception:
        return None


async def _resolve_owner_emails(prisma_client: Any, sa: Any) -> List[str]:
    """Resolve a service account's owner_ids to their user_emails."""
    owner_ids = list(getattr(sa, "owner_ids", []) or [])
    emails: List[str] = []
    try:
        if owner_ids:
            owners = await prisma_client.db.litellm_usertable.find_many(
                where={"user_id": {"in": owner_ids}}
            )
            for o in owners:
                email = getattr(o, "user_email", None)
                if email:
                    emails.append(email)
    except Exception as e:
        verbose_proxy_logger.warning(
            "service_account_endpoints: failed to resolve owner emails "
            f"(continuing with owner IDs): {e}"
        )
    return emails


def _slack_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple, set)):
        rendered = ", ".join(_slack_value(v) for v in value if v is not None)
        return rendered or "None"
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or "None"


def _service_account_status(sa: Any) -> str:
    is_active = bool(getattr(sa, "is_active", False))
    is_key_rotation_requested = bool(
        getattr(sa, "is_key_rotation_requested", False)
    )
    if not is_active and not is_key_rotation_requested:
        return "Creation pending"
    if is_active and is_key_rotation_requested:
        return "Rotation requested"
    if is_active:
        return "Active"
    return "Inactive"


def _service_account_snapshot(sa: Any, **updates: Any) -> Any:
    fields = (
        "user_id",
        "owner_ids",
        "name",
        "requested_models",
        "use_case",
        "requested_rpm_limit",
        "requested_parallel_requests_limit",
        "is_active",
        "is_key_rotation_requested",
        "public_key",
        "requester",
        "created_at",
    )
    data = {field: getattr(sa, field, None) for field in fields}
    data.update(updates)
    return SimpleNamespace(**data)


def _format_service_account_details(
    sa: Any,
    requester_email: Optional[str],
    owner_emails: List[str],
) -> str:
    user_id = getattr(sa, "user_id", None) or ""
    name = getattr(sa, "name", None)
    owner_ids = list(getattr(sa, "owner_ids", []) or [])
    requester = requester_email or getattr(sa, "requester", None)
    owners = owner_emails or owner_ids

    lines = [
        "Service account details:",
        f"Name: {_slack_value(name)}",
        f"User ID: {_slack_value(user_id)}",
        f"Service account email: {_slack_value(_sa_user_email(name, user_id))}",
        f"Key alias: {_slack_value(_sa_key_alias(name, user_id))}",
        f"Requester: {_slack_value(requester)}",
        f"Owners: {_slack_value(owners)}",
        f"Use case: {_slack_value(getattr(sa, 'use_case', None))}",
        f"Requested models: {_slack_value(getattr(sa, 'requested_models', None))}",
        f"Requested RPM limit: {_slack_value(getattr(sa, 'requested_rpm_limit', None))}",
        "Requested parallel requests limit: "
        f"{_slack_value(getattr(sa, 'requested_parallel_requests_limit', None))}",
        f"Status: {_service_account_status(sa)}",
        f"Active: {_slack_value(bool(getattr(sa, 'is_active', False)))}",
        "Key rotation requested: "
        f"{_slack_value(bool(getattr(sa, 'is_key_rotation_requested', False)))}",
        "GPG public key on file: "
        f"{_slack_value(bool(getattr(sa, 'public_key', None)))}",
    ]
    created_at = getattr(sa, "created_at", None)
    if created_at:
        lines.append(f"Created at: {_slack_value(created_at)}")
    return "\n".join(lines)


async def _send_sa_slack_dm(emails: List[str], message: str) -> None:
    """DM each email via the Slack Web API (bot token).

    Falls back to the channel webhook (AlertType.service_account_request) when
    no bot token is configured, so service-account notifications still post when
    only the legacy webhook is set up. Never raises — Slack misconfiguration
    must not fail an approve/reject.
    """
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj
        from litellm.types.integrations.slack_alerting import AlertType

        instance = proxy_logging_obj.slack_alerting_instance
        bot_token = os.getenv("SLACK_WEB_API_TOKEN") or os.getenv("SLACK_BOT_TOKEN")
        if bot_token and emails:
            # DM every resolved recipient. A missing Slack account for one user
            # must not prevent the remaining owners from being notified.
            sent_count = 0
            for email in emails:
                try:
                    sent = await instance.send_dm(user_email=email, message=message)
                except Exception as dm_error:
                    verbose_proxy_logger.warning(
                        "service_account_endpoints: Slack DM failed for "
                        f"{email} (continuing): {dm_error}"
                    )
                    continue
                if sent:
                    sent_count += 1
                    verbose_proxy_logger.info(
                        "service_account_endpoints: Slack DM sent to "
                        f"{email}"
                    )
                else:
                    verbose_proxy_logger.warning(
                        "service_account_endpoints: Slack DM skipped/failed for "
                        f"{email}; user may not exist in Slack or bot scopes may be missing"
                    )
            if sent_count > 0:
                return
            verbose_proxy_logger.warning(
                "service_account_endpoints: no service-account Slack DMs were "
                "delivered; falling back to channel webhook"
            )
        # Fallback: post to the SA channel webhook (opt-in alert type).
        await instance.send_alert(
            message=message,
            level="Low",
            alert_type=AlertType.service_account_request,
            alerting_metadata={},
        )
    except Exception as e:  # non-blocking
        verbose_proxy_logger.warning(
            f"service_account_endpoints: Slack notify failed (non-blocking): {e}"
        )


async def _notify_sa_event_slack(
    prisma_client: Any,
    sa: Any,
    event: str,
    extra: str = "",
    requester_id: Optional[str] = None,
) -> None:
    """Best-effort Slack DM for a service-account lifecycle event.

    DMs the requester (if known) and the owners. `event` is a short label like
    "creation requested", "creation rejected", "rotation requested",
    "rotation approved", "rotation rejected". `requester_id` (the filing user)
    is named in the message; when None the owners are addressed instead.
    """
    try:
        sa_name = getattr(sa, "name", None) or getattr(sa, "user_id", "")
        requester_email = await _resolve_user_email(prisma_client, requester_id)
        owner_emails = await _resolve_owner_emails(prisma_client, sa)

        message = (
            f"`Service Account {event}`\n"
            f"{_format_service_account_details(sa, requester_email, owner_emails)}"
        )
        if extra:
            message += f"\n{extra}"

        # DM the requester (the most interested party) + every owner. Dedup in
        # case the requester is also an owner.
        recipients: List[str] = []
        if requester_email:
            recipients.append(requester_email)
        for e in owner_emails:
            if e not in recipients:
                recipients.append(e)
        if not recipients:
            # No emails resolvable — let the webhook fallback post to the channel.
            recipients = []
        verbose_proxy_logger.info(
            "service_account_endpoints: preparing Slack notification "
            f"event={event!r} service_account={sa_name!r} "
            f"requester_resolved={bool(requester_email)} "
            f"owner_email_count={len(owner_emails)} recipient_count={len(recipients)}"
        )
        await _send_sa_slack_dm(recipients, message)
    except Exception as e:  # non-blocking
        verbose_proxy_logger.warning(
            f"service_account_endpoints: Slack notify failed (non-blocking): {e}"
        )


async def _notify_requester_encrypted_key_slack(
    prisma_client: Any,
    sa: Any,
    requester_id: Optional[str],
    encrypted_key: str,
    expires: Optional[datetime],
    extra: str = "",
) -> None:
    """DM the requester/owners the encrypted service-account key on approve.

    The encrypted payload is attached as `filename.gpg`, and the message names
    the requester and explicitly instructs recipients to decrypt locally with
    `gpg --decrypt filename.gpg` using the matching private key. The plaintext
    key is never put in any Slack message. Best-effort, non-blocking.
    """
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj

        sa_name = getattr(sa, "name", None) or getattr(sa, "user_id", "")
        requester_email = await _resolve_user_email(prisma_client, requester_id)
        owner_emails = await _resolve_owner_emails(prisma_client, sa)
        recipients: List[str] = []
        for email in [requester_email, *owner_emails]:
            if email and email not in recipients:
                recipients.append(email)

        expires_line = (
            f"Key expires: {expires}\n" if expires else "Key never expires.\n"
        )
        details = _format_service_account_details(sa, requester_email, owner_emails)
        if extra:
            details += f"\n{extra}"
        message = (
            f"`Service Account creation approved`\n"
            f"{details}\n"
            f"{expires_line}"
            "Your service account key is attached as `filename.gpg`, encrypted "
            "to the GPG public key submitted with the request. Download it and "
            "decrypt locally with:\n"
            "    gpg --decrypt filename.gpg\n"
            "Keep the matching private key safe; only that key can read this file."
        )
        instance = proxy_logging_obj.slack_alerting_instance
        send_file = getattr(instance, "send_dm_file", None)
        file_sent = False
        if callable(send_file):
            for email in recipients:
                sent = await send_file(
                    user_email=email,
                    message=message,
                    filename="filename.gpg",
                    title="filename.gpg",
                    file_content=encrypted_key,
                )
                if sent:
                    file_sent = True
                    verbose_proxy_logger.info(
                        "service_account_endpoints: encrypted key file sent to "
                        f"{email}"
                    )
                else:
                    verbose_proxy_logger.warning(
                        "service_account_endpoints: encrypted key file failed for "
                        f"{email}; falling back to encrypted text DM"
                    )
        if file_sent:
            return

        fallback_message = (
            f"{message}\n\n"
            "Slack file upload failed, so the encrypted payload is included below. "
            "Save it as `filename.gpg`, then run:\n"
            "    gpg --decrypt filename.gpg\n\n"
            f"```\n{encrypted_key}\n```"
        )
        await _send_sa_slack_dm(recipients, fallback_message)
    except Exception as e:  # non-blocking
        verbose_proxy_logger.warning(
            f"service_account_endpoints: encrypted-key Slack DM failed "
            f"(non-blocking): {e}"
        )


# ─── Endpoint 1: list ────────────────────────────────────────────────────────


@router.get(
    "/service-account/list",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=ServiceAccountListResponse,
)
@management_endpoint_wrapper
async def list_service_accounts(
    status: str = "my_keys",
    owner_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List service accounts filtered by lifecycle state.

    status:
        - my_keys:            is_active=True, is_key_rotation_requested=False
        - creation_requests:  is_active=False, is_key_rotation_requested=False
        - rotation_requests:  is_active=True, is_key_rotation_requested=True

    For my_keys, owner_user_id further restricts to SAs where that user is an
    owner (Prisma `has` filter on the owner_ids array).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected",
        )

    valid = {"my_keys", "creation_requests", "rotation_requests"}
    if status not in valid:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"status must be one of {sorted(valid)}",
        )

    where: dict = {}
    if status == "my_keys":
        where = {"is_active": True, "is_key_rotation_requested": False}
        if owner_user_id:
            where["owner_ids"] = {"has": owner_user_id}
    elif status == "creation_requests":
        where = {"is_active": False, "is_key_rotation_requested": False}
    elif status == "rotation_requests":
        where = {"is_active": True, "is_key_rotation_requested": True}

    try:
        sa_rows = await prisma_client.db.litellm_serviceaccounttable.find_many(
            where=where
        )
        # Batch-fetch keys for each SA.
        items: List[ServiceAccountListItem] = []
        for sa in sa_rows:
            keys = await _fetch_sa_keys(prisma_client, sa.user_id)
            items.append(_build_sa_list_item(sa, keys))
        return ServiceAccountListResponse(service_accounts=items)
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception_on_proxy(e)


# ─── Endpoint 4: request-rotation (owner) ────────────────────────────────────


@router.post(
    "/service-account/request-rotation",
    tags=_TAGS,
    dependencies=_DEPS,
)
@management_endpoint_wrapper
async def request_service_account_rotation(
    data: RequestRotationRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Owner-side: file a key-rotation request for a service account.

    Flips is_key_rotation_requested False→True. The caller (requested_by_user_id)
    must be in the SA's owner_ids. Only an active SA with no pending rotation
    can request one.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected",
        )

    try:
        sa = await _get_sa_or_404(prisma_client, data.user_id)

        owner_ids = list(getattr(sa, "owner_ids", []) or [])
        if data.requested_by_user_id not in owner_ids:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Only service account owners can request a key rotation",
            )

        if not sa.is_active:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail="Service account is not active — request a creation instead",
            )
        if sa.is_key_rotation_requested:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail="A key rotation is already requested for this service account",
            )

        await prisma_client.db.litellm_serviceaccounttable.update(
            where={"user_id": data.user_id},
            data={"is_key_rotation_requested": True},
        )
        sa_for_notification = _service_account_snapshot(
            sa,
            is_key_rotation_requested=True,
        )

        asyncio.create_task(
            _notify_sa_event_slack(  # type: ignore[arg-type]
                prisma_client,
                sa_for_notification,
                "rotation requested",
                requester_id=data.requested_by_user_id,
            )
        )

        return {"success": True, "user_id": data.user_id}
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception_on_proxy(e)


# ─── Endpoint 2: approve (creation OR rotation) ───────────────────────────────


@router.post(
    "/service-account/approve",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=ApproveServiceAccountResponse,
)
@management_endpoint_wrapper
async def approve_service_account(
    data: ApproveServiceAccountRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Approve a service account creation OR key-rotation request.

    Branches on the SA's current state:
        - creation-pending (is_active=False): issue the first key → SA activated.
        - rotation-pending (is_active=True, is_key_rotation_requested=True):
          EXTEND the existing key's `expires` in place (now + duration) and clear
          the rotation flag. The key's secret, alias, and team are unchanged —
          the owners keep using the key they already have, so the key value is
          NOT re-revealed. No new key is minted and no prior key is blocked.

    Creation approval calls generate_key_helper_fn (which creates a key with
    `expires` from `data.duration`) and then
    _activate_service_account_and_block_previous_keys (which flips
    is_active→True, is_key_rotation_requested→False, and blocks the SA's prior
    keys). The new key value is returned once.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected",
        )

    try:
        sa = await _get_sa_or_404(prisma_client, data.user_id)

        is_creation_pending = (
            sa.is_active is False and sa.is_key_rotation_requested is False
        )
        is_rotation_pending = (
            sa.is_active is True and sa.is_key_rotation_requested is True
        )
        if not is_creation_pending and not is_rotation_pending:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail=(
                    "Service account is not in a pending state to approve. "
                    "Creation requires is_active=False; rotation requires "
                    "is_active=True and is_key_rotation_requested=True."
                ),
            )

        sa_name = getattr(sa, "name", None) or data.user_id

        # On creation approval, the approver may edit the request's fields
        # before the key is issued. Apply any provided editable fields to the
        # SA row (overwrite). Rotation approval ignores identity edits.
        update_fields: dict = {}
        if is_creation_pending:
            if data.name is not None:
                update_fields["name"] = data.name
                sa_name = data.name
            if data.use_case is not None:
                update_fields["use_case"] = data.use_case
            if data.requested_models is not None:
                update_fields["requested_models"] = list(data.requested_models)
            if data.requested_rpm_limit is not None:
                update_fields["requested_rpm_limit"] = data.requested_rpm_limit
            if data.requested_parallel_requests_limit is not None:
                update_fields["requested_parallel_requests_limit"] = (
                    data.requested_parallel_requests_limit
                )
            if data.owner_ids is not None:
                if len(data.owner_ids) < 2:
                    raise HTTPException(
                        status_code=http_status.HTTP_400_BAD_REQUEST,
                        detail="A service account requires at least 2 owners.",
                    )
                update_fields["owner_ids"] = list(data.owner_ids)
            if update_fields:
                await prisma_client.db.litellm_serviceaccounttable.update(
                    where={"user_id": data.user_id},
                    data=update_fields,
                )

        key_alias = _sa_key_alias(sa_name, data.user_id)

        # ── Rotation approval: extend the existing key's expiry in place ──
        # A rotation request is NOT a re-key — the whole point is to keep the
        # same secret the owners already hold and just give it a fresh validity
        # window. So we find the SA's existing active key, set its `expires` to
        # now(UTC) + duration, clear is_key_rotation_requested, and return. No
        # new key is minted and no prior key is blocked. (Creation approval
        # below still mints a new key, since there is no key to extend yet.)
        if is_rotation_pending:
            existing_keys = await prisma_client.db.litellm_verificationtoken.find_many(
                where={"user_id": data.user_id}
            )
            # The key to extend is the active (non-blocked) one; fall back to any.
            active_key = next(
                (k for k in existing_keys if not getattr(k, "blocked", False)), None
            ) or (existing_keys[0] if existing_keys else None)
            if active_key is None:
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail="No existing key found to extend for this service "
                    "account. Reject the rotation request and approve a "
                    "creation instead.",
                )

            expires = _expires_from_duration(data.duration)
            await prisma_client.db.litellm_verificationtoken.update(
                where={"token": getattr(active_key, "token", None)},
                data={"expires": expires},
            )
            # Clear the rotation flag — the rotation is fulfilled by this
            # extension. (is_active stays True; the SA was already active.)
            await prisma_client.db.litellm_serviceaccounttable.update(
                where={"user_id": data.user_id},
                data={"is_key_rotation_requested": False},
            )
            sa_for_notification = _service_account_snapshot(
                sa,
                is_key_rotation_requested=False,
            )

            # Notify owners (best-effort, non-blocking). The key value is
            # unchanged, so it is NOT re-revealed — owners keep using the
            # secret they already have.
            event = "rotation approved"
            extra = f"Key expiry extended to: {expires}" if expires else "Key expiry cleared (never expires)"
            asyncio.create_task(
                _notify_sa_event_slack(  # type: ignore[arg-type]
                    prisma_client,
                    sa_for_notification,
                    event,
                    extra=extra,
                    requester_id=getattr(sa, "requester", None),
                )
            )

            return ApproveServiceAccountResponse(
                user_id=data.user_id,
                key="",  # unchanged — not re-revealed on rotation
                key_id=getattr(active_key, "token", None),
                expires=expires,
            )

        # ── Creation approval: mint the first key ──
        # Resolve the team for the new key. The approver picks it from the
        # team dropdown (required) — there is no existing key to inherit from.
        if not data.team_id:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail="team_id is required to approve (pick a team for the key).",
            )
        team_id_for_key = data.team_id

        # A creation request is required to carry a public key (validated at filing
        # time). Check before key generation so a bad request does not mint a
        # service-account key that cannot be safely returned.
        public_key = getattr(sa, "public_key", None)
        requester_id = getattr(sa, "requester", None)
        if not public_key:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Service account has no GPG public key on file — cannot "
                    "encrypt the issued key. Reject this request and have the "
                    "requester re-file it with a valid public key."
                ),
            )

        # Generate the new key. generate_key_helper_fn returns key_data where
        # `token` is the actual sk-... key value and `token_id` is the hash.
        requested_rpm_limit = update_fields.get(
            "requested_rpm_limit", getattr(sa, "requested_rpm_limit", None)
        )
        requested_parallel_requests_limit = update_fields.get(
            "requested_parallel_requests_limit",
            getattr(sa, "requested_parallel_requests_limit", None),
        )
        changed_by = user_api_key_dict.user_id or requester_id or data.user_id
        key_data_resp = await generate_key_helper_fn(
            request_type="key",
            user_id=data.user_id,
            key_alias=key_alias,
            team_id=team_id_for_key,
            duration=data.duration,
            models=[SpecialModelNames.all_team_models.value],
            max_parallel_requests=requested_parallel_requests_limit,
            rpm_limit=requested_rpm_limit,
            created_by=changed_by,
            updated_by=changed_by,
            allowed_routes=_SERVICE_ACCOUNT_ALLOWED_ROUTES,
            table_name="key",
        )

        new_key_value = cast(str, key_data_resp.get("token", ""))
        new_key_hash = key_data_resp.get("token_id")
        expires = key_data_resp.get("expires")

        try:
            # GPG-encrypt the freshly minted key to the requester's stored public
            # key so the plaintext key never leaves litellm. Only the
            # ASCII-armored ciphertext is returned to the caller and DM'd to the
            # requester.
            encrypted_key = _gpg_encrypt(new_key_value, public_key)

            # Activate the SA + block prior keys. This is the same helper the
            # /key/generate route calls, so behaviour is identical to a direct
            # key generation for an SA user.
            await _activate_service_account_and_block_previous_keys(
                user_id=data.user_id,
                new_key_token=new_key_hash,
                prisma_client=prisma_client,
            )

            # Mirror /user/new's team_id handling: adding a user to a team goes
            # through team_member_add so LiteLLM_TeamTable.members_with_roles
            # and LiteLLM_TeamMembership stay in sync with the user row.
            from litellm.proxy.management_endpoints.internal_user_endpoints import (
                _add_user_to_team,
            )

            service_account_email = _sa_user_email(sa_name, data.user_id)
            await _add_user_to_team(
                user_id=data.user_id,
                team_id=team_id_for_key,
                user_api_key_dict=user_api_key_dict,
                user_email=service_account_email,
                max_budget_in_team=None,
                user_role="user",
            )

            user_update_fields: dict = {
                "team_id": team_id_for_key,
                "teams": {"set": [team_id_for_key]},
                "models": [SpecialModelNames.no_default_models.value],
            }
            # If the approver edited the name, re-sync the SA user row's
            # user_email so it stays equal to <key_alias>@juspay.in. The
            # email was set provisionally at filing time (grid proxy); an
            # approve-time name edit would otherwise desync it.
            if data.name is not None:
                user_update_fields["user_email"] = service_account_email
            await prisma_client.db.litellm_usertable.update(
                where={"user_id": data.user_id},
                data=user_update_fields,
            )
        except Exception:
            await _cleanup_failed_creation_key(
                prisma_client=prisma_client,
                user_id=data.user_id,
                key_token=new_key_hash,
            )
            raise

        approved_sa = _service_account_snapshot(
            sa,
            **update_fields,
            is_active=True,
            is_key_rotation_requested=False,
        )
        approval_details = "\n".join(
            [
                f"Team: {_slack_value(team_id_for_key)}",
                f"Key ID: {_slack_value(new_key_hash)}",
            ]
        )

        # DM the creation-approved notification with the encrypted key attached
        # as filename.gpg. This is the creation-approval Slack message; avoid a
        # separate text-only approval DM so the file appears on the notification.
        asyncio.create_task(
            _notify_requester_encrypted_key_slack(  # type: ignore[func-returns-value]
                prisma_client,
                approved_sa,
                requester_id=requester_id,
                encrypted_key=encrypted_key,
                expires=expires,
                extra=approval_details,
            )
        )

        return ApproveServiceAccountResponse(
            user_id=data.user_id,
            key=encrypted_key,  # encrypted — the plaintext key is never returned
            key_id=new_key_hash,
            expires=expires,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception_on_proxy(e)


# ─── Endpoint 3: reject (creation OR rotation) ───────────────────────────────


@router.post(
    "/service-account/reject",
    tags=_TAGS,
    dependencies=_DEPS,
)
@management_endpoint_wrapper
async def reject_service_account(
    data: RejectServiceAccountRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Reject a service account creation OR key-rotation request.

    Branches on the SA's current state:
        - creation-pending: delete the SA row and the underlying LiteLLM user
          row (a rejected creation leaves an orphan user with no key).
        - rotation-pending: clear is_key_rotation_requested only — keep the SA
          active and its existing key intact.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected",
        )

    try:
        sa = await _get_sa_or_404(prisma_client, data.user_id)

        is_creation_pending = (
            sa.is_active is False and sa.is_key_rotation_requested is False
        )
        is_rotation_pending = (
            sa.is_active is True and sa.is_key_rotation_requested is True
        )
        if not is_creation_pending and not is_rotation_pending:
            raise HTTPException(
                status_code=http_status.HTTP_409_CONFLICT,
                detail="Service account is not in a pending state to reject.",
            )

        if is_creation_pending:
            # Delete the SA row, then the LiteLLM user row to avoid an orphan
            # user with no key. Best-effort user deletion — if it fails we still
            # removed the SA entry (the request record), which is the contract.
            await prisma_client.db.litellm_serviceaccounttable.delete(
                where={"user_id": data.user_id}
            )
            try:
                await prisma_client.db.litellm_usertable.delete(
                    where={"user_id": data.user_id}
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"service_account_endpoints: failed to delete orphan user "
                    f"{data.user_id} on creation reject (non-blocking): {e}"
                )
            event = "creation rejected"
        else:
            # rotation-pending: clear the flag only.
            await prisma_client.db.litellm_serviceaccounttable.update(
                where={"user_id": data.user_id},
                data={"is_key_rotation_requested": False},
            )
            event = "rotation rejected"
            sa = _service_account_snapshot(
                sa,
                is_key_rotation_requested=False,
            )

        extra = f"Reason: {data.reason}" if data.reason else ""

        asyncio.create_task(
            _notify_sa_event_slack(  # type: ignore[arg-type]
                prisma_client,
                sa,
                event,
                extra=extra,
                requester_id=getattr(sa, "requester", None),
            )
        )

        return {"success": True, "user_id": data.user_id}
    except HTTPException:
        raise
    except Exception as e:
        raise handle_exception_on_proxy(e)
