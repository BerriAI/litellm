"""
GPU PLAYGROUND — litellm-owned booking system.

User-facing:
  GET    /playground/slots                   — tonight's availability
  POST   /playground/bookings                — create a booking for tonight
  GET    /playground/bookings/me             — caller's recent bookings
  DELETE /playground/bookings/{booking_id}   — cancel
  GET    /playground/ssh-keys                — list caller's SSH keys
  POST   /playground/ssh-keys                — register a public key
  DELETE /playground/ssh-keys/{ssh_key_id}   — remove

Admin (PROXY_ADMIN):
  GET/POST/PUT/DELETE  /playground/admin/nodes[/{node_id}]
  GET                  /playground/admin/status

Cron (PROXY_ADMIN virtual key):
  GET    /playground/internal/allocations-tonight
  POST   /playground/internal/activation-status
  POST   /playground/internal/teardown-status

Design: docs/superpowers/plans/2026-04-09-playground-litellm-integration.md
"""

import base64
import binascii
import hashlib
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from prisma.errors import RecordNotFoundError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    ActivationStatusReportRequest,
    ActivationStatusReportResponse,
    AddPlaygroundSSHKeyRequest,
    CreatePlaygroundBookingRequest,
    CreatePlaygroundNodeRequest,
    LitellmUserRoles,
    PlaygroundAdminStatusResponse,
    PlaygroundAllocationBooking,
    PlaygroundAllocationNode,
    PlaygroundAllocationsTonightResponse,
    PlaygroundBookingPhase,
    PlaygroundBookingResponse,
    PlaygroundNodeResponse,
    PlaygroundSlotNode,
    PlaygroundSlotsResponse,
    PlaygroundSSHKeyResponse,
    TeardownStatusResponse,
    UpdatePlaygroundNodeRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper

try:
    import pytz

    IST = pytz.timezone("Asia/Kolkata")
except ImportError:  # pragma: no cover
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")


router = APIRouter()

# Config — override via env var; defaults match grid's original settings.
MAX_GPUS = int(os.getenv("PLAYGROUND_MAX_GPUS_PER_USER", "8"))
WEEKLY_LIMIT = int(os.getenv("PLAYGROUND_WEEKLY_BOOKING_LIMIT", "1"))
OVERFLOW_HOUR = int(os.getenv("PLAYGROUND_OVERFLOW_START_HOUR", "22"))
OVERFLOW_MINUTE = int(os.getenv("PLAYGROUND_OVERFLOW_START_MINUTE", "0"))
CUTOFF_HOUR = int(os.getenv("PLAYGROUND_CUTOFF_HOUR", "22"))
CUTOFF_MINUTE = int(os.getenv("PLAYGROUND_CUTOFF_MINUTE", "30"))

_SSH_KEY_TYPES = {
    "ssh-rsa",
    "ssh-ed25519",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
}

_TAGS = ["gpu playground"]
_DEPS = [Depends(user_api_key_auth)]


# ─── helpers ─────────────────────────────────────────────────────────────────


def _tonight() -> date:
    return datetime.now(IST).date()


def _tonight_dt() -> datetime:
    # Prisma-client-py can't serialize bare datetime.date into queries
    # (see prisma/_builder.py — only datetime.datetime has a serializer),
    # so every prisma query that filters or sets `night_of` must pass a
    # datetime. The DB column is DATE, so the time component is discarded;
    # use midnight UTC for a stable, unambiguous value.
    return datetime.combine(_tonight(), datetime.min.time(), tzinfo=timezone.utc)


def _booking_phase() -> PlaygroundBookingPhase:
    now = datetime.now(IST)
    cutoff = now.replace(hour=CUTOFF_HOUR, minute=CUTOFF_MINUTE, second=0, microsecond=0)
    overflow = now.replace(hour=OVERFLOW_HOUR, minute=OVERFLOW_MINUTE, second=0, microsecond=0)
    if now >= cutoff:
        return "closed"
    if now >= overflow:
        return "overflow"
    return "open"


def _ssh_fingerprint(public_key: str) -> str:
    """Validate + fingerprint an SSH public key. Raises HTTPException(400) on bad input."""
    parts = public_key.strip().split()
    if len(parts) < 2 or parts[0] not in _SSH_KEY_TYPES:
        raise HTTPException(400, "invalid or unsupported SSH key")
    try:
        raw = base64.b64decode(parts[1], validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "SSH key data is not valid base64")
    digest = hashlib.sha256(raw).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _prisma():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(500, "prisma_client not initialized")
    return prisma_client


def _is_admin(key: UserAPIKeyAuth) -> bool:
    role = key.user_role
    value = role.value if role and hasattr(role, "value") else role
    return value == LitellmUserRoles.PROXY_ADMIN.value


def _require_admin(key: UserAPIKeyAuth) -> None:
    if not _is_admin(key):
        raise HTTPException(403, "PROXY_ADMIN required")


def _effective_user_id(key: UserAPIKeyAuth, target_user_id: Optional[str]) -> str:
    """Return the user_id this operation acts on.

    Admins may override via `target_user_id` — that's grid's admin-on-behalf-of
    forwarding path. Non-admins always act as themselves; target_user_id is
    silently ignored.
    """
    if target_user_id and _is_admin(key):
        return target_user_id
    if not key.user_id:
        raise HTTPException(400, "calling key has no user_id")
    return key.user_id


async def _can_book(user_id: str) -> Tuple[bool, str]:
    """Gate booking creation: SSH key required, window open, no existing
    booking tonight, weekly limit (relaxed during overflow)."""
    prisma = _prisma()

    if not await prisma.db.litellm_usersshkey.count(where={"user_id": user_id}):
        return False, "Register an SSH key before booking"

    phase = _booking_phase()
    if phase == "closed":
        return False, "Booking cutoff has passed for tonight"

    if await prisma.db.litellm_playgroundbooking.count(
        where={
            "user_id": user_id,
            "night_of": _tonight_dt(),
            "status": {"not": "cancelled"},
        }
    ):
        return False, "You already have a booking for tonight"

    # Overflow window relaxes the weekly limit but not the per-night unique check.
    if phase == "open":
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        recent = await prisma.db.litellm_playgroundbooking.count(
            where={
                "user_id": user_id,
                "created_at": {"gte": week_ago},
                "status": {"not": "cancelled"},
            }
        )
        if recent >= WEEKLY_LIMIT:
            return False, f"Weekly booking limit ({WEEKLY_LIMIT}) reached"

    return True, ""


async def _allocate(
    gpu_count: int, preferred_node: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Pick a node with enough free GPUs tonight. Returns (ip, "0,1,2") or (None, None).

    Known race: read-then-write isn't serialized at the DB layer, so two
    concurrent bookings can overlap on GPU indices. Acceptable at current
    user count; add an advisory lock if it ever bites.
    """
    prisma = _prisma()
    tonight_dt = _tonight_dt()

    nodes = await prisma.db.litellm_playgroundnode.find_many(
        where={"is_playground_eligible": True, "is_healthy": True},
    )
    nodes.sort(key=lambda n: (n.ip_address != preferred_node, n.ip_address))

    for node in nodes:
        existing = await prisma.db.litellm_playgroundbooking.find_many(
            where={
                "allocated_node": node.ip_address,
                "night_of": tonight_dt,
                "status": {"in": ["allocated", "active"]},
            }
        )
        used = {g for b in existing for g in (b.allocated_gpus or "").split(",") if g}
        free = [str(i) for i in range(node.total_gpus) if str(i) not in used]
        if len(free) >= gpu_count:
            return node.ip_address, ",".join(free[:gpu_count])

    return None, None


# ─── slots ──────────────────────────────────────────────────────────────────


@router.get(
    "/playground/slots",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundSlotsResponse,
)
@management_endpoint_wrapper
async def get_playground_slots(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundSlotsResponse:
    """Available GPUs per eligible node tonight + booking phase."""
    prisma = _prisma()
    tonight = _tonight()
    tonight_dt = _tonight_dt()

    nodes = await prisma.db.litellm_playgroundnode.find_many(
        where={"is_playground_eligible": True, "is_healthy": True},
    )
    bookings = await prisma.db.litellm_playgroundbooking.find_many(
        where={"night_of": tonight_dt, "status": {"in": ["allocated", "active"]}}
    )
    used_by_node: Dict[str, int] = {}
    for b in bookings:
        used_by_node[b.allocated_node] = used_by_node.get(b.allocated_node, 0) + len(
            [g for g in (b.allocated_gpus or "").split(",") if g]
        )

    return PlaygroundSlotsResponse(
        night_of=tonight,
        booking_phase=_booking_phase(),
        nodes=[
            PlaygroundSlotNode(
                node_id=n.node_id,
                name=n.name,
                ip_address=n.ip_address,
                gpu_type=n.gpu_type,
                total_gpus=n.total_gpus,
                available_gpus=max(0, n.total_gpus - used_by_node.get(n.ip_address, 0)),
            )
            for n in nodes
        ],
    )


# ─── bookings ───────────────────────────────────────────────────────────────


@router.post(
    "/playground/bookings",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundBookingResponse,
)
@management_endpoint_wrapper
async def create_playground_booking(
    request: Request,
    data: CreatePlaygroundBookingRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundBookingResponse:
    """Create a booking for tonight."""
    prisma = _prisma()
    user_id = _effective_user_id(user_api_key_dict, data.target_user_id)

    if data.gpu_count > MAX_GPUS:
        raise HTTPException(400, f"gpu_count must be <= {MAX_GPUS}")

    ok, reason = await _can_book(user_id)
    if not ok:
        raise HTTPException(409, reason)

    node_ip, gpu_ids = await _allocate(data.gpu_count, data.preferred_node)
    if not node_ip:
        raise HTTPException(409, "No GPUs available on any node")

    booking = await prisma.db.litellm_playgroundbooking.create(
        data={
            "user_id": user_id,
            "gpu_count": data.gpu_count,
            "preferred_node": data.preferred_node,
            "allocated_node": node_ip,
            "allocated_gpus": gpu_ids,
            "night_of": _tonight_dt(),
            "is_overflow": _booking_phase() == "overflow",
        }
    )
    verbose_proxy_logger.info(
        f"playground: booking {booking.booking_id} user={user_id} gpus={gpu_ids} node={node_ip}"
    )
    return PlaygroundBookingResponse(**booking.model_dump())


@router.get(
    "/playground/bookings/me",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=List[PlaygroundBookingResponse],
)
@management_endpoint_wrapper
async def list_my_playground_bookings(
    request: Request,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[PlaygroundBookingResponse]:
    """Caller's recent bookings (last 30 days, newest first)."""
    prisma = _prisma()
    user_id = _effective_user_id(user_api_key_dict, target_user_id)
    since = datetime.now(timezone.utc) - timedelta(days=30)
    rows = await prisma.db.litellm_playgroundbooking.find_many(
        where={"user_id": user_id, "created_at": {"gte": since}},
        order={"created_at": "desc"},
    )
    return [PlaygroundBookingResponse(**r.model_dump()) for r in rows]


@router.delete(
    "/playground/bookings/{booking_id}",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundBookingResponse,
)
@management_endpoint_wrapper
async def cancel_playground_booking(
    request: Request,
    booking_id: str,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundBookingResponse:
    """Cancel an allocated booking. Only the owner (or an admin acting on
    behalf via target_user_id) may cancel."""
    prisma = _prisma()
    booking = await prisma.db.litellm_playgroundbooking.find_unique(
        where={"booking_id": booking_id}
    )
    if booking is None:
        raise HTTPException(404, "booking not found")

    if booking.user_id != _effective_user_id(user_api_key_dict, target_user_id):
        raise HTTPException(403, "not your booking")

    if booking.status != "allocated":
        raise HTTPException(
            409, f"cannot cancel booking in status '{booking.status}'"
        )

    updated = await prisma.db.litellm_playgroundbooking.update(
        where={"booking_id": booking_id},
        data={"status": "cancelled"},
    )
    return PlaygroundBookingResponse(**updated.model_dump())


# ─── ssh keys ───────────────────────────────────────────────────────────────


@router.get(
    "/playground/ssh-keys",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=List[PlaygroundSSHKeyResponse],
)
@management_endpoint_wrapper
async def list_playground_ssh_keys(
    request: Request,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[PlaygroundSSHKeyResponse]:
    prisma = _prisma()
    user_id = _effective_user_id(user_api_key_dict, target_user_id)
    rows = await prisma.db.litellm_usersshkey.find_many(
        where={"user_id": user_id},
        order={"created_at": "desc"},
    )
    return [PlaygroundSSHKeyResponse(**r.model_dump()) for r in rows]


@router.post(
    "/playground/ssh-keys",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundSSHKeyResponse,
)
@management_endpoint_wrapper
async def add_playground_ssh_key(
    request: Request,
    data: AddPlaygroundSSHKeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundSSHKeyResponse:
    prisma = _prisma()
    user_id = _effective_user_id(user_api_key_dict, data.target_user_id)

    public_key = data.public_key.strip()
    name = data.name.strip()
    if not public_key or not name:
        raise HTTPException(400, "public_key and name are required")

    fingerprint = _ssh_fingerprint(public_key)

    if await prisma.db.litellm_usersshkey.find_unique(where={"fingerprint": fingerprint}):
        raise HTTPException(409, "SSH key already registered")

    created = await prisma.db.litellm_usersshkey.create(
        data={
            "user_id": user_id,
            "public_key": public_key,
            "fingerprint": fingerprint,
            "name": name,
        }
    )
    return PlaygroundSSHKeyResponse(**created.model_dump())


@router.delete(
    "/playground/ssh-keys/{ssh_key_id}",
    tags=_TAGS,
    dependencies=_DEPS,
)
@management_endpoint_wrapper
async def delete_playground_ssh_key(
    request: Request,
    ssh_key_id: str,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Delete an SSH key. Only the owner (or admin via target_user_id) may delete."""
    prisma = _prisma()
    key_row = await prisma.db.litellm_usersshkey.find_unique(
        where={"ssh_key_id": ssh_key_id}
    )
    if key_row is None:
        raise HTTPException(404, "SSH key not found")

    if key_row.user_id != _effective_user_id(user_api_key_dict, target_user_id):
        raise HTTPException(403, "not your SSH key")

    await prisma.db.litellm_usersshkey.delete(where={"ssh_key_id": ssh_key_id})
    return {"success": True}


# ─── admin: nodes ───────────────────────────────────────────────────────────


@router.get(
    "/playground/admin/nodes",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=List[PlaygroundNodeResponse],
)
@management_endpoint_wrapper
async def admin_list_playground_nodes(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[PlaygroundNodeResponse]:
    _require_admin(user_api_key_dict)
    nodes = await _prisma().db.litellm_playgroundnode.find_many(order={"ip_address": "asc"})
    return [PlaygroundNodeResponse(**n.model_dump()) for n in nodes]


@router.post(
    "/playground/admin/nodes",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundNodeResponse,
)
@management_endpoint_wrapper
async def admin_create_playground_node(
    request: Request,
    data: CreatePlaygroundNodeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundNodeResponse:
    _require_admin(user_api_key_dict)
    created = await _prisma().db.litellm_playgroundnode.create(
        data=data.model_dump(exclude_unset=True)
    )
    verbose_proxy_logger.info(
        f"playground: node {created.node_id} registered ({created.ip_address})"
    )
    return PlaygroundNodeResponse(**created.model_dump())


@router.put(
    "/playground/admin/nodes/{node_id}",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundNodeResponse,
)
@management_endpoint_wrapper
async def admin_update_playground_node(
    request: Request,
    node_id: str,
    data: UpdatePlaygroundNodeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundNodeResponse:
    _require_admin(user_api_key_dict)
    fields = data.model_dump(exclude_unset=True, exclude_none=True)
    if not fields:
        raise HTTPException(400, "no update fields provided")
    try:
        updated = await _prisma().db.litellm_playgroundnode.update(
            where={"node_id": node_id}, data=fields
        )
    except RecordNotFoundError:
        raise HTTPException(404, "node not found")
    return PlaygroundNodeResponse(**updated.model_dump())


@router.delete(
    "/playground/admin/nodes/{node_id}",
    tags=_TAGS,
    dependencies=_DEPS,
)
@management_endpoint_wrapper
async def admin_delete_playground_node(
    request: Request,
    node_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Delete a node. ON DELETE RESTRICT on the booking FK blocks this while
    any bookings still reference the node's ip_address — intentional guardrail."""
    _require_admin(user_api_key_dict)
    prisma = _prisma()
    if await prisma.db.litellm_playgroundnode.find_unique(where={"node_id": node_id}) is None:
        raise HTTPException(404, "node not found")
    try:
        await prisma.db.litellm_playgroundnode.delete(where={"node_id": node_id})
    except Exception as e:  # noqa: BLE001
        verbose_proxy_logger.exception(
            "playground: failed to delete node %s: %s", node_id, e
        )
        raise HTTPException(409, "cannot delete node with existing bookings")
    return {"success": True}


@router.get(
    "/playground/admin/status",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundAdminStatusResponse,
)
@management_endpoint_wrapper
async def admin_playground_status(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundAdminStatusResponse:
    _require_admin(user_api_key_dict)
    prisma = _prisma()
    tonight = _tonight()
    nodes = await prisma.db.litellm_playgroundnode.find_many(order={"ip_address": "asc"})
    bookings = await prisma.db.litellm_playgroundbooking.find_many(
        where={"night_of": _tonight_dt(), "status": {"not": "cancelled"}}
    )
    by_node: Dict[str, int] = {}
    for b in bookings:
        by_node[b.allocated_node] = by_node.get(b.allocated_node, 0) + 1
    return PlaygroundAdminStatusResponse(
        night_of=tonight,
        booking_phase=_booking_phase(),
        total_bookings_tonight=len(bookings),
        nodes=[PlaygroundNodeResponse(**n.model_dump()) for n in nodes],
        bookings_by_node=by_node,
    )


# ─── internal: cron ─────────────────────────────────────────────────────────


@router.get(
    "/playground/internal/allocations-tonight",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundAllocationsTonightResponse,
)
@management_endpoint_wrapper
async def get_playground_allocations_tonight(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundAllocationsTonightResponse:
    """Tonight's allocated bookings grouped by node, with each user's first SSH key."""
    _require_admin(user_api_key_dict)
    prisma = _prisma()
    tonight = _tonight()

    bookings = await prisma.db.litellm_playgroundbooking.find_many(
        where={"night_of": _tonight_dt(), "status": "allocated"},
        order={"created_at": "asc"},
    )
    if not bookings:
        return PlaygroundAllocationsTonightResponse(night_of=tonight, nodes=[])

    user_ids = list({b.user_id for b in bookings})
    node_by_ip = {
        n.ip_address: n for n in await prisma.db.litellm_playgroundnode.find_many()
    }

    first_key: Dict[str, str] = {}
    for k in await prisma.db.litellm_usersshkey.find_many(
        where={"user_id": {"in": user_ids}}, order={"created_at": "asc"}
    ):
        first_key.setdefault(k.user_id, k.public_key)

    email_by_id = {
        u.user_id: getattr(u, "user_email", None)
        for u in await prisma.db.litellm_usertable.find_many(
            where={"user_id": {"in": user_ids}}
        )
    }

    grouped: Dict[str, PlaygroundAllocationNode] = {}
    for b in bookings:
        node = node_by_ip[b.allocated_node]
        if b.allocated_node not in grouped:
            grouped[b.allocated_node] = PlaygroundAllocationNode(
                node_ip=node.ip_address,
                ssh_user=node.ssh_user,
                model_path=node.model_path,
                bookings=[],
            )
        grouped[b.allocated_node].bookings.append(
            PlaygroundAllocationBooking(
                booking_id=b.booking_id,
                user_id=b.user_id,
                user_email=email_by_id.get(b.user_id),
                gpu_devices=b.allocated_gpus,
                ssh_public_key=first_key.get(b.user_id, ""),
            )
        )

    return PlaygroundAllocationsTonightResponse(
        night_of=tonight, nodes=list(grouped.values())
    )


@router.post(
    "/playground/internal/activation-status",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=ActivationStatusReportResponse,
)
@management_endpoint_wrapper
async def report_playground_activation_status(
    request: Request,
    data: ActivationStatusReportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ActivationStatusReportResponse:
    """Cron reports activation results — flip each booking to active/activation_failed."""
    _require_admin(user_api_key_dict)
    prisma = _prisma()

    updated: List[PlaygroundBookingResponse] = []
    for item in data.results:
        payload: Dict[str, Any] = {"status": item.status}
        if item.container_id:
            payload["container_id"] = item.container_id
        if item.status == "activation_failed":
            verbose_proxy_logger.warning(
                f"playground: activation_failed booking={item.booking_id} "
                f"error={item.error or 'unspecified'}"
            )
        try:
            row = await prisma.db.litellm_playgroundbooking.update(
                where={"booking_id": item.booking_id}, data=payload
            )
            updated.append(PlaygroundBookingResponse(**row.model_dump()))
        except RecordNotFoundError:
            verbose_proxy_logger.warning(
                f"playground: activation-status skipped unknown booking={item.booking_id}"
            )

    verbose_proxy_logger.info(
        f"playground: activation-status updated {len(updated)} bookings"
    )
    return ActivationStatusReportResponse(updated_count=len(updated), bookings=updated)


@router.post(
    "/playground/internal/teardown-status",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=TeardownStatusResponse,
)
@management_endpoint_wrapper
async def report_playground_teardown_status(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TeardownStatusResponse:
    """Flip every `active` booking to `terminated`. No night_of filter — the
    cron runs the morning after activation so today's date won't match the
    booking's night_of, and only one night of bookings can be active at once."""
    _require_admin(user_api_key_dict)
    prisma = _prisma()

    active = await prisma.db.litellm_playgroundbooking.find_many(
        where={"status": "active"}
    )
    terminated: List[PlaygroundBookingResponse] = []
    for b in active:
        row = await prisma.db.litellm_playgroundbooking.update(
            where={"booking_id": b.booking_id},
            data={"status": "terminated"},
        )
        if row:
            terminated.append(PlaygroundBookingResponse(**row.model_dump()))

    night = max((b.night_of for b in terminated), default=_tonight())
    verbose_proxy_logger.info(
        f"playground: teardown-status terminated {len(terminated)} bookings "
        f"(latest night_of={night})"
    )
    return TeardownStatusResponse(
        night_of=night, terminated_count=len(terminated), bookings=terminated
    )
