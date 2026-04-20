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
  GET    /playground/internal/pending-teardowns
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
    PlaygroundPendingTeardownBooking,
    PlaygroundPendingTeardownNode,
    PlaygroundPendingTeardownsResponse,
    PlaygroundSlotNode,
    PlaygroundSlotsResponse,
    PlaygroundSSHKeyResponse,
    TeardownStatusReportRequest,
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
# How far ahead users can book (days, inclusive). Today is day 0, so the
# default of 7 means today + the next 7 calendar nights.
BOOKING_HORIZON_DAYS = int(os.getenv("PLAYGROUND_BOOKING_HORIZON_DAYS", "7"))

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


def _night_of_dt(night: date) -> datetime:
    """Same serialization rule as _tonight_dt but for an arbitrary date."""
    return datetime.combine(night, datetime.min.time(), tzinfo=timezone.utc)


async def _can_book(user_id: str, night_of: date) -> Tuple[bool, str]:
    """Gate booking creation for a specific night:
      - SSH key required
      - night_of within [today, today + BOOKING_HORIZON_DAYS]
      - for same-day bookings, tonight's cutoff hasn't passed
      - user doesn't already hold a (non-cancelled) booking for night_of
      - weekly limit honored, except during tonight's overflow window
    """
    prisma = _prisma()

    if not await prisma.db.litellm_usersshkey.count(where={"user_id": user_id}):
        return False, "Register an SSH key before booking"

    today = _tonight()
    if night_of < today:
        return False, "Cannot book a night in the past"
    if night_of > today + timedelta(days=BOOKING_HORIZON_DAYS):
        return (
            False,
            f"Bookings are open up to {BOOKING_HORIZON_DAYS} days ahead",
        )

    is_tonight = night_of == today
    # Cutoff / overflow phase only applies to tonight. Future-night bookings
    # are always "open" regardless of the current wall clock.
    phase = _booking_phase() if is_tonight else "open"
    if is_tonight and phase == "closed":
        return False, "Booking cutoff has passed for tonight"

    # A booking "holds a seat" only while it's allocated or active; cancelled,
    # terminated, and activation_failed bookings have released their seat and
    # must not block a fresh booking for the same night. Using a positive
    # allowlist so any new terminal status added in the future doesn't
    # accidentally gate retries.
    if await prisma.db.litellm_playgroundbooking.count(
        where={
            "user_id": user_id,
            "night_of": _night_of_dt(night_of),
            "status": {"in": ["allocated", "active"]},
        }
    ):
        return False, f"You already have a booking for {night_of.isoformat()}"

    # Overflow window relaxes the weekly limit but not the per-night unique check.
    # Weekly count measures actual GPU usage, so terminated (successful nightly
    # wipe) counts while cancelled / activation_failed do not — a user who was
    # never provisioned shouldn't burn a slot.
    if phase == "open":
        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        recent = await prisma.db.litellm_playgroundbooking.count(
            where={
                "user_id": user_id,
                "created_at": {"gte": week_ago},
                "status": {"in": ["allocated", "active", "terminated"]},
            }
        )
        if recent >= WEEKLY_LIMIT:
            return False, f"Weekly booking limit ({WEEKLY_LIMIT}) reached"

    return True, ""


def _parse_gpu_indices(raw: str) -> List[int]:
    """Parse 'a,b,c' → [a, b, c]. Raises HTTP 400 on any malformed input.

    Enforces non-empty, unique, MAX_GPUS ceiling. Range-vs-total_gpus check
    happens in _reserve_seats once the node is known.
    """
    try:
        parsed = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        raise HTTPException(400, "gpu_indices must be comma-separated integers")
    if not parsed:
        raise HTTPException(400, "gpu_indices cannot be empty")
    if len(parsed) != len(set(parsed)):
        raise HTTPException(400, "gpu_indices contains duplicates")
    if len(parsed) > MAX_GPUS:
        raise HTTPException(400, f"Cannot book more than {MAX_GPUS} GPUs at once")
    return parsed


async def _reserve_seats(night_of: date, node_name: str, gpu_indices_str: str):
    """Resolve node_name, validate gpu_indices against node capacity and
    existing bookings for night_of. Returns (node, sorted_indices) on success,
    raises HTTPException on any collision or validation failure.

    Known race: the overlap check and the subsequent INSERT are not in a
    single DB transaction, so two concurrent callers requesting overlapping
    seats could both succeed. Acceptable at current user count; if it bites,
    wrap with a pg advisory lock keyed on (node_id, night_of).
    """
    prisma = _prisma()

    indices = _parse_gpu_indices(gpu_indices_str)

    node = await prisma.db.litellm_playgroundnode.find_first(
        where={"name": node_name}
    )
    if node is None:
        raise HTTPException(404, f"Node '{node_name}' not found")
    if not node.is_playground_eligible:
        raise HTTPException(409, f"Node '{node_name}' is not playground-eligible")
    if not node.is_healthy:
        raise HTTPException(409, f"Node '{node_name}' is currently unhealthy")

    out_of_range = [i for i in indices if i < 0 or i >= node.total_gpus]
    if out_of_range:
        raise HTTPException(
            400,
            f"GPU indices {out_of_range} out of range for node "
            f"'{node_name}' (valid: 0..{node.total_gpus - 1})",
        )

    # A seat is occupied if a live booking holds it (allocated/active) OR if
    # a cancelled booking still has a container_id — the physical container
    # lives on until the next teardown cycle, so activating a new booking on
    # those GPUs would collide.
    existing = await prisma.db.litellm_playgroundbooking.find_many(
        where={
            "allocated_node": node.ip_address,
            "night_of": _night_of_dt(night_of),
            "OR": [
                {"status": {"in": ["allocated", "active"]}},
                {"status": "cancelled", "container_id": {"not": None}},
            ],
        }
    )
    already_booked = {
        int(g)
        for b in existing
        for g in (b.allocated_gpus or "").split(",")
        if g.strip() != ""
    }
    conflicts = sorted(i for i in indices if i in already_booked)
    if conflicts:
        raise HTTPException(
            409,
            f"GPU indices {conflicts} already booked on '{node_name}' "
            f"for {night_of.isoformat()}",
        )

    return node, sorted(indices)


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
    night_of: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundSlotsResponse:
    """Per-node GPU availability for a given night + booking phase.

    Query params:
      night_of (YYYY-MM-DD, default = today IST): which night to show.
    """
    prisma = _prisma()

    if night_of is not None:
        try:
            night = date.fromisoformat(night_of)
        except ValueError:
            raise HTTPException(400, "night_of must be YYYY-MM-DD")
    else:
        night = _tonight()

    today = _tonight()
    if night < today:
        raise HTTPException(400, "Cannot query slots for a past night")
    if night > today + timedelta(days=BOOKING_HORIZON_DAYS):
        raise HTTPException(
            400,
            f"night_of is beyond the {BOOKING_HORIZON_DAYS}-day booking horizon",
        )

    nodes = await prisma.db.litellm_playgroundnode.find_many(
        where={"is_playground_eligible": True, "is_healthy": True},
    )
    # Mirror _reserve_seats's occupancy logic: cancelled-with-container
    # bookings still hold their seat until the container is actually torn
    # down, so the UI must render those GPUs as booked.
    bookings = await prisma.db.litellm_playgroundbooking.find_many(
        where={
            "night_of": _night_of_dt(night),
            "OR": [
                {"status": {"in": ["allocated", "active"]}},
                {"status": "cancelled", "container_id": {"not": None}},
            ],
        }
    )

    booked_by_node: Dict[str, List[int]] = {}
    for b in bookings:
        for g in (b.allocated_gpus or "").split(","):
            g = g.strip()
            if g == "":
                continue
            booked_by_node.setdefault(b.allocated_node, []).append(int(g))

    # Phase only applies when we're showing tonight; future-night slots are
    # always reported as "open" so the frontend renders them as bookable.
    phase = _booking_phase() if night == today else "open"

    return PlaygroundSlotsResponse(
        night_of=night,
        booking_phase=phase,
        nodes=[
            PlaygroundSlotNode(
                node_id=n.node_id,
                name=n.name,
                ip_address=n.ip_address,
                gpu_type=n.gpu_type,
                total_gpus=n.total_gpus,
                available_gpus=max(
                    0, n.total_gpus - len(booked_by_node.get(n.ip_address, []))
                ),
                booked_gpu_indices=sorted(booked_by_node.get(n.ip_address, [])),
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
    """Create a seat-style booking: user picks a specific night + node + GPU
    indices from the frontend seat grid. The server validates, reserves, and
    persists the exact seats the caller asked for (no allocator fallback)."""
    prisma = _prisma()
    user_id = _effective_user_id(user_api_key_dict, data.target_user_id)

    ok, reason = await _can_book(user_id, data.night_of)
    if not ok:
        raise HTTPException(409, reason)

    node, indices = await _reserve_seats(data.night_of, data.node_name, data.gpu_indices)

    # `is_overflow` is a same-day-after-OVERFLOW_HOUR distinction; future-night
    # bookings are never overflow regardless of the wall clock.
    is_overflow = (
        data.night_of == _tonight() and _booking_phase() == "overflow"
    )

    booking = await prisma.db.litellm_playgroundbooking.create(
        data={
            "user_id": user_id,
            "gpu_count": len(indices),
            "preferred_node": None,
            "allocated_node": node.ip_address,
            "allocated_gpus": ",".join(str(i) for i in indices),
            "night_of": _night_of_dt(data.night_of),
            "is_overflow": is_overflow,
        }
    )
    verbose_proxy_logger.info(
        f"playground: booking {booking.booking_id} user={user_id} "
        f"gpus={indices} node={node.ip_address} night={data.night_of.isoformat()}"
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

    if booking.status not in ("allocated", "active"):
        raise HTTPException(
            409, f"cannot cancel booking in status '{booking.status}'"
        )

    # Cancelling an `active` booking flips the DB row immediately but leaves
    # the running container on the node — the nightly teardown cron (or the
    # targeted teardown worker, once implemented) will deprovision it. The
    # conflict check in _reserve_seats treats cancelled-with-container as
    # still holding the seat so no one can rebook those GPUs tonight and
    # crash the activator when it tries to add them on top of the stale
    # container.
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
    payload = data.model_dump(exclude_unset=True)
    # Whitespace in ip_address silently breaks downstream SSH targets
    # (saw ' 103.48.42.12' registered once from a manual admin call).
    # Strip here so the gateway cron and authkeys command= entries always
    # see a clean address.
    if "ip_address" in payload and isinstance(payload["ip_address"], str):
        payload["ip_address"] = payload["ip_address"].strip()
    created = await _prisma().db.litellm_playgroundnode.create(data=payload)
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
    if "ip_address" in fields and isinstance(fields["ip_address"], str):
        fields["ip_address"] = fields["ip_address"].strip()
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
    data: Optional[TeardownStatusReportRequest] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TeardownStatusResponse:
    """Two modes:

    - Sweep (nightly cron, no body or `results=None`): flip every `active`
      booking AND every `cancelled` booking with a container_id set to
      `terminated`, clearing container_id. Covers both normal overnight
      bookings and user-cancelled bookings whose container survived until
      the nightly wipe.
    - Per-booking (spark-activator worker, `results=[...]`): for each item,
      if status is `terminated`, flip that booking to `terminated` and
      clear container_id. `teardown_failed` is logged and left as-is so
      the worker retries next cycle.

    No night_of filter on sweep — only one night of bookings can be
    `active` at a time, and cancelled-with-container rows are orthogonal
    to any notion of "tonight".
    """
    _require_admin(user_api_key_dict)
    prisma = _prisma()
    terminated: List[PlaygroundBookingResponse] = []

    if data is None or not data.results:
        sweep = await prisma.db.litellm_playgroundbooking.find_many(
            where={
                "OR": [
                    {"status": "active"},
                    {"status": "cancelled", "container_id": {"not": None}},
                ]
            }
        )
        for b in sweep:
            row = await prisma.db.litellm_playgroundbooking.update(
                where={"booking_id": b.booking_id},
                data={"status": "terminated", "container_id": None},
            )
            if row:
                terminated.append(PlaygroundBookingResponse(**row.model_dump()))
    else:
        for item in data.results:
            if item.status == "teardown_failed":
                verbose_proxy_logger.warning(
                    f"playground: teardown_failed booking={item.booking_id} "
                    f"error={item.error or 'unspecified'}"
                )
                continue
            try:
                row = await prisma.db.litellm_playgroundbooking.update(
                    where={"booking_id": item.booking_id},
                    data={"status": "terminated", "container_id": None},
                )
                terminated.append(PlaygroundBookingResponse(**row.model_dump()))
            except RecordNotFoundError:
                verbose_proxy_logger.warning(
                    f"playground: teardown-status skipped unknown booking={item.booking_id}"
                )

    night = max((b.night_of for b in terminated), default=_tonight())
    verbose_proxy_logger.info(
        f"playground: teardown-status terminated {len(terminated)} bookings "
        f"(latest night_of={night})"
    )
    return TeardownStatusResponse(
        night_of=night, terminated_count=len(terminated), bookings=terminated
    )


@router.get(
    "/playground/internal/pending-teardowns",
    tags=_TAGS,
    dependencies=_DEPS,
    response_model=PlaygroundPendingTeardownsResponse,
)
@management_endpoint_wrapper
async def get_playground_pending_teardowns(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundPendingTeardownsResponse:
    """Cancelled bookings whose container is still up. The spark-activator
    worker polls this and runs manage-users.sh remove for each, then POSTs
    teardown-status with the booking_ids to finalize (status=terminated,
    container_id=NULL). Grouped by allocated_node to match the shape of
    allocations-tonight so the worker can reuse its SSH plumbing."""
    _require_admin(user_api_key_dict)
    prisma = _prisma()

    bookings = await prisma.db.litellm_playgroundbooking.find_many(
        where={"status": "cancelled", "container_id": {"not": None}},
        order={"created_at": "asc"},
    )
    if not bookings:
        return PlaygroundPendingTeardownsResponse(nodes=[])

    user_ids = list({b.user_id for b in bookings})
    node_by_ip = {
        n.ip_address: n for n in await prisma.db.litellm_playgroundnode.find_many()
    }
    email_by_id = {
        u.user_id: getattr(u, "user_email", None)
        for u in await prisma.db.litellm_usertable.find_many(
            where={"user_id": {"in": user_ids}}
        )
    }

    grouped: Dict[str, PlaygroundPendingTeardownNode] = {}
    for b in bookings:
        node = node_by_ip.get(b.allocated_node)
        if node is None:
            # Node was deleted after the booking was cancelled — nothing to SSH
            # into. Skip and let a future cleanup task drop the orphan row.
            verbose_proxy_logger.warning(
                f"playground: pending-teardowns skipped booking={b.booking_id} "
                f"— allocated_node={b.allocated_node} no longer registered"
            )
            continue
        if b.allocated_node not in grouped:
            grouped[b.allocated_node] = PlaygroundPendingTeardownNode(
                node_ip=node.ip_address,
                ssh_user=node.ssh_user,
                bookings=[],
            )
        grouped[b.allocated_node].bookings.append(
            PlaygroundPendingTeardownBooking(
                booking_id=b.booking_id,
                user_id=b.user_id,
                user_email=email_by_id.get(b.user_id),
                gpu_devices=b.allocated_gpus or "",
                container_id=b.container_id or "",
            )
        )

    return PlaygroundPendingTeardownsResponse(nodes=list(grouped.values()))
