# Forgot Password (self-service) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an internal (non-SSO) LiteLLM proxy user who forgot their password request a reset link by email from the login screen, with no admin involvement, and use it once to set a new password.

**Architecture:** A new `LiteLLM_PasswordResetToken` Prisma table stores SHA-256-hashed, single-use, 30-minute tokens. Three new FastAPI endpoints (unauthenticated, mirroring the existing `/onboarding/get_token` + `/onboarding/claim_token` pattern) issue, validate, and consume tokens, reusing the existing `send_email`, `hash_password`, and `hash_token` utilities and the proxy-wide `user_api_key_cache` (`DualCache`) for rate limiting. Two new Next.js pages (`/ui/forgot-password`, `/ui/reset-password`) mirror the existing onboarding page's component structure (loading/error/form views).

**Tech Stack:** Python 3 (FastAPI, Prisma Python client), TypeScript/React (Next.js app router, antd, TanStack Query), pytest (+ pytest-mock), Vitest + React Testing Library.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-28-forgot-password-design.md` — every requirement in it must be covered.
- No comments in new code unless explicitly requested.
- Every function parameter fully typed; no `Any`/bare `dict`.
- Composition over inheritance, early returns, no mutation of existing variables.
- Follow existing patterns exactly (see file-by-file precedents cited in each task) rather than inventing new ones.
- Python max line length: 120.
- `tests/test_litellm/` mirrors `litellm/` in a parallel path; one new test file per new source file.
- Frontend: colocate `<Component>.test.tsx` next to the component, Vitest + RTL, following `OnboardingFormBody.test.tsx` conventions.
- Run `make format && make lint-ruff && make lint-basedpyright` (backend) / `npm run test` + `npm run build` (frontend) before each commit; run `make pre-commit` right before committing.
- Conventional Commits for every commit message.
- Never commit directly to `main`; this work happens on `litellm_forgot_password_design` (branched from `upstream/litellm_internal_staging`).

---

## Task 1: Prisma schema — add `LiteLLM_PasswordResetToken`

**Files:**
- Modify: `litellm/proxy/schema.prisma:234-268` (add table + back-relation on `LiteLLM_UserTable`)
- Modify: `schema.prisma` (root copy, keep byte-identical to `litellm/proxy/schema.prisma`)
- Modify: `litellm-proxy-extras/litellm_proxy_extras/schema.prisma` (third copy, keep byte-identical)
- Create: `litellm-proxy-extras/litellm_proxy_extras/migrations/20260728120000_add_password_reset_token/migration.sql`

**Interfaces:**
- Produces: Prisma model `LiteLLM_PasswordResetToken` with fields `token_hash: String @id`, `user_id: String`, `requested_ip: String?`, `created_at: DateTime`, `expires_at: DateTime`, `used_at: DateTime?`, and relation `user LiteLLM_UserTable`. Generated Python accessor: `prisma_client.db.litellm_passwordresettoken`.

- [ ] **Step 1: Add the back-relation field to `LiteLLM_UserTable`**

In `litellm/proxy/schema.prisma`, inside the `LiteLLM_UserTable` model (the block starting at line 234), add one line among the other relation fields (right after `invitations_user`):

```prisma
    invitations_user    LiteLLM_InvitationLink[] @relation("UserId")
    password_reset_tokens LiteLLM_PasswordResetToken[]
    object_permission LiteLLM_ObjectPermissionTable?   @relation(fields: [object_permission_id], references: [object_permission_id])
```

- [ ] **Step 2: Add the new model**

Immediately after the closing `}` of `LiteLLM_InvitationLink` (around line 718 of `litellm/proxy/schema.prisma`), add:

```prisma
model LiteLLM_PasswordResetToken {
  // self-service "forgot password" tokens; independent of LiteLLM_InvitationLink
  token_hash   String    @id
  user_id      String
  requested_ip String?
  created_at   DateTime  @default(now())
  expires_at   DateTime
  used_at      DateTime?

  user LiteLLM_UserTable @relation(fields: [user_id], references: [user_id], onDelete: Cascade)

  @@index([user_id])
}
```

- [ ] **Step 3: Apply the same two edits to the root `schema.prisma` and the `litellm-proxy-extras` copy**

```bash
cp litellm/proxy/schema.prisma schema.prisma
cp litellm/proxy/schema.prisma litellm-proxy-extras/litellm_proxy_extras/schema.prisma
diff litellm/proxy/schema.prisma schema.prisma
diff litellm/proxy/schema.prisma litellm-proxy-extras/litellm_proxy_extras/schema.prisma
```

Expected: both `diff` commands print nothing (files identical).

- [ ] **Step 4: Write the migration SQL**

Create `litellm-proxy-extras/litellm_proxy_extras/migrations/20260728120000_add_password_reset_token/migration.sql`:

```sql
-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_PasswordResetToken" (
    "token_hash" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "requested_ip" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "used_at" TIMESTAMP(3),

    CONSTRAINT "LiteLLM_PasswordResetToken_pkey" PRIMARY KEY ("token_hash")
);

-- CreateIndex
CREATE INDEX "LiteLLM_PasswordResetToken_user_id_idx" ON "LiteLLM_PasswordResetToken"("user_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_PasswordResetToken" ADD CONSTRAINT "LiteLLM_PasswordResetToken_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id") ON DELETE CASCADE ON UPDATE CASCADE;
```

- [ ] **Step 5: Regenerate the Prisma client and verify**

```bash
uv run python scripts/prisma_generate_if_needed.py
uv run prisma validate --schema litellm/proxy/schema.prisma
```

Expected: both commands exit 0 with no schema errors. `prisma validate` will fail loudly if the back-relation from Step 1 is missing or mismatched.

- [ ] **Step 6: Commit**

```bash
git add schema.prisma litellm/proxy/schema.prisma litellm-proxy-extras/litellm_proxy_extras/schema.prisma litellm-proxy-extras/litellm_proxy_extras/migrations/20260728120000_add_password_reset_token/
git commit -m "feat(db): add LiteLLM_PasswordResetToken table"
```

---

## Task 2: Backend model + repository

**Files:**
- Create: `litellm/models/password_reset_token.py`
- Create: `litellm/repositories/password_reset_token_repository.py`
- Test: `tests/test_litellm/repositories/test_password_reset_token_repository.py`

**Interfaces:**
- Consumes: `litellm.types.llms.base.LiteLLMPydanticObjectBase` (base class, same as `litellm/models/user.py:18`), `litellm.repositories.base_repository.BaseRepository` (generic `create`, `update`, `find_by_id`, `find_many` — see `litellm/repositories/base_repository.py:63-113`).
- Produces: `LiteLLM_PasswordResetToken` Pydantic model (fields: `token_hash: str`, `user_id: str`, `requested_ip: Optional[str]`, `created_at: datetime`, `expires_at: datetime`, `used_at: Optional[datetime]`). `PasswordResetTokenRepository(prisma_client)` with `.table` returning `prisma_client.db.litellm_passwordresettoken`, plus two custom async methods: `find_valid_by_hash(token_hash: str, now: datetime) -> Optional[LiteLLM_PasswordResetToken]` and `invalidate_unused_for_user(user_id: str, now: datetime) -> None`. Both are consumed by Task 3/4/5's endpoint code.

- [ ] **Step 1: Write the failing repository test**

Create `tests/test_litellm/repositories/test_password_reset_token_repository.py`:

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.repositories.password_reset_token_repository import (
    PasswordResetTokenRepository,
)


def _mock_prisma():
    prisma_client = MagicMock()
    prisma_client.db.litellm_passwordresettoken = MagicMock()
    return prisma_client


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_none_when_not_found():
    prisma_client = _mock_prisma()
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=None)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=datetime.now(timezone.utc))

    assert result is None


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_none_when_expired():
    prisma_client = _mock_prisma()
    expired_row = MagicMock()
    expired_row.dict.return_value = {
        "token_hash": "abc",
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
        "expires_at": datetime.now(timezone.utc) - timedelta(minutes=1),
        "used_at": None,
    }
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=expired_row)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=datetime.now(timezone.utc))

    assert result is None


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_none_when_already_used():
    prisma_client = _mock_prisma()
    used_row = MagicMock()
    now = datetime.now(timezone.utc)
    used_row.dict.return_value = {
        "token_hash": "abc",
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now - timedelta(minutes=5),
        "expires_at": now + timedelta(minutes=25),
        "used_at": now - timedelta(minutes=1),
    }
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=used_row)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=now)

    assert result is None


@pytest.mark.asyncio
async def test_find_valid_by_hash_returns_model_when_valid():
    prisma_client = _mock_prisma()
    now = datetime.now(timezone.utc)
    valid_row = MagicMock()
    valid_row.dict.return_value = {
        "token_hash": "abc",
        "user_id": "user-1",
        "requested_ip": "127.0.0.1",
        "created_at": now - timedelta(minutes=5),
        "expires_at": now + timedelta(minutes=25),
        "used_at": None,
    }
    prisma_client.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=valid_row)

    repo = PasswordResetTokenRepository(prisma_client)
    result = await repo.find_valid_by_hash(token_hash="abc", now=now)

    assert result is not None
    assert result.user_id == "user-1"


@pytest.mark.asyncio
async def test_invalidate_unused_for_user_calls_update_many():
    prisma_client = _mock_prisma()
    prisma_client.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=2)
    now = datetime.now(timezone.utc)

    repo = PasswordResetTokenRepository(prisma_client)
    await repo.invalidate_unused_for_user(user_id="user-1", now=now)

    prisma_client.db.litellm_passwordresettoken.update_many.assert_awaited_once_with(
        where={"user_id": "user-1", "used_at": None},
        data={"used_at": now},
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_litellm/repositories/test_password_reset_token_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'litellm.repositories.password_reset_token_repository'`

- [ ] **Step 3: Write the model**

Create `litellm/models/password_reset_token.py`:

```python
from datetime import datetime
from typing import Optional

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_PasswordResetToken(LiteLLMPydanticObjectBase):
    token_hash: str
    user_id: str
    requested_ip: Optional[str] = None
    created_at: datetime
    expires_at: datetime
    used_at: Optional[datetime] = None
```

- [ ] **Step 4: Write the repository**

Create `litellm/repositories/password_reset_token_repository.py`:

```python
from datetime import datetime
from typing import Any, Optional, Type

from litellm.models.password_reset_token import LiteLLM_PasswordResetToken
from litellm.repositories.base_repository import BaseRepository


class PasswordResetTokenRepository(BaseRepository[LiteLLM_PasswordResetToken]):
    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_passwordresettoken

    @property
    def model_class(self) -> Type[LiteLLM_PasswordResetToken]:
        return LiteLLM_PasswordResetToken

    async def find_valid_by_hash(self, token_hash: str, now: datetime) -> Optional[LiteLLM_PasswordResetToken]:
        record = await self.table.find_unique(where={"token_hash": token_hash})
        model = self._to_model(record)
        if model is None:
            return None
        if model.used_at is not None:
            return None
        if model.expires_at < now:
            return None
        return model

    async def invalidate_unused_for_user(self, user_id: str, now: datetime) -> None:
        await self.table.update_many(
            where={"user_id": user_id, "used_at": None},
            data={"used_at": now},
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_litellm/repositories/test_password_reset_token_repository.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add litellm/models/password_reset_token.py litellm/repositories/password_reset_token_repository.py tests/test_litellm/repositories/test_password_reset_token_repository.py
git commit -m "feat(db): add PasswordResetToken model and repository"
```

---

## Task 3: Request/response types

**Files:**
- Create: `litellm/types/proxy/management_endpoints/password_reset_endpoints.py`

**Interfaces:**
- Produces: `ForgotPasswordRequest(BaseModel)` with field `email: str`; `ResetPasswordRequest(BaseModel)` with fields `token: str`, `new_password: str`. Consumed by Task 4/5/6's endpoint bodies.

- [ ] **Step 1: Write the types file**

Create `litellm/types/proxy/management_endpoints/password_reset_endpoints.py`:

```python
from pydantic import BaseModel


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `uv run python -c "from litellm.types.proxy.management_endpoints.password_reset_endpoints import ForgotPasswordRequest, ResetPasswordRequest; print(ForgotPasswordRequest(email='a@b.com'))"`
Expected: prints `email='a@b.com'` with no error.

- [ ] **Step 3: Commit**

```bash
git add litellm/types/proxy/management_endpoints/password_reset_endpoints.py
git commit -m "feat(proxy): add forgot/reset password request types"
```

---

## Task 4: `POST /user/forgot_password` endpoint

**Files:**
- Create: `litellm/proxy/management_endpoints/password_reset_endpoints.py`
- Test: `tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py`

**Interfaces:**
- Consumes: `PasswordResetTokenRepository` (Task 2), `ForgotPasswordRequest` (Task 3), `litellm.repositories.user_repository.UserRepository` (existing, `find_first(where=...)` via `.table`), `litellm.proxy.utils.send_email` (`litellm/proxy/utils.py:5163`), `litellm.proxy.utils.hash_token` (`litellm/proxy/utils.py:5232`), `litellm.proxy.utils.get_custom_url`, `user_api_key_cache.async_increment_cache` (`litellm/caching/dual_cache.py:371`).
- Produces: `router` (`APIRouter`) with route `POST /user/forgot_password`. This same `router` instance is extended by Tasks 5 and 6 with the other two routes, then imported by Task 7.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py`:

```python
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.proxy_server import app

client = TestClient(app)


def _mock_user(user_id="user-1", email="alice@example.com", password="scrypt:hash"):
    user = MagicMock()
    user.user_id = user_id
    user.user_email = email
    user.password = password
    return user


@pytest.mark.asyncio
async def test_forgot_password_existing_user_sends_email(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=_mock_user())
    mock_prisma.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=0)
    mock_prisma.db.litellm_passwordresettoken.create = AsyncMock(
        return_value=MagicMock(
            dict=lambda: {
                "token_hash": "h",
                "user_id": "user-1",
                "requested_ip": "testclient",
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc),
                "used_at": None,
            }
        )
    )
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 200
    assert "message" in response.json()
    mock_send_email.assert_awaited_once()
    assert mock_send_email.call_args.kwargs["receiver_email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_forgot_password_unknown_email_same_response_no_email_sent(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "unknown@example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "If an account exists for this email, a password reset link has been sent."
    }
    mock_send_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_forgot_password_sso_only_user_same_response_no_email_sent(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(
        return_value=_mock_user(password=None)
    )
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mock_send_email = mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "message": "If an account exists for this email, a password reset link has been sent."
    }
    mock_send_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_forgot_password_rate_limited_by_email(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=_mock_user())
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    mocker.patch(
        "litellm.proxy.management_endpoints.password_reset_endpoints.send_email",
        new=AsyncMock(),
    )
    mock_cache = mocker.patch("litellm.proxy.proxy_server.user_api_key_cache")
    mock_cache.async_increment_cache = AsyncMock(side_effect=[4.0, 1.0])

    response = client.post("/user/forgot_password", json={"email": "alice@example.com"})

    assert response.status_code == 429
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py -v`
Expected: FAIL — `404` responses / `ModuleNotFoundError`, since neither the module nor the route exist yet.

- [ ] **Step 3: Write the endpoint module**

Create `litellm/proxy/management_endpoints/password_reset_endpoints.py`:

```python
"""
Self-service password reset endpoints for internal (non-SSO) users.

/user/forgot_password
/user/reset_password/validate
/user/reset_password
"""

import secrets
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.utils import get_custom_url, hash_password, hash_token, send_email
from litellm.repositories.password_reset_token_repository import (
    PasswordResetTokenRepository,
)
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.management_endpoints.password_reset_endpoints import (
    ForgotPasswordRequest,
    ResetPasswordRequest,
)

router = APIRouter()

_GENERIC_FORGOT_PASSWORD_MESSAGE = "If an account exists for this email, a password reset link has been sent."
_GENERIC_INVALID_TOKEN_MESSAGE = "This link is invalid or has expired."
_MAX_REQUESTS_PER_EMAIL_PER_HOUR = 3
_MAX_REQUESTS_PER_IP_PER_HOUR = 10
_RESET_TOKEN_TTL_MINUTES = 30
_RATE_LIMIT_WINDOW_SECONDS = 3600


@router.post("/user/forgot_password", include_in_schema=False)
async def forgot_password(data: ForgotPasswordRequest, request: Request):
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    client_ip = request.client.host if request.client else "unknown"
    email_count = await user_api_key_cache.async_increment_cache(
        key=f"password_reset_rl:email:{data.email.lower()}", value=1, ttl=_RATE_LIMIT_WINDOW_SECONDS
    )
    ip_count = await user_api_key_cache.async_increment_cache(
        key=f"password_reset_rl:ip:{client_ip}", value=1, ttl=_RATE_LIMIT_WINDOW_SECONDS
    )

    if (email_count is not None and email_count > _MAX_REQUESTS_PER_EMAIL_PER_HOUR) or (
        ip_count is not None and ip_count > _MAX_REQUESTS_PER_IP_PER_HOUR
    ):
        verbose_proxy_logger.warning("Password reset rate limit exceeded for ip=%s", client_ip)
        raise HTTPException(status_code=429, detail={"error": "Too many requests. Please try again later."})

    user_obj = await UserRepository(prisma_client).table.find_first(
        where={"user_email": {"equals": data.email, "mode": "insensitive"}}
    )

    if user_obj is None or getattr(user_obj, "password", None) is None:
        verbose_proxy_logger.warning("Password reset requested for an unknown or SSO-only email")
        return {"message": _GENERIC_FORGOT_PASSWORD_MESSAGE}

    now = litellm.utils.get_utc_datetime()
    token_repo = PasswordResetTokenRepository(prisma_client)
    await token_repo.invalidate_unused_for_user(user_id=user_obj.user_id, now=now)

    raw_token = secrets.token_urlsafe(32)
    await token_repo.create(
        data={
            "token_hash": hash_token(raw_token),
            "user_id": user_obj.user_id,
            "requested_ip": client_ip,
            "expires_at": now + timedelta(minutes=_RESET_TOKEN_TTL_MINUTES),
        }
    )

    reset_base_url = get_custom_url(str(request.base_url)).rstrip("/") + "/ui/reset-password"
    reset_link = f"{reset_base_url}?token={raw_token}"

    try:
        await send_email(
            receiver_email=data.email,
            subject="Reset your LiteLLM password",
            html=(
                f"<p>Click the link below to reset your password. This link expires in "
                f"{_RESET_TOKEN_TTL_MINUTES} minutes.</p><p><a href='{reset_link}'>{reset_link}</a></p>"
            ),
        )
    except ValueError as e:
        verbose_proxy_logger.warning("Password reset email not sent, SMTP misconfigured: %s", e)

    return {"message": _GENERIC_FORGOT_PASSWORD_MESSAGE}
```

- [ ] **Step 4: Add the router include so `/user/forgot_password` is reachable in tests**

In `litellm/proxy/proxy_server.py`, add the import next to the other `management_endpoints` imports (near line 410, right after the `internal_user_endpoints` import block):

```python
from litellm.proxy.management_endpoints.password_reset_endpoints import (
    router as password_reset_router,
)
```

And add the include next to the other `app.include_router(...)` calls (near line 16316):

```python
app.include_router(password_reset_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add litellm/proxy/management_endpoints/password_reset_endpoints.py litellm/proxy/proxy_server.py tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py
git commit -m "feat(proxy): add POST /user/forgot_password endpoint"
```

---

## Task 5: `GET /user/reset_password/validate` endpoint

**Files:**
- Modify: `litellm/proxy/management_endpoints/password_reset_endpoints.py` (append route to the same `router`)
- Modify: `tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py` (append tests)

**Interfaces:**
- Consumes: `PasswordResetTokenRepository.find_valid_by_hash` (Task 2), `hash_token` (existing).
- Produces: `GET /user/reset_password/validate?token=...` → `{"user_email": str}` on success, `400` otherwise. Consumed by the frontend in Task 10.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py`:

```python
from litellm.proxy.utils import hash_token


@pytest.mark.asyncio
async def test_validate_reset_token_valid(mocker):
    mock_prisma = MagicMock()
    now = datetime.now(timezone.utc)
    token_row = MagicMock()
    token_row.dict.return_value = {
        "token_hash": hash_token("raw-token"),
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now,
        "expires_at": now.replace(year=now.year + 1),
        "used_at": None,
    }
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=token_row)
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=_mock_user())
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.get("/user/reset_password/validate", params={"token": "raw-token"})

    assert response.status_code == 200
    assert response.json() == {"user_email": "alice@example.com"}


@pytest.mark.asyncio
async def test_validate_reset_token_invalid_returns_generic_400(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.get("/user/reset_password/validate", params={"token": "bogus"})

    assert response.status_code == 400
    assert response.json() == {"detail": {"error": "This link is invalid or has expired."}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py -v -k validate_reset_token`
Expected: FAIL with `404 Not Found` (route doesn't exist yet).

- [ ] **Step 3: Append the endpoint**

In `litellm/proxy/management_endpoints/password_reset_endpoints.py`, add below `forgot_password`:

```python
@router.get("/user/reset_password/validate", include_in_schema=False)
async def validate_reset_password_token(token: str):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    token_repo = PasswordResetTokenRepository(prisma_client)
    now = litellm.utils.get_utc_datetime()
    token_row = await token_repo.find_valid_by_hash(token_hash=hash_token(token), now=now)

    if token_row is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    user_obj = await UserRepository(prisma_client).table.find_unique(where={"user_id": token_row.user_id})
    if user_obj is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    return {"user_email": user_obj.user_email}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py -v -k validate_reset_token`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add litellm/proxy/management_endpoints/password_reset_endpoints.py tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py
git commit -m "feat(proxy): add GET /user/reset_password/validate endpoint"
```

---

## Task 6: `POST /user/reset_password` endpoint

**Files:**
- Modify: `litellm/proxy/management_endpoints/password_reset_endpoints.py` (append route)
- Modify: `tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py` (append tests)

**Interfaces:**
- Consumes: `ResetPasswordRequest` (Task 3), `hash_password`/`hash_token` (existing), transaction pattern from `litellm/proxy/proxy_server.py:14046-14078` (`async with prisma_client.db.tx() as tx:`).
- Produces: `POST /user/reset_password` → `{"message": str}` on success (`200`), `400` on invalid/expired/already-used token. No session/token is issued. Consumed by the frontend in Task 10.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py`:

```python
def _install_tx_context(mock_prisma):
    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=mock_prisma.db)
    tx_cm.__aexit__ = AsyncMock(return_value=None)
    mock_prisma.db.tx = MagicMock(return_value=tx_cm)


@pytest.mark.asyncio
async def test_reset_password_happy_path(mocker):
    mock_prisma = MagicMock()
    now = datetime.now(timezone.utc)
    token_row = MagicMock()
    token_row.dict.return_value = {
        "token_hash": hash_token("raw-token"),
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now,
        "expires_at": now.replace(year=now.year + 1),
        "used_at": None,
    }
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=token_row)
    mock_prisma.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=1)
    mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=_mock_user())
    _install_tx_context(mock_prisma)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.post(
        "/user/reset_password", json={"token": "raw-token", "new_password": "correct horse battery staple"}
    )

    assert response.status_code == 200
    mock_prisma.db.litellm_usertable.update.assert_awaited_once()
    _, update_kwargs = mock_prisma.db.litellm_usertable.update.call_args
    assert update_kwargs["data"]["password"] != "correct horse battery staple"


@pytest.mark.asyncio
async def test_reset_password_second_claim_fails(mocker):
    """A token already marked used_at fails the atomic update_many (updated_count == 0)."""
    mock_prisma = MagicMock()
    now = datetime.now(timezone.utc)
    token_row = MagicMock()
    token_row.dict.return_value = {
        "token_hash": hash_token("raw-token"),
        "user_id": "user-1",
        "requested_ip": None,
        "created_at": now,
        "expires_at": now.replace(year=now.year + 1),
        "used_at": None,
    }
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=token_row)
    mock_prisma.db.litellm_passwordresettoken.update_many = AsyncMock(return_value=0)
    _install_tx_context(mock_prisma)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.post(
        "/user/reset_password", json={"token": "raw-token", "new_password": "correct horse battery staple"}
    )

    assert response.status_code == 400
    mock_prisma.db.litellm_usertable.update.assert_not_called()


@pytest.mark.asyncio
async def test_reset_password_expired_token_rejected(mocker):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_passwordresettoken.find_unique = AsyncMock(return_value=None)
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    response = client.post(
        "/user/reset_password", json={"token": "expired-token", "new_password": "correct horse battery staple"}
    )

    assert response.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py -v -k reset_password`
Expected: FAIL with `404 Not Found`.

- [ ] **Step 3: Append the endpoint**

In `litellm/proxy/management_endpoints/password_reset_endpoints.py`, add below `validate_reset_password_token`:

```python
@router.post("/user/reset_password", include_in_schema=False)
async def reset_password(data: ResetPasswordRequest):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    token_hash = hash_token(data.token)
    now = litellm.utils.get_utc_datetime()

    token_row = await PasswordResetTokenRepository(prisma_client).find_valid_by_hash(token_hash=token_hash, now=now)
    if token_row is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    hashed_pw = hash_password(data.new_password)

    async with prisma_client.db.tx() as tx:
        updated_count = await tx.litellm_passwordresettoken.update_many(
            where={"token_hash": token_hash, "used_at": None},
            data={"used_at": now},
        )
        if updated_count == 0:
            raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

        user_obj = await tx.litellm_usertable.update(where={"user_id": token_row.user_id}, data={"password": hashed_pw})
        if user_obj is None:
            raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

        await tx.litellm_passwordresettoken.update_many(
            where={"user_id": token_row.user_id, "used_at": None},
            data={"used_at": now},
        )

    return {"message": "Password reset successfully. Please log in with your new password."}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py -v`
Expected: PASS (all tests in the file, 9 total)

- [ ] **Step 5: Commit**

```bash
git add litellm/proxy/management_endpoints/password_reset_endpoints.py tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py
git commit -m "feat(proxy): add POST /user/reset_password endpoint"
```

---

## Task 7: Backend lint/type/full-suite verification

**Files:** none new — verification only.

- [ ] **Step 1: Run the full backend lint + type check**

```bash
make format
make lint-ruff
make lint-basedpyright
```

Expected: all exit 0. Fix any reported issues in the three files created/modified in Tasks 1-6 (add `# noqa`/`# pyright: ignore[...]` with a reason only if a rule is genuinely unavoidable; prefer fixing the underlying issue).

- [ ] **Step 2: Run the full new test file plus its neighbors**

```bash
uv run pytest tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py tests/test_litellm/repositories/test_password_reset_token_repository.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit any lint/format fixes**

```bash
git add -u
git commit -m "chore(proxy): fix lint findings for password reset endpoints"
```

(Skip this commit if `git status` shows nothing to commit.)

---

## Task 8: Frontend — `networking.tsx` calls

**Files:**
- Modify: `ui/litellm-dashboard/src/components/networking.tsx` (insert after `claimOnboardingToken`, currently ending at line 1527)
- Test: `ui/litellm-dashboard/src/components/networking.forgotPassword.test.ts`

**Interfaces:**
- Consumes: module-scoped `proxyBaseUrl`, `deriveErrorMessage` (both already used by `getOnboardingCredentials`, `networking.tsx:1475-1504`).
- Produces: `forgotPasswordCall(email: string): Promise<{ message: string }>`, `validateResetTokenCall(token: string): Promise<{ user_email: string }>`, `resetPasswordCall(token: string, newPassword: string): Promise<{ message: string }>`. Consumed by Task 9's hooks.

- [ ] **Step 1: Write the failing test**

Create `ui/litellm-dashboard/src/components/networking.forgotPassword.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { forgotPasswordCall, validateResetTokenCall, resetPasswordCall } from "./networking";

describe("forgot/reset password networking calls", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ message: "ok" }),
      }),
    );
  });

  it("forgotPasswordCall posts the email as JSON", async () => {
    await forgotPasswordCall("alice@example.com");
    const [url, options] = (fetch as any).mock.calls[0];
    expect(url).toContain("/user/forgot_password");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({ email: "alice@example.com" });
  });

  it("validateResetTokenCall issues a GET with the token as a query param", async () => {
    (fetch as any).mockResolvedValueOnce({ ok: true, json: async () => ({ user_email: "alice@example.com" }) });
    await validateResetTokenCall("tok-123");
    const [url, options] = (fetch as any).mock.calls[0];
    expect(url).toContain("/user/reset_password/validate?token=tok-123");
    expect(options.method).toBe("GET");
  });

  it("resetPasswordCall posts token and new_password as JSON", async () => {
    await resetPasswordCall("tok-123", "new-secret");
    const [url, options] = (fetch as any).mock.calls[0];
    expect(url).toContain("/user/reset_password");
    expect(JSON.parse(options.body)).toEqual({ token: "tok-123", new_password: "new-secret" });
  });

  it("throws with the derived error message on a non-ok response", async () => {
    (fetch as any).mockResolvedValueOnce({ ok: false, json: async () => ({ detail: { error: "boom" } }) });
    await expect(forgotPasswordCall("alice@example.com")).rejects.toThrow();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui/litellm-dashboard && npm run test -- networking.forgotPassword.test.ts`
Expected: FAIL — `forgotPasswordCall is not a function` (or similar import error).

- [ ] **Step 3: Add the three functions**

In `ui/litellm-dashboard/src/components/networking.tsx`, insert immediately after `claimOnboardingToken` (after the closing `};` currently at line 1527):

```typescript
export const forgotPasswordCall = async (email: string): Promise<{ message: string }> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/forgot_password` : `/user/forgot_password`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(deriveErrorMessage(errorData));
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to submit forgot password request:", error);
    throw error;
  }
};

export const validateResetTokenCall = async (token: string): Promise<{ user_email: string }> => {
  try {
    let url = proxyBaseUrl ? `${proxyBaseUrl}/user/reset_password/validate` : `/user/reset_password/validate`;
    url += `?token=${encodeURIComponent(token)}`;

    const response = await fetch(url, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(deriveErrorMessage(errorData));
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to validate reset token:", error);
    throw error;
  }
};

export const resetPasswordCall = async (token: string, newPassword: string): Promise<{ message: string }> => {
  try {
    const url = proxyBaseUrl ? `${proxyBaseUrl}/user/reset_password` : `/user/reset_password`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, new_password: newPassword }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(deriveErrorMessage(errorData));
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to reset password:", error);
    throw error;
  }
};
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ui/litellm-dashboard && npm run test -- networking.forgotPassword.test.ts`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ui/litellm-dashboard/src/components/networking.tsx ui/litellm-dashboard/src/components/networking.forgotPassword.test.ts
git commit -m "feat(ui): add forgot/reset password networking calls"
```

---

## Task 9: Frontend — hooks

**Files:**
- Create: `ui/litellm-dashboard/src/app/(dashboard)/hooks/passwordReset/usePasswordReset.ts`
- Test: `ui/litellm-dashboard/src/app/(dashboard)/hooks/passwordReset/usePasswordReset.test.ts`

**Interfaces:**
- Consumes: `forgotPasswordCall`, `validateResetTokenCall`, `resetPasswordCall` (Task 8), `createQueryKeys` (`ui/litellm-dashboard/src/app/(dashboard)/hooks/common/queryKeysFactory.ts:25`).
- Produces: `useForgotPassword()` (mutation, `mutate(email: string)`), `useValidateResetToken(token: string | null)` (query, `{ data: { user_email: string } | undefined, isLoading, isError }`), `useResetPassword()` (mutation, `mutate({ token, newPassword })`). Consumed by Tasks 10 and 11.

- [ ] **Step 1: Write the failing test**

Create `ui/litellm-dashboard/src/app/(dashboard)/hooks/passwordReset/usePasswordReset.test.ts`:

```typescript
import { describe, it, expect, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useValidateResetToken } from "./usePasswordReset";
import * as networking from "@/components/networking";

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe("useValidateResetToken", () => {
  it("does not fetch when token is null", () => {
    const spy = vi.spyOn(networking, "validateResetTokenCall");
    renderHook(() => useValidateResetToken(null), { wrapper });
    expect(spy).not.toHaveBeenCalled();
  });

  it("fetches validation data when token is present", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockResolvedValue({ user_email: "alice@example.com" });
    const { result } = renderHook(() => useValidateResetToken("tok-123"), { wrapper });
    await waitFor(() => expect(result.current.data).toEqual({ user_email: "alice@example.com" }));
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui/litellm-dashboard && npm run test -- usePasswordReset.test.ts`
Expected: FAIL — cannot find module `./usePasswordReset`.

- [ ] **Step 3: Write the hooks**

Create `ui/litellm-dashboard/src/app/(dashboard)/hooks/passwordReset/usePasswordReset.ts`:

```typescript
import { useMutation, useQuery } from "@tanstack/react-query";
import { forgotPasswordCall, resetPasswordCall, validateResetTokenCall } from "@/components/networking";
import { createQueryKeys } from "../common/queryKeysFactory";

const passwordResetKeys = createQueryKeys("passwordReset");

export const useForgotPassword = () => {
  return useMutation({
    mutationFn: async (email: string) => forgotPasswordCall(email),
  });
};

export interface ResetTokenValidation {
  user_email: string;
}

export const useValidateResetToken = (token: string | null) => {
  return useQuery<ResetTokenValidation>({
    queryKey: passwordResetKeys.detail(token ?? ""),
    queryFn: async () => {
      if (!token) throw new Error("token is required");
      return validateResetTokenCall(token);
    },
    enabled: Boolean(token),
    retry: false,
  });
};

export interface ResetPasswordParams {
  token: string;
  newPassword: string;
}

export const useResetPassword = () => {
  return useMutation({
    mutationFn: async ({ token, newPassword }: ResetPasswordParams) => resetPasswordCall(token, newPassword),
  });
};
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ui/litellm-dashboard && npm run test -- usePasswordReset.test.ts`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add "ui/litellm-dashboard/src/app/(dashboard)/hooks/passwordReset/"
git commit -m "feat(ui): add forgot/reset password hooks"
```

---

## Task 10: Frontend — `/ui/forgot-password` page

**Files:**
- Create: `ui/litellm-dashboard/src/app/forgot-password/page.tsx`
- Create: `ui/litellm-dashboard/src/app/forgot-password/ForgotPasswordForm.tsx`
- Test: `ui/litellm-dashboard/src/app/forgot-password/ForgotPasswordForm.test.tsx`

**Interfaces:**
- Consumes: `useForgotPassword` (Task 9), `getLoginUrl` (`ui/litellm-dashboard/src/utils/returnUrlUtils.ts`, already used by `OnboardingErrorView.tsx:3`).
- Produces: route `/ui/forgot-password` rendering `ForgotPasswordForm`. Linked to from Task 12 (`LoginPage.tsx`).

- [ ] **Step 1: Write the failing test**

Create `ui/litellm-dashboard/src/app/forgot-password/ForgotPasswordForm.test.tsx`:

```tsx
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ForgotPasswordForm } from "./ForgotPasswordForm";
import * as networking from "@/components/networking";

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe("ForgotPasswordForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("submits the typed email and shows the generic success message", async () => {
    vi.spyOn(networking, "forgotPasswordCall").mockResolvedValue({ message: "sent" });
    const user = userEvent.setup();
    renderWithClient(<ForgotPasswordForm />);

    await user.type(screen.getByLabelText("Email Address"), "alice@example.com");
    await user.click(screen.getByRole("button", { name: /send reset link/i }));

    await waitFor(() => {
      expect(networking.forgotPasswordCall).toHaveBeenCalledWith("alice@example.com");
    });
    await waitFor(() => {
      expect(
        screen.getByText("If an account exists for this email, a password reset link has been sent."),
      ).toBeInTheDocument();
    });
  });

  it("shows a generic error message when the request fails", async () => {
    vi.spyOn(networking, "forgotPasswordCall").mockRejectedValue(new Error("Too many requests"));
    const user = userEvent.setup();
    renderWithClient(<ForgotPasswordForm />);

    await user.type(screen.getByLabelText("Email Address"), "alice@example.com");
    await user.click(screen.getByRole("button", { name: /send reset link/i }));

    await waitFor(() => {
      expect(screen.getByText("Too many requests")).toBeInTheDocument();
    });
  });

  it("has a link back to the login page", () => {
    renderWithClient(<ForgotPasswordForm />);
    expect(screen.getByRole("link", { name: /back to login/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui/litellm-dashboard && npm run test -- ForgotPasswordForm.test.tsx`
Expected: FAIL — cannot find module `./ForgotPasswordForm`.

- [ ] **Step 3: Write the form component**

Create `ui/litellm-dashboard/src/app/forgot-password/ForgotPasswordForm.tsx`:

```tsx
"use client";

import React from "react";
import { Alert, Button, Card, Form, Input, Typography } from "antd";
import { useForgotPassword } from "@/app/(dashboard)/hooks/passwordReset/usePasswordReset";
import { getLoginUrl } from "@/utils/returnUrlUtils";

export function ForgotPasswordForm() {
  const { mutate: submitForgotPassword, isPending, isSuccess, error } = useForgotPassword();

  const handleSubmit = (values: { email: string }) => {
    submitForgotPassword(values.email);
  };

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card>
        <Typography.Title level={5} className="text-center mb-5">
          🚅 LiteLLM
        </Typography.Title>
        <Typography.Title level={3}>Forgot Password</Typography.Title>
        <Typography.Text>Enter your email address and we will send you a link to reset your password.</Typography.Text>

        {isSuccess ? (
          <Alert
            className="mt-4"
            type="success"
            message="If an account exists for this email, a password reset link has been sent."
            showIcon
          />
        ) : (
          <Form className="mt-10 mb-5" layout="vertical" onFinish={handleSubmit}>
            <Form.Item
              label="Email Address"
              name="email"
              rules={[{ required: true, type: "email", message: "Please enter a valid email address" }]}
            >
              <Input type="email" />
            </Form.Item>

            {error && <Alert type="error" message={(error as Error).message} showIcon className="mb-4" />}

            <div className="mt-10">
              <Button htmlType="submit" loading={isPending}>
                Send Reset Link
              </Button>
            </div>
          </Form>
        )}

        <div className="mt-4">
          <Button type="link" href={getLoginUrl()}>
            Back to Login
          </Button>
        </div>
      </Card>
    </div>
  );
}
```

Create `ui/litellm-dashboard/src/app/forgot-password/page.tsx`:

```tsx
"use client";

import React from "react";
import { ForgotPasswordForm } from "./ForgotPasswordForm";

export default function ForgotPassword() {
  return <ForgotPasswordForm />;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ui/litellm-dashboard && npm run test -- ForgotPasswordForm.test.tsx`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ui/litellm-dashboard/src/app/forgot-password/
git commit -m "feat(ui): add /ui/forgot-password page"
```

---

## Task 11: Frontend — `/ui/reset-password` page

**Files:**
- Create: `ui/litellm-dashboard/src/app/reset-password/page.tsx`
- Create: `ui/litellm-dashboard/src/app/reset-password/ResetPasswordForm.tsx`
- Test: `ui/litellm-dashboard/src/app/reset-password/ResetPasswordForm.test.tsx`

**Interfaces:**
- Consumes: `useValidateResetToken`, `useResetPassword` (Task 9), `useZodForm` (`ui/litellm-dashboard/src/lib/forms/useZodForm.ts`), `FormField`/`FieldGroup` (`ui/litellm-dashboard/src/components/shared/form/`), `getLoginUrl`. Built with shadcn/ui primitives (`Button`, `Card`, `Input`), not antd — antd new imports are blocked by this project's `no-restricted-imports` ESLint rule (`ui/litellm-dashboard/eslint.config.mjs`).
- Produces: route `/ui/reset-password?token=...` rendering `ResetPasswordForm`.

- [ ] **Step 1: Write the failing test**

Create `ui/litellm-dashboard/src/app/reset-password/ResetPasswordForm.test.tsx`:

```tsx
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ResetPasswordForm } from "./ResetPasswordForm";
import * as networking from "@/components/networking";

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe("ResetPasswordForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows an invalid-link message when there is no token", () => {
    renderWithClient(<ResetPasswordForm token={null} />);
    expect(screen.getByText("This link is invalid or has expired.")).toBeInTheDocument();
  });

  it("shows an invalid-link message when validation fails", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockRejectedValue(new Error("invalid"));
    renderWithClient(<ResetPasswordForm token="bad-token" />);
    await waitFor(() => {
      expect(screen.getByText("This link is invalid or has expired.")).toBeInTheDocument();
    });
  });

  it("shows the target email and submits matching passwords", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockResolvedValue({ user_email: "alice@example.com" });
    vi.spyOn(networking, "resetPasswordCall").mockResolvedValue({ message: "ok" });
    const user = userEvent.setup();
    renderWithClient(<ResetPasswordForm token="good-token" />);

    await waitFor(() => {
      expect(screen.getByText(/alice@example.com/)).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText("New Password"), "correct horse battery staple");
    await user.type(screen.getByLabelText("Confirm New Password"), "correct horse battery staple");
    await user.click(screen.getByRole("button", { name: /reset password/i }));

    await waitFor(() => {
      expect(networking.resetPasswordCall).toHaveBeenCalledWith("good-token", "correct horse battery staple");
    });
    await waitFor(() => {
      expect(screen.getByText("Password reset successfully.")).toBeInTheDocument();
    });
  });

  it("shows a validation error when the two password fields do not match", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockResolvedValue({ user_email: "alice@example.com" });
    const user = userEvent.setup();
    renderWithClient(<ResetPasswordForm token="good-token" />);

    await waitFor(() => screen.getByLabelText("New Password"));
    await user.type(screen.getByLabelText("New Password"), "password-one");
    await user.type(screen.getByLabelText("Confirm New Password"), "password-two");
    await user.click(screen.getByRole("button", { name: /reset password/i }));

    await waitFor(() => {
      expect(screen.getByText("Passwords do not match")).toBeInTheDocument();
    });
    expect(networking.resetPasswordCall).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui/litellm-dashboard && npm run test -- ResetPasswordForm.test.tsx`
Expected: FAIL — cannot find module `./ResetPasswordForm`.

- [ ] **Step 3: Write the form component**

Create `ui/litellm-dashboard/src/app/reset-password/ResetPasswordForm.tsx`:

```tsx
"use client";

import React from "react";
import { z } from "zod";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useValidateResetToken, useResetPassword } from "@/app/(dashboard)/hooks/passwordReset/usePasswordReset";
import { useZodForm } from "@/lib/forms/useZodForm";
import { FormField } from "@/components/shared/form/FormField";
import { FieldGroup } from "@/components/shared/form/field";
import { getLoginUrl } from "@/utils/returnUrlUtils";

const resetPasswordSchema = z
  .object({
    password: z.string().min(1, "Please enter a new password"),
    confirm_password: z.string().min(1, "Please confirm your new password"),
  })
  .refine((data) => data.password === data.confirm_password, {
    path: ["confirm_password"],
    message: "Passwords do not match",
  });

type ResetPasswordFormProps = {
  token: string | null;
};

export function ResetPasswordForm({ token }: ResetPasswordFormProps) {
  const { data: validationData, isLoading: isValidating, isError: isValidationError } = useValidateResetToken(token);
  const { mutate: submitResetPassword, isPending, isSuccess, error: resetError } = useResetPassword();
  const form = useZodForm(resetPasswordSchema, { defaultValues: { password: "", confirm_password: "" } });

  if (!token || isValidationError) {
    return (
      <div className="mx-auto w-full max-w-md mt-10">
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-red-700">
          This link is invalid or has expired.
        </div>
        <div className="mt-4">
          <a href="/ui/forgot-password">Request a new link</a>
        </div>
      </div>
    );
  }

  if (isValidating) {
    return (
      <div className="mx-auto w-full max-w-md mt-10 flex justify-center">
        <Loader2 className="h-6 w-6 animate-spin" />
      </div>
    );
  }

  if (isSuccess) {
    return (
      <div className="mx-auto w-full max-w-md mt-10">
        <div className="rounded-md border border-green-200 bg-green-50 px-4 py-2 text-green-700">
          Password reset successfully.
        </div>
        <div className="mt-4">
          <a href={getLoginUrl()}>Back to Login</a>
        </div>
      </div>
    );
  }

  const onSubmit = form.handleSubmit((values) => {
    if (!token) return;
    submitResetPassword({ token, newPassword: values.password });
  });

  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <Card className="p-6">
        <h1 className="text-center text-lg font-semibold mb-5">🚅 LiteLLM</h1>
        <h2 className="text-xl font-semibold">Reset Password</h2>
        <p className="text-sm text-muted-foreground">Resetting password for {validationData?.user_email}</p>

        <form className="mt-10 mb-5 space-y-4" onSubmit={onSubmit}>
          <FieldGroup>
            <FormField control={form.control} name="password" label="New Password">
              {({ ref, ...field }) => <Input {...field} ref={ref} type="password" />}
            </FormField>
            <FormField control={form.control} name="confirm_password" label="Confirm New Password">
              {({ ref, ...field }) => <Input {...field} ref={ref} type="password" />}
            </FormField>
          </FieldGroup>

          {resetError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2 text-red-700">
              {(resetError as Error).message}
            </div>
          )}

          <Button type="submit" disabled={isPending}>
            {isPending && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
            Reset Password
          </Button>
        </form>
      </Card>
    </div>
  );
}
```

Create `ui/litellm-dashboard/src/app/reset-password/page.tsx`:

```tsx
"use client";

import React, { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { ResetPasswordForm } from "./ResetPasswordForm";

function ResetPasswordContent() {
  const searchParams = useSearchParams()!;
  const token = searchParams.get("token");
  return <ResetPasswordForm token={token} />;
}

export default function ResetPassword() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <ResetPasswordContent />
    </Suspense>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ui/litellm-dashboard && npm run test -- ResetPasswordForm.test.tsx`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ui/litellm-dashboard/src/app/reset-password/
git commit -m "feat(ui): add /ui/reset-password page"
```

---

## Task 12: Frontend — "Forgot password?" link on the login page

**Files:**
- Modify: `ui/litellm-dashboard/src/app/login/LoginPage.tsx:251-264` (password `Form.Item` block)
- Modify: `ui/litellm-dashboard/src/app/login/LoginPage.test.tsx` (append test)

**Interfaces:**
- Consumes: nothing new (plain `<a>` link to `/ui/forgot-password`, route created in Task 10).

- [ ] **Step 1: Write the failing test**

Append to `ui/litellm-dashboard/src/app/login/LoginPage.test.tsx` (inside the existing top-level `describe` block, following that file's existing mocking setup):

```typescript
it("renders a 'Forgot password?' link pointing to /ui/forgot-password", async () => {
  render(<LoginPage />);
  await waitFor(() => {
    const link = screen.getByRole("link", { name: /forgot password/i });
    expect(link).toHaveAttribute("href", "/ui/forgot-password");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd ui/litellm-dashboard && npm run test -- LoginPage.test.tsx -t "Forgot password"`
Expected: FAIL — `Unable to find role="link" with name /forgot password/i`.

- [ ] **Step 3: Add the link**

In `ui/litellm-dashboard/src/app/login/LoginPage.tsx`, insert right after the password `Form.Item` closes (after line 264, before the submit-button `Form.Item` at line 266):

```tsx
            <div className="text-right mb-4">
              <a href="/ui/forgot-password">Forgot password?</a>
            </div>

```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd ui/litellm-dashboard && npm run test -- LoginPage.test.tsx`
Expected: PASS (all `LoginPage.test.tsx` tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add ui/litellm-dashboard/src/app/login/LoginPage.tsx ui/litellm-dashboard/src/app/login/LoginPage.test.tsx
git commit -m "feat(ui): link to forgot-password flow from the login page"
```

---

## Task 13: Frontend full suite + build verification

**Files:** none new — verification only.

- [ ] **Step 1: Run the full frontend test suite**

```bash
cd ui/litellm-dashboard && npm run test
```

Expected: all tests pass, including every test file touched in Tasks 8-12.

- [ ] **Step 2: Run the production build**

```bash
cd ui/litellm-dashboard && npm run build
```

Expected: build succeeds with no type errors (the two new routes `forgot-password` and `reset-password` appear in the build output route list).

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix(ui): resolve build/test issues in password reset flow"
```

(Skip if `git status` shows nothing to commit.)

---

## Task 14: Manual QA against a live proxy (proof of fix)

**Files:** none — this task produces the "Screenshots / Proof of Fix" section content for the PR, per this repo's convention of curling a live instance rather than showing pytest output.

- [ ] **Step 1: Start a local proxy with SMTP configured against a real test inbox**

Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_SENDER_EMAIL` in `.env` (e.g. pointing at a Mailtrap/personal test inbox), then:

```bash
python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
```

- [ ] **Step 2: Create a test internal user with a password via the existing `/user/update` endpoint**

```bash
curl -s --location 'http://0.0.0.0:4000/user/new' \
  --header 'Authorization: Bearer sk-1234' \
  --header 'Content-Type: application/json' \
  --data '{"user_email": "qa-test@example.com", "password": "OldPassword123!", "user_role": "internal_user"}'
```

- [ ] **Step 3: Trigger the forgot-password flow via curl and confirm the email arrives**

```bash
curl -s --location 'http://0.0.0.0:4000/user/forgot_password' \
  --header 'Content-Type: application/json' \
  --data '{"email": "qa-test@example.com"}'
```

Expected: `{"message": "If an account exists for this email, a password reset link has been sent."}`, and the reset email arrives in the test inbox with a `/ui/reset-password?token=...` link.

- [ ] **Step 4: Confirm anti-enumeration with an unknown email**

```bash
curl -s --location 'http://0.0.0.0:4000/user/forgot_password' \
  --header 'Content-Type: application/json' \
  --data '{"email": "does-not-exist@example.com"}'
```

Expected: identical `200` response body to Step 3, no email sent.

- [ ] **Step 5: Walk the UI reset flow**

Go to `http://localhost:3000/ui/login` (UI dev server via `npm run dev` in `ui/litellm-dashboard`), click "Forgot password?", submit `qa-test@example.com`, confirm the generic success message appears. Open the emailed link, confirm it lands on `/ui/reset-password?token=...` showing "Resetting password for qa-test@example.com", set a new password, submit, confirm redirect/prompt back to `/ui/login`, then log in with the new password.

- [ ] **Step 6: Confirm single-use enforcement**

Re-open the same emailed link after Step 5's successful reset and attempt to submit a password again.

Expected: "This link is invalid or has expired."

---

## Self-Review Notes

- **Spec coverage:** every section of `docs/superpowers/specs/2026-07-28-forgot-password-design.md` maps to a task — data model → Task 1/2, `/user/forgot_password` → Task 4, `/user/reset_password/validate` → Task 5, `/user/reset_password` → Task 6, rate limiting/anti-enumeration → Task 4 tests, frontend pages → Tasks 10/11, login link → Task 12, testing conventions → every task's Step 1/2/4, manual QA → Task 14.
- **Type consistency checked:** `PasswordResetTokenRepository.find_valid_by_hash`/`invalidate_unused_for_user` signatures (Task 2) match every call site in Tasks 4-6. `ForgotPasswordRequest`/`ResetPasswordRequest` (Task 3) field names (`email`, `token`, `new_password`) match the endpoint bodies (Tasks 4/6) and the frontend `networking.tsx` payload shapes (Task 8). Hook return shapes (`useValidateResetToken` → `{ user_email }`, Task 9) match what `ResetPasswordForm` destructures (Task 11).
- **No placeholders:** every step has concrete, complete code — no "TBD"/"add error handling" left for the implementer to invent.
