# Forgot password (self-service) — design spec

Date: 2026-07-28
Status: approved by user, pending implementation plan

## Context

LiteLLM proxy internal (non-SSO) users currently have no way to recover access if they
forget their password. The only existing mechanism is admin-initiated: an admin opens
the user's detail page, clicks "Reset Password", which generates a `LiteLLM_InvitationLink`
row and a shareable one-time onboarding link the admin has to copy/paste and send manually
(`POST /invitation/new` → `/ui/onboarding?action=reset_password`). There is no
unauthenticated, user-initiated "forgot password" entry point anywhere (no link on the
login page, no endpoint, no email flow).

This is the first of three planned, independently-shippable PRs to improve internal user
password management:

1. **Forgot password (self-service)** — this spec.
2. **Password policy** (complexity, min length, expiration/lifetime) configurable by an
   admin via `general_settings.password_policy` (config.yaml default, DB override through
   the existing `/config/update` / `/config/field/update` + `LiteLLM_Config` mechanism).
   Includes login-time enforcement (expired/non-compliant password → blocked login,
   redirected into the forgot-password flow built here).
3. **Change my password** — self-service, for an already-authenticated user, with current-
   password verification (today's `/user/update` lets a logged-in user overwrite their own
   password with no verification at all — a gap worth closing but out of scope here).

This sequencing was chosen deliberately: policy's login-time enforcement needs a place to
redirect the user to (this PR), so forgot-password had to come first.

## Scope of this PR

An internal (non-SSO) user who forgot their password can, from the login screen, request
a reset link sent to their email, with no admin involvement. Out of scope: password
complexity rules (PR2), authenticated change-password flow (PR3), any email-sending
infra changes beyond calling the existing `send_email()` utility.

## Data model

New table, independent of `LiteLLM_InvitationLink` (that table stays as-is, scoped to
admin-initiated invites/resets):

```prisma
model LiteLLM_PasswordResetToken {
  token_hash   String    @id
  user_id      String
  requested_ip String?
  created_at   DateTime  @default(now())
  expires_at   DateTime
  used_at      DateTime?
  user         LiteLLM_UserTable @relation(fields: [user_id], references: [user_id], onDelete: Cascade)
}
```

- `token_hash`: SHA-256 hash of a `secrets.token_urlsafe(32)` raw token. The raw token only
  ever appears in the emailed URL; it is never persisted in plaintext, mirroring the
  existing `hash_token()` treatment of virtual keys.
- `expires_at`: fixed 30 minutes from `created_at`. Not admin-configurable in this PR
  (YAGNI — can be exposed later if a real need shows up).
- Every new `/user/forgot_password` request invalidates (marks used or deletes) any
  outstanding unused token rows for that `user_id` before creating a new one.

Rationale for a dedicated table over reusing `LiteLLM_InvitationLink`: the invite table's
semantics (7-day expiry, admin-initiated, `is_accepted` boolean) don't map cleanly onto an
unauthenticated, self-service, short-lived, rate-limited flow. Overloading it would
conflate two different security postures.

## Backend endpoints and flow

Three new endpoints. The first two are unauthenticated by necessity (the user has no
session).

### `POST /user/forgot_password`

Body: `{ "email": str }`

1. Check rate limit (see Security below). If exceeded, return `429` with a generic message.
2. Look up `LiteLLM_UserTable` by `user_email`, case-insensitive (same lookup logic as
   `authenticate_user()` in `login_utils.py`).
3. If the user exists **and** has a non-null `password` (i.e. an internal account, not
   SSO-only): invalidate previous unused tokens for that user, create a new
   `LiteLLM_PasswordResetToken`, send an email via the existing
   `litellm/proxy/utils.py:send_email()` with a link to
   `{base_url}/ui/reset-password?token={raw_token}`.
4. If the user doesn't exist, is SSO-only, or SMTP isn't configured
   (`SMTP_HOST`/`SMTP_SENDER_EMAIL` env vars missing): nothing is sent, but a `logger.warning`
   is emitted server-side so an admin can tell the feature is misconfigured or that an
   unknown/SSO email was targeted.
5. **In every case**, respond `200` with the same generic message ("if an account exists,
   an email has been sent"). No observable difference in status code, body, or timing
   across these branches.

### `GET /user/reset_password/validate?token=...`

1. Hash the incoming token, look up the row, check `expires_at > now()` and
   `used_at IS NULL`.
2. If valid: return `{ user_email }` for display purposes ("Resetting password for
   x@y.com").
3. If invalid/expired/already used: `400` with a generic "This link is invalid or has
   expired" message.

### `POST /user/reset_password`

Body: `{ "token": str, "new_password": str }`

1. Re-validate the token as above, inside a transaction to prevent a concurrent double-claim.
2. Hash the new password with the existing `hash_password()` and update
   `LiteLLM_UserTable.password`.
3. Mark the token `used_at = now()`, invalidate any other pending token for that user.
4. Respond `200`. **No session is issued** — the user is redirected to `/ui/login` to log
   in manually with the new password.
5. If the token is already used/expired by the time this runs (race, double-submit):
   generic `400`, no detail leaked about the exact state.

## Security

- **Anti-enumeration**: identical response (`200`, same body) regardless of whether the
  email exists, is SSO-only, has no password set, or is entirely unknown.
- **Rate limiting**: reuse the existing `DualCache`/`RedisCache.increment_cache` primitive
  already instantiated proxy-wide (`redis_usage_cache`), with graceful in-memory fallback
  when Redis isn't configured (the same degradation pattern already used elsewhere in the
  proxy, e.g. PKCE / control-plane login codes).
  - Per email: max 3 requests / hour.
  - Per IP: max 10 requests / hour (limits spraying across many different target emails).
- **Single-use tokens**, previous tokens invalidated on every new request, SHA-256 hash
  stored (never the raw token).
- No sensitive data (target email, account existence) is ever returned to the client
  beyond the generic message; anything diagnostic goes to server-side warning logs only.

## Frontend

- **New "Forgot password?" link** on `LoginPage.tsx`, under the password field.
- **New page `/ui/forgot-password`**: simple email form → `forgotPasswordCall(email)` →
  always shows the same generic success message, regardless of whether an account exists.
- **New page `/ui/reset-password?token=...`**: on mount, validates the token via
  `GET /user/reset_password/validate` (reusing the loading/error visual pattern from
  `OnboardingLoadingView`/`OnboardingErrorView`). If valid, shows the target email plus a
  new-password form (password + confirmation field, client-side match check only — no
  complexity rule yet, that's PR2). On submit, calls `resetPasswordCall(token, new_password)`
  and redirects to `/ui/login?reset=success` with a confirmation banner.
- **Deliberately separate from the existing `/ui/onboarding` flow** (which stays scoped to
  admin-initiated invites/resets backed by `LiteLLM_InvitationLink`/JWT) — new dedicated
  pages avoid mixing the two token semantics.
- **networking.tsx**: three new functions, `forgotPasswordCall`, `validateResetTokenCall`,
  `resetPasswordCall`, patterned after `loginCall`/`getOnboardingCredentials`/
  `claimOnboardingToken`.

## Testing

- **Backend** (`tests/test_litellm/proxy/management_endpoints/test_password_reset_endpoints.py`,
  mocked DB, no real network calls): identical response for existing/non-existing/SSO-only
  email (anti-enumeration regression test), rate limit triggers after N requests, token is
  single-use (second claim fails), expired token rejected, a new request invalidates prior
  tokens, SMTP not configured still returns a generic `200` while logging a warning,
  concurrent double-claim race handled via transaction.
- **Frontend** (vitest + RTL, following `OnboardingForm.test.tsx` conventions): forgot-password
  form submission shows the generic message, reset-password page loading/error/success
  states, "Forgot password?" link present and navigable from `LoginPage`.
- Proof of fix per project convention: real local proxy + `curl` commands (not pytest output
  as "proof"), with a real test SMTP server where possible, or server log capture when SMTP
  is intentionally left unconfigured.

## Out of scope / deferred to later PRs

- Password complexity/length/expiration rules (PR2).
- Login-time enforcement of expired/non-compliant passwords redirecting into this flow
  (PR2, once this flow exists to redirect into).
- Authenticated "change my password" self-service with current-password verification (PR3).
- Making the reset-token TTL or rate-limit thresholds admin-configurable.
