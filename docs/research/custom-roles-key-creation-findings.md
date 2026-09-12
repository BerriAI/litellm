# Findings: a role with platform access but no API key creation

Status: research only, no behavior change. Branch `litellm_custom-roles-key-creation-research`

## Question

A customer wants a role that keeps useful platform access (call models, see their own keys and spend, read team/model info) but explicitly cannot mint new API keys. They are building a control layer in front of the proxy on the assumption that this is not possible natively. This doc checks that assumption against current `main` (commit `362033cb2a`), maps every role and key related permission, records live reproduction results, and scopes what is still missing

## Short answer

It mostly exists today, but it is undiscoverable because it is spread across three separate mechanisms. There are two built in roles that already meet the ask exactly, `internal_user_viewer` and `proxy_admin_viewer`, at the cost of also losing key regenerate, key update and key delete. For a user who needs to keep editing keys or managing team members while still being blocked from minting new keys, `litellm_settings.key_generation_settings` gets within one gap: `/key/{key}/regenerate` is not gated by that setting, so a user who already owns a key can rotate it into a new secret. Closing that gap is a small change, scoped at the end of this doc

## Where the checks live

Role to route authorization is `RouteChecks.non_proxy_admin_allowed_routes_check` in `litellm/proxy/auth/route_checks.py`. Route bundles are the `LiteLLMRoutes` enum in `litellm/proxy/_types.py`. Key specific policy lives in `litellm/proxy/management_endpoints/key_management_endpoints.py` (`key_generation_check`, `_team_key_generation_check`, `_personal_key_generation_check`) and `litellm/proxy/management_helpers/team_member_permission_checks.py` (`TeamMemberPermissionChecks`). The Admin UI mirrors roles in `ui/litellm-dashboard/src/utils/roles.ts`

The check order for a non `proxy_admin` request to `/key/generate` is: role route check (view only roles are denied here), then `key_generation_check`, which for a team key runs `_team_key_operation_team_member_check` (`allowed_team_member_roles`, default `["admin", "user"]`) followed by `TeamMemberPermissionChecks` (team admins pass, plain members need `/key/generate` in `team_member_permissions`), and for a personal key runs `_personal_key_membership_check` (`allowed_user_roles`, unrestricted when unset)

## Role map

`LitellmUserRoles` in `litellm/proxy/_types.py` defines seven roles. `team` is a JWT scope and `customer` is an end user identity, neither is a dashboard login role, so they are omitted below. "Own key" means a key where `user_id` matches the caller

| Capability | proxy_admin | proxy_admin_viewer | org_admin (membership) | internal_user | internal_user_viewer |
| --- | --- | --- | --- | --- | --- |
| Call models (`/chat/completions` etc.) | yes | yes | yes | yes | yes |
| `/key/generate` personal | yes | no | yes | yes | no |
| `/key/generate` team | yes | no | team admin or `team_member_permissions` | team admin or `team_member_permissions` | no |
| `/key/service-account/generate` | yes | no | team admin only | team admin only | no |
| `/key/{key}/regenerate` own key | yes | no | yes | yes | no |
| `/key/update`, `/key/delete` own key | yes | no | yes | yes | no |
| `/key/info`, `/key/list` own | yes | yes (all) | yes | yes | yes |
| Spend and activity read | yes (all) | yes (all) | yes | yes (own) | yes (own) |
| `/user/new`, `/team/new` | yes | no | yes (inside org) | no | no |
| `/model/new` etc. | yes | no | via `self_managed_routes` | via `self_managed_routes` (team admin) | no |

`org_admin` is not assignable through `/user/new` `user_role`. It is a membership role set with `/organization/member_add` on an `internal_user`, and `_user_is_org_admin` resolves it from the organization memberships at request time

The concrete route lists backing this table: `internal_user_routes` includes `key_management_routes` in full, `internal_user_view_only_routes` is `spend_tracking_routes + compliance_check_routes + tag routes` and contains no key write routes, and `_PROXY_ADMIN_VIEW_ONLY_BLOCKED_ROUTES` explicitly lists `KEY_GENERATE`, `KEY_UPDATE`, `KEY_DELETE`, `KEY_REGENERATE`, `KEY_GENERATE_SERVICE_ACCOUNT`, `KEY_BLOCK`, `KEY_UNBLOCK` and the bulk update routes

The UI treats `proxy_admin_viewer` as an `"Admin"` session with a view only flag and puts `"Internal Viewer"` and `"Admin Viewer"` in `viewOnlyRoles`, so the create key button is hidden for both viewer roles. `rolesWithWriteAccess` is `["Internal User", "Admin", "proxy_admin"]`

## Configuration knobs that restrict key creation

`litellm_settings.key_generation_settings` (documented in `docs/proxy/virtual_keys.md` in the docs repo) has two halves. `personal_key_generation.allowed_user_roles` gates personal keys by `user_role`. `team_key_generation.allowed_team_member_roles` gates team keys by team membership role and may be an empty list, which denies team admins too. Both are checked in `key_generation_check`, which is called by `/key/generate` and `/key/service-account/generate` only

`team_member_permissions` on a team (`/team/permissions_update`) grants plain team members extra `KeyManagementRoutes`. Baseline is always `KEY_INFO` and `KEY_HEALTH`. Adding `/key/generate` lets a plain member create team keys, but it does not override `allowed_team_member_roles`, and it does not imply `/key/service-account/generate`

`general_settings.admin_only_routes` (enterprise, needs a license) denies listed routes to all non admin roles. It matches on the literal route string, so `"/key/regenerate"` does not match the real path `/key/{key}/regenerate`. Listing `"/key/{key}/regenerate"` was not tested and is worth a follow up

## Live reproduction

All scenarios were run with curl against a local proxy on port 4000 backed by PostgreSQL, hitting a real Anthropic model for the `/chat/completions` checks. Scenario A is default config. Scenario B adds `personal_key_generation.allowed_user_roles: ["proxy_admin"]` and `team_key_generation.allowed_team_member_roles: ["admin"]`. Scenario C is B plus `team_member_permissions: ["/key/generate"]` on the team. Scenario D is default config plus that team permission. Scenario E is a licensed proxy with `admin_only_routes: ["/key/generate", "/key/service-account/generate", "/key/regenerate"]`. Scenario F sets `allowed_user_roles: ["proxy_admin"]` and `allowed_team_member_roles: []`

| Scenario | Caller | personal generate | team generate | service account | own regenerate | chat, key info, spend |
| --- | --- | --- | --- | --- | --- | --- |
| A | internal_user, team member | 200 | 401 | 401 | 200 | 200 |
| A | internal_user, team admin | 200 | 200 | 200 | 200 | 200 |
| A | internal_user_viewer | 401 | 401 | 401 | 401 | 200 |
| A | proxy_admin_viewer | 401 | 401 | 401 | 401 | 200 |
| A | org_admin membership | 200 | 200 (own org team) | 200 | 200 | 200 |
| B | internal_user, team member | 400 | 400 | 400 | 200 | 200 |
| B | internal_user, team admin | 400 | 200 | 200 | 200 | 200 |
| B | org_admin membership | 400 | 400 (not a team member) | 400 | 200 | 200 |
| C | internal_user, team member | 400 | 400 | 400 | 200 | 200 |
| D | internal_user, team member | 200 | 200 | 401 | 200 | 200 |
| E | internal_user, team member or admin | 401 | 401 | 401 | 200 | 200 |
| F | internal_user, team member or admin | 400 | 400 | 400 | 200 | 200 |
| F | proxy_admin | 200 | 200 | 200 | 200 | 200 |

Team admins in scenario F could still `/team/member_add` and list their team. The viewer roles behaved the same in every scenario because their role route check runs before `key_generation_check`

The `400` and `401` distinction is meaningful: `401` comes from the role route check or `TeamMemberPermissionChecks`, `400` comes from `key_generation_check`

## Existing way to do this

If the user does not need to edit, rotate or delete keys, assign `internal_user_viewer` (own scope) or `proxy_admin_viewer` (proxy wide read). Both call models with a key issued by an admin, see their own keys and spend, and are denied every key write route. No config needed. This is the exact ask and it is already covered by `tests/proxy_admin_ui_tests/test_role_based_access.py`

If the user must keep `internal_user` (for example a team admin who manages members) but should not mint keys, use

```yaml
litellm_settings:
  key_generation_settings:
    personal_key_generation:
      allowed_user_roles: ["proxy_admin"]
    team_key_generation:
      allowed_team_member_roles: []
```

with the caveat that `/key/{key}/regenerate` on a key they already own still returns a fresh secret. Whether that matters depends on the threat model: regenerate does not change ownership, models, budget or team, it only rotates the secret, so the number of live credentials does not grow. If "cannot create new API keys" means "cannot increase the set of credentials", the config above satisfies it today. If it means "cannot obtain any new secret value", it does not

## Scoped proposal to close the gap

The minimal change is to make regenerate honor `key_generation_settings`. In `key_management_endpoints.py`, `regenerate_key_fn` should resolve the existing key's `team_id` and call `key_generation_check(team_table=..., user_api_key_dict=..., data=GenerateKeyRequest(team_id=existing.team_id, user_id=existing.user_id), route=KeyManagementRoutes.KEY_REGENERATE)` before rotating, skipping for `proxy_admin` as today. `_team_key_operation_team_member_check` already takes `route`, so the error messages stay accurate. No enum or type changes. Roughly 20 lines plus tests, one file, plus a sentence in `virtual_keys.md`

A more discoverable option is a dedicated flag rather than the two list config. Add `disallow_key_creation: bool` to `StandardKeyGenerationConfig` (a `TypedDict` in `litellm/types/utils.py`, follow `ReadOnly[...]`), and have `key_generation_check` short circuit to a 400 for every non `proxy_admin` caller when set, covering `/key/generate`, `/key/service-account/generate` and `/key/{key}/regenerate`. Still no new `LitellmUserRoles` member, since the ask is a restriction on top of existing roles, not a new access bundle. Roughly 40 lines plus tests and docs

A per user granular permission (a real "no key creation" role) would need a new `LitellmUserRoles` value, a route bundle in `LiteLLMRoutes`, a branch in `non_proxy_admin_allowed_routes_check`, UI role tables in `roles.ts` plus the user create and edit forms, and `/user/new` validation. That is several hundred lines across Python and TypeScript and does not buy anything over the flag for this ask, so it is not recommended

## Related GitHub issues

Closed: #8211 "Internal User[view only] role - able to create key" (the viewer fix that gives us `internal_user_viewer` today), #18521 "Separation of UI and virtual key permissions", #5300 "Permissions for a team admin", #425 and #1793 on user roles generally. Open and adjacent but not the same ask: #20962, #33212, #33194 and #27005 are about non admin users being unable to create or configure keys, the opposite direction, and #32451 is about hiding usage views from internal users. No open issue asks for a "can use the platform but cannot create keys" role, so the customer's ask is not tracked anywhere yet

## Tests added

`tests/test_litellm/proxy/management_endpoints/test_key_management_endpoints.py` gained four unit tests that pin the current `key_generation_check` behavior used in the recommendation: `allowed_user_roles` blocks `internal_user` personal keys and passes `proxy_admin`, an empty `allowed_team_member_roles` blocks team admins, a `team_member_permissions` grant does not override `allowed_team_member_roles`, and default `team_member_permissions` deny `/key/generate` to plain members while a grant or admin role allows it. The regenerate gap is documented here rather than pinned by a test, since the intent is to change it
