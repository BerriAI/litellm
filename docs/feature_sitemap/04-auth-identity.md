# 04 Auth and identity

Who can call the gateway and what they can touch: the master key, virtual keys and their scopes, users, teams, organizations, projects, end users, RBAC, access groups, SSO and SAML, JWT, SCIM, and the UI access controls. Budgets and rate limits attached to these identities live in [05-budgets-ratelimits-spend.md](05-budgets-ratelimits-spend.md); the Admin UI pages that expose them are cross-referenced where relevant

## Keys

### auth.master_key: Master key and rotation
surfaces: config, api | flags: none
docs: https://docs.litellm.ai/docs/proxy/virtual_keys, https://docs.litellm.ai/docs/proxy/master_key_rotations
code: `litellm/proxy/auth/user_api_key_auth.py` (master key check), `litellm/proxy/management_endpoints/key_management_endpoints.py` (POST `/key/regenerate`)
tests: `tests/test_litellm/proxy/auth/`, `tests/e2e/other/` (master key gate)
registry: other.yaml other.auth.master_key.*
verify: rotate with POST /key/regenerate on the master key, then confirm the old master key is rejected and the new one works on /key/list

### auth.virtual_keys: Virtual keys (/key/generate, update, delete, info, list, block, regenerate)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/virtual_keys, https://docs.litellm.ai/docs/proxy/user_keys
code: `litellm/proxy/management_endpoints/key_management_endpoints.py` (`/key/generate`, `/key/update`, `/key/delete`, `/key/info`, `/key/list`, `/key/block`, `/key/regenerate`), `litellm/proxy/auth/user_api_key_auth.py`
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/management/`
registry: mgmt.yaml mgmt.key.*, mgmt.virtual_key.*
verify: curl -X POST http://localhost:4000/key/generate -H "Authorization: Bearer sk-1234" -d '{}' then call /chat/completions with the returned key

### auth.key_scopes: Key permissions: models, allowed_routes, llm-only keys, metadata, expiry
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/virtual_keys, https://docs.litellm.ai/docs/proxy/access_control
code: `litellm/proxy/management_endpoints/key_management_endpoints.py`, `litellm/proxy/auth/user_api_key_auth.py` (`allowed_routes`, model allowlist checks), `litellm/proxy/auth/auth_checks.py`
tests: `tests/test_litellm/proxy/auth/`, `tests/e2e/access_control/`
registry: other.yaml other.auth.*, mgmt.yaml mgmt.key.*
verify: generate a key with `allowed_routes: ["/chat/completions"]` and confirm a /key/info call with it is denied while chat works

### auth.service_accounts: Service account keys
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/service_accounts
code: `litellm/proxy/management_endpoints/key_management_endpoints.py` (POST `/key/service-account/generate`)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.key.*
verify: POST /key/service-account/generate and confirm the key works on LLM routes but cannot create users

## People and groups

### auth.users: Internal users (/user/*)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/users, https://docs.litellm.ai/docs/proxy/ui/bulk_edit_users
code: `litellm/proxy/management_endpoints/internal_user_endpoints.py` (`/user/new`, `/user/update`, `/user/delete`, `/user/list`, `/user/info`, `/user/available_roles`, `/user/bulk_update`)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/management/`
registry: mgmt.yaml mgmt.user.*
verify: POST /user/new {"user_email":"t@x.com"} then confirm it appears in GET /user/list and on http://localhost:4000/ui/?page=users

### auth.teams: Teams (/team/*)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/virtual_keys, https://docs.litellm.ai/docs/proxy/team_budgets
code: `litellm/proxy/management_endpoints/team_endpoints.py` (`/team/new`, `/team/update`, `/team/member_add`, `/team/member_delete`, `/team/delete`, `/team/info`, `/team/block`, `/team/permissions_update`), `litellm/proxy/management_endpoints/team_admin_field_permissions.py`
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/management/`
registry: mgmt.yaml mgmt.team.*
verify: POST /team/new {"team_alias":"t"}, add a member with /team/member_add, and check membership in GET /team/info

### auth.organizations: Organizations (/organization/*)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/user_management_heirarchy
code: `litellm/proxy/management_endpoints/organization_endpoints.py` (`/organization/new`, `/organization/update`, `/organization/delete`, `/organization/member_add`, `/organization/list`, `/organization/info`)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/management/`
registry: mgmt.yaml mgmt.organization.*
verify: POST /organization/new, attach a team to it, and confirm org-scoped visibility on http://localhost:4000/ui/?page=organizations

### auth.projects: Projects
surfaces: api, ui | flags: db, beta
docs: https://docs.litellm.ai/docs/proxy/project_management, https://docs.litellm.ai/docs/proxy/ui_project_management
code: `litellm/proxy/management_endpoints/team_endpoints.py` (project fields on teams), `ui/litellm-dashboard/src/app/(dashboard)/projects/` if present
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: none
verify: open http://localhost:4000/ui/?page=projects, create a project under a team, and confirm keys can be scoped to it

### auth.end_users: End users / customers (/customer/*, /end_user/*)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/customers, https://docs.litellm.ai/docs/proxy/customer_routing, https://docs.litellm.ai/docs/proxy/customer_usage
code: `litellm/proxy/management_endpoints/customer_endpoints.py` (`/customer/new`, `/customer/block`, `/customer/info`, `/end_user/daily/activity`)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/management/`
registry: mgmt.yaml mgmt.customer.*, mgmt.end_user.*
verify: POST /customer/new {"user_id":"cust1"}, send a chat completion with `user: "cust1"`, then GET /end_user/daily/activity to see attribution

## Access control and federation

### auth.rbac: Role-based access control (proxy admin, org admin, team admin, internal user, viewer)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/access_control, https://docs.litellm.ai/docs/proxy/user_management_heirarchy
code: `litellm/proxy/auth/user_api_key_auth.py`, `litellm/proxy/management_endpoints/internal_user_endpoints.py` (GET `/user/available_roles`), `litellm/proxy/_types.py` (`LitellmUserRoles`)
tests: `tests/test_litellm/proxy/auth/`, `tests/e2e/access_control/`
registry: other.yaml other.auth.*, mgmt.yaml mgmt.*.admin_only
verify: create an internal_user key and confirm POST /team/new with it returns 403 while /chat/completions works

### auth.access_groups: Access groups (/access_group)
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/access_groups, https://docs.litellm.ai/docs/proxy/model_access_groups, https://docs.litellm.ai/docs/proxy/model_access
code: `litellm/proxy/management_endpoints/model_access_group_management_endpoints.py` (`/access_group/*`), `litellm/proxy/management_endpoints/access_group_endpoints.py` (`/v1/access_group`)
tests: `tests/test_litellm/proxy/management_endpoints/`, `tests/e2e/access_control/`
registry: mgmt.yaml mgmt.access_group.*, other.yaml other.auth.model_access_group.*
verify: create an access group over two models, scope a key to it, and confirm a third model is denied

### auth.sso: SSO (Google, Microsoft, Okta, generic OIDC)
surfaces: config, ui | flags: ent
docs: https://docs.litellm.ai/docs/proxy/admin_ui_sso, https://docs.litellm.ai/docs/proxy/custom_sso, https://docs.litellm.ai/docs/oidc
code: `litellm/proxy/management_endpoints/ui_sso.py` (GET `/sso/key/generate`, GET `/sso/callback`)
tests: `tests/test_litellm/proxy/management_endpoints/` (sso related files)
registry: none
verify: configure GOOGLE_CLIENT_ID/SECRET in env, open http://localhost:4000/sso/key/generate, complete Google login, and land in the UI

### auth.saml: SAML SSO
surfaces: config, ui | flags: ent
docs: https://docs.litellm.ai/docs/proxy/saml_sso
code: `litellm/proxy/management_endpoints/ui_sso.py` (GET `/sso/saml/login`, `/sso/saml/metadata`, POST `/sso/saml/callback`)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: none
verify: point the proxy at your IdP metadata, GET /sso/saml/metadata to fetch the SP descriptor, then run the login flow

### auth.cli_sso: CLI login via SSO (device flow)
surfaces: cli, api | flags: ent
docs: https://docs.litellm.ai/docs/proxy/cli_sso
code: `litellm/proxy/management_endpoints/ui_sso.py` (POST `/sso/cli/start`, `/sso/cli/complete/{login_id}`, GET `/sso/cli/poll/{key_id}`), `litellm/proxy/anthropic_endpoints/gateway_endpoints.py` (POST `/oauth/device_authorization`, `/oauth/token`)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: none
verify: run `litellm login` against the proxy, complete the browser flow, and confirm the CLI stores a working virtual key

### auth.jwt: JWT auth (JWKS, role mapping, team mapping)
surfaces: config, api | flags: ent
docs: https://docs.litellm.ai/docs/proxy/token_auth, https://docs.litellm.ai/docs/proxy/jwt_auth_arch, https://docs.litellm.ai/docs/proxy/jwt_key_mapping
code: `litellm/proxy/auth/handle_jwt.py`, `litellm/proxy/management_endpoints/jwt_key_mapping_endpoints.py` (`/jwt/key/mapping/*`), `litellm/proxy/auth/user_api_key_auth.py`
tests: `tests/test_litellm/proxy/auth/` (jwt files), `tests/e2e/other/` (Keycloak-backed JWT suite)
registry: other.yaml other.auth.jwt.*, mgmt.yaml mgmt.jwt_key_mapping.*, mgmt.key.jwt.*
verify: set `JWT_PUBLIC_KEY_URL` to your JWKS, mint a token, call /chat/completions with it, and confirm role mapping applies

### auth.custom_auth: Custom auth hook
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/custom_auth
code: `litellm/proxy/auth/user_api_key_auth.py` (custom_auth hook dispatch), `litellm/proxy/custom_auth_auto.py`
tests: `tests/test_litellm/proxy/auth/`
registry: none
verify: write a `custom_auth` function returning a UserAPIKeyAuth per the doc, register it in config, and confirm it gates a request

### auth.scim: SCIM provisioning
surfaces: api, ui | flags: ent
docs: https://docs.litellm.ai/docs/tutorials/scim_litellm, https://docs.litellm.ai/docs/proxy/identity_provisioning
code: `litellm/proxy/management_endpoints/scim/scim_v2.py` (`/Users`, `/Groups`, `/ServiceProviderConfig`, `/Schemas`)
tests: `tests/test_litellm/proxy/management_endpoints/` (scim files)
registry: none
verify: GET /scim/v2/ServiceProviderConfig with admin creds, then POST /scim/v2/Users to provision a user and see it in /user/list

### auth.ip_allowlist: IP address allowlist
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/proxy/ip_address
code: `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` (`/get/allowed_ips`, `/add/allowed_ip`, `/delete/allowed_ip`), `litellm/proxy/auth/user_api_key_auth.py`
tests: `tests/test_litellm/proxy/` (ip allowlist files)
registry: other.yaml other.auth.ip_allowlist.*
verify: POST /add/allowed_ip with a foreign CIDR, call from a disallowed address, and confirm 403

## UI access and onboarding

### auth.public_routes: Public and admin-only route config
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/proxy/public_routes
code: `litellm/proxy/auth/user_api_key_auth.py` (route classification), `litellm/proxy/_types.py` (`LiteLLMRoutes`)
tests: `tests/test_litellm/proxy/auth/`
registry: none
verify: add a route to `general_settings.public_routes`, hit it with no Authorization header, and confirm 200

### auth.ui_access_control: UI access mode and page visibility
surfaces: config, ui | flags: none
docs: https://docs.litellm.ai/docs/proxy/ui/page_visibility
code: `ui/litellm-dashboard/src/components/leftnav.tsx` (roles gating), `litellm/proxy/management_endpoints/ui_sso.py` (GET `/sso/get/ui_settings`)
tests: `ui/litellm-dashboard/` (leftnav tests)
registry: none
verify: set `general_settings.ui_access_mode: restricted`, log in as an internal user, and confirm admin-only nav items are hidden

### auth.self_serve: Self-serve onboarding, public teams, invitation flow
surfaces: ui, api | flags: db
docs: https://docs.litellm.ai/docs/proxy/self_serve, https://docs.litellm.ai/docs/proxy/public_teams, https://docs.litellm.ai/docs/tutorials/default_team_self_serve
code: `litellm/proxy/proxy_server.py` (`/invitation/new`, `/onboarding/get_token`, `/onboarding/claim_token`), `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` (`/get/default_team_settings`)
tests: `tests/test_litellm/proxy/` (self serve files)
registry: none
verify: POST /invitation/new for a new user, hit the invitation link, and confirm they land with a default team

### auth.password_policy: Password policy and failed login lockout
surfaces: config | flags: none
docs: https://docs.litellm.ai/docs/guides/security_settings
code: `litellm/proxy/_types.py` (`max_failed_login_attempts_per_source`, `failed_login_window_seconds` fields), `litellm/proxy/proxy_server.py` (POST `/login`)
tests: `tests/test_litellm/proxy/` (login lockout files)
registry: none
verify: set `general_settings.max_failed_login_attempts_per_source: 3` with trusted proxies, fail the UI login 4 times, and confirm 429s

### auth.multiple_admins: Multiple admins
surfaces: api, ui | flags: db
docs: https://docs.litellm.ai/docs/proxy/multiple_admins
code: `litellm/proxy/management_endpoints/internal_user_endpoints.py` (role assignment on `/user/new`, `/user/update`)
tests: `tests/test_litellm/proxy/management_endpoints/`
registry: mgmt.yaml mgmt.user.*
verify: create a second user with `user_role: proxy_admin` and confirm they can call /user/list

### auth.oauth2_proxy_token: OAuth2 token auth
surfaces: config | flags: ent
docs: https://docs.litellm.ai/docs/proxy/token_auth
code: `litellm/proxy/auth/handle_jwt.py`, `litellm/proxy/auth/user_api_key_auth.py`
tests: `tests/test_litellm/proxy/auth/`
registry: other.yaml other.auth.oauth2.*
verify: configure the OAuth2 introspection settings per token_auth doc, send a bearer token, and confirm it authenticates
