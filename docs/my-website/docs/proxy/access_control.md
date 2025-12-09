import Image from '@theme/IdealImage';

# Role-based Access Controls (RBAC)

Role-based access control (RBAC) is based on Organizations, Teams and Internal User Roles

<Image img={require('../../img/litellm_user_heirarchy.png')} style={{ width: '100%', maxWidth: '4000px' }} />


- `Organizations` are the top-level entities that contain Teams.
- `Team` - A Team is a collection of multiple `Internal Users`
- `Internal Users` - users that can create keys, make LLM API calls, view usage on LiteLLM. Users can be on multiple teams.
- `Virtual Keys` - Keys are used for authentication to the LiteLLM API. Each key can optionally be associated with a `user_id`, a `team_id`, or both:
  - **User-only key**: Has a `user_id` but no `team_id`. Tracked individually, deleted when the user is deleted.
  - **Team key (Service Account)**: Has a `team_id` but no `user_id`. Shared by the team, not deleted when users are removed. [Learn more about service account keys](https://docs.litellm.ai/docs/proxy/virtual_keys#service-account-keys).
  - **User + Team key**: Has both `user_id` and `team_id`. Belongs to a specific user within a team context.

### When to Use Each Key Type

| Key Type | Use Case | Spend Tracking | Lifecycle |
|----------|----------|----------------|-----------|
| **User-only** | Personal API keys for individual developers | Tracked to the user | Deleted when user is deleted |
| **Team (Service Account)** | Production apps, CI/CD pipelines, shared services | Tracked to the team only | Persists even when team members leave |
| **User + Team** | User working within a team context | Tracked to both user and team | Deleted when user is deleted |

**Example scenarios:**
- Use **user-only keys** for developers testing locally
- Use **team service account keys** for your production application that shouldn't break when employees leave
- Use **user + team keys** when you want individual accountability within a team budget

---

## User Roles

LiteLLM has two types of roles:

1. **Global Proxy Roles** - Platform-wide roles that apply across all organizations and teams
2. **Organization/Team Specific Roles** - Roles scoped to specific organizations or teams (**Premium Feature**)

### Global Proxy Roles

| Role Name | Permissions |
|-----------|-------------|
| `proxy_admin` | Admin over the entire platform. Full control over all organizations, teams, and users |
| `proxy_admin_viewer` | Can login, view all keys, view all spend across the platform. **Cannot** create keys/delete keys/add new users |
| `internal_user` | Can login, view/create (when allowed by team-specific permissions)/delete their own keys, view their spend. **Cannot** add new users |
| `internal_user_viewer` | ⚠️ **DEPRECATED** - Use team/org specific roles instead. Can login, view their own keys, view their own spend. **Cannot** create/delete keys, add new users |

### Organization/Team Specific Roles

| Role Name | Permissions |
|-----------|-------------|
| `org_admin` | Admin over a specific organization. Can create teams and users within their organization ✨ **Premium Feature** |
| `team_admin` | Admin over a specific team. Can manage team members, update team settings, and create keys for their team. ✨ **Premium Feature** |

## What Can Each Role Do?

Here's what each role can actually do. Think of it like levels of access.

---

## Global Proxy Roles

These roles apply across the entire LiteLLM platform, regardless of organization or team boundaries.

### Proxy Admin - Full Access

The proxy admin controls everything. They're like the owner of the whole platform.

**What they can do:**
- Create and manage all organizations
- Create and manage all teams (across all organizations)
- Create and manage all users
- View all spend and usage across the platform
- Create and delete keys for anyone
- Update team budgets, rate limits, and models
- Manage team members and assign roles

**Who should be a proxy admin:** Only the people running the LiteLLM instance.

---

### Proxy Admin Viewer - Platform-Wide Read Access

The proxy admin viewer can see everything across the platform but cannot make changes.

**What they can do:**
- View all organizations, teams, and users
- View all spend and usage across the platform
- View all API keys
- Login to the admin dashboard

**What they cannot do:**
- Create or delete keys
- Add or remove users
- Modify budgets, rate limits, or settings
- Make any changes to the platform

**Who should be a proxy admin viewer:** Finance teams, auditors, or stakeholders who need platform-wide visibility without modification rights.

---

### Internal User

An internal user can create API keys (when allowed by team-specific permissions) and make calls. They see their own stuff only. They can become a team admin or org admin if they are assigned the respective roles.

**What they can do:**
- Create API keys for themselves
- Delete their own API keys
- View their own spend and usage
- Make API calls using their keys


**Who should be an internal user:** Anyone who needs UI access for team/org specific operations **OR** for developers you plan to give multiple keys to.

---

### Internal User Viewer - Read-Only Access

:::warning DEPRECATED
This role is deprecated in favor of team/org specific roles. Use `org_admin` or `team_admin` roles for better granular control over user permissions within organizations and teams.
:::

An internal user viewer can view their own information but cannot create or delete keys.

**What they can do:**
- View their own API keys
- View their own spend and usage
- Login to see their dashboard

**What they cannot do:**
- Create or delete API keys
- Make changes to any settings
- Create teams or add users
- View other people's information

**Who should be an internal user viewer (deprecated):** Consider using team/org specific roles instead for better access control.

---

## Organization/Team Specific Roles

:::info 
Organization/Team specific roles are premium features. You need to be a LiteLLM Enterprise user to use them. [Get a 7 day trial here](https://www.litellm.ai/#trial).
:::

These roles are scoped to specific organizations or teams. Users with these roles can only manage resources within their assigned organization or team.

### Org Admin - Organization Level Access

An org admin manages one or more organizations. They can create teams within their organization but can't touch other organizations.

**What they can do:**
- Create teams within their organization
- Add users to teams in their organization
- View spend for their organization
- Create keys for users in their organization

**What they cannot do:**
- Create or manage other organizations
- Modify org budgets / rate limits
- Modify org allowed models (e.g. adding a proxy-level model to the org)

**Who should be an org admin:** Department leads or managers who need to manage multiple teams.

---

### Team Admin - Team Level Access

✨ **This is a Premium Feature**

A team admin manages a specific team. They're like a team lead who can add people, update settings, but only for their team.

**What they can do:**
- Add or remove team members from their team
- Update team members' budgets and rate limits within the team
- Change team settings (budget, rate limits, models)
- Create and delete keys for team members
- Onboard a [team-BYOK](./team_model_add) model to LiteLLM (e.g. onboarding a team's finetuned model)
- Configure [team member permissions](#team-member-permissions) to control what regular team members can do

**What they cannot do:**
- Create new teams
- Modify team's budget / rate limits
- Add/remove global proxy models to their team


**Who should be a team admin:** Team leads who need to manage their team's API access without bothering IT.

:::info How to create a team admin

You need to be a LiteLLM Enterprise user to assign team admins. [Get a 7 day trial here](https://www.litellm.ai/#trial).

```shell
curl -X POST 'http://0.0.0.0:4000/team/member_add' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{"team_id": "team-123", "member": {"role": "admin", "user_id": "user@company.com"}}'
```

:::

---

## Team Member Permissions

✨ **This is a Premium Feature**

Team member permissions allow you to control what regular team members (with role=`user`) can do with API keys in their team. By default, team members can only view key information, but you can grant them additional permissions to create, update, or delete keys.

### How It Works

- **Applies to**: Team members with role=`user` (not team admins or org admins)
- **Scope**: Permissions only apply to keys belonging to their team
- **Configuration**: Set at the team level using `team_member_permissions`
- **Override**: Team admins and org admins always have full permissions regardless of these settings

### Available Permissions

| Permission | Method | Description |
|-----------|--------|-------------|
| `/key/info` | GET | View information about virtual keys in the team |
| `/key/health` | GET | Check health status of virtual keys in the team |
| `/key/list` | GET | List all virtual keys belonging to the team |
| `/key/generate` | POST | Create new virtual keys for the team |
| `/key/service-account/generate` | POST | Create service account keys (not tied to a specific user) for the team |
| `/key/update` | POST | Modify existing virtual keys in the team |
| `/key/delete` | POST | Delete virtual keys belonging to the team |
| `/key/regenerate` | POST | Regenerate virtual keys in the team |
| `/key/block` | POST | Block virtual keys in the team |
| `/key/unblock` | POST | Unblock virtual keys in the team |

### Default Permissions

By default, team members can only:
- `/key/info` - View key information
- `/key/health` - Check key health

### Common Permission Scenarios

**Read-only access** (default):
```json
["/key/info", "/key/health"]
```

**Allow key creation but not deletion**:
```json
["/key/info", "/key/health", "/key/generate", "/key/update"]
```

**Full key management**:
```json
["/key/info", "/key/health", "/key/generate", "/key/update", "/key/delete", "/key/regenerate", "/key/block", "/key/unblock", "/key/list"]
```

### How to Configure Team Member Permissions

#### View Current Permissions

```shell
curl --location 'http://0.0.0.0:4000/team/permissions_list?team_id=team-123' \
    --header 'Authorization: Bearer sk-1234'
```

Expected Response:
```json
{
  "team_id": "team-123",
  "team_member_permissions": ["/key/info", "/key/health"],
  "all_available_permissions": ["/key/generate", "/key/update", "/key/delete", ...]
}
```

#### Update Team Member Permissions

```shell
curl --location 'http://0.0.0.0:4000/team/update' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "team-123",
        "team_member_permissions": ["/key/info", "/key/health", "/key/generate", "/key/update"]
    }'
```

This allows team members to:
- View key information
- Create new keys
- Update existing keys
- But NOT delete keys

### Who Can Configure These Permissions?

- **Proxy Admin**: Can configure permissions for any team
- **Org Admin**: Can configure permissions for teams in their organization
- **Team Admin**: Can configure permissions for their own team

---

## Quick Comparison

Here's the quick version:

### Global Proxy Roles

| Action | Proxy Admin | Proxy Admin Viewer | Internal User | Internal User Viewer ⚠️ (Deprecated) |
|--------|-------------|-------------------|---------------|-------------------------------------|
| Create organizations | ✅ | ❌ | ❌ | ❌ |
| Create teams | ✅ | ❌ | ❌ | ❌ |
| Manage all teams | ✅ | ❌ | ❌ | ❌ |
| Create/delete any keys | ✅ | ❌ | ❌ | ❌ |
| Create/delete own keys | ✅ | ❌ | ✅ | ❌ |
| View all platform spend | ✅ | ✅ | ❌ | ❌ |
| View own spend | ✅ | ✅ | ✅ | ✅ |
| View all keys | ✅ | ✅ | ❌ | ❌ |
| View own keys | ✅ | ✅ | ✅ | ✅ |
| Add/remove users | ✅ | ❌ | ❌ | ❌ |

> **Note:** The `internal_user_viewer` role is deprecated. Use team/org specific roles for better granular access control.

### Organization/Team Specific Roles

| Action | Org Admin | Team Admin |
|--------|-----------|------------|
| Create teams (in their org) | ✅ | ❌ |
| Manage teams in their org | ✅ | ❌ |
| Manage their specific team | ✅ | ✅ |
| Add/remove team members | ✅ (in their org) | ✅ (their team only) |
| Update team budgets | ✅ (in their org) | ✅ (their team only) |
| Create keys for team members | ✅ (in their org) | ✅ (their team only) |
| View organization spend | ✅ (their org) | ❌ |
| View team spend | ✅ (in their org) | ✅ (their team) |
| Create organizations | ❌ | ❌ |
| View all platform spend | ❌ | ❌ |

## Onboarding Organizations 

✨ **This is a Premium Feature**

### 1. Creating a new Organization

Any user with role=`proxy_admin` can create a new organization

**Usage**

[**API Reference for /organization/new**](https://litellm-api.up.railway.app/#/organization%20management/new_organization_organization_new_post)

```shell
curl --location 'http://0.0.0.0:4000/organization/new' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "organization_alias": "marketing_department",
        "models": ["gpt-4"],
        "max_budget": 20
    }'
```

Expected Response 

```json
{
  "organization_id": "ad15e8ca-12ae-46f4-8659-d02debef1b23",
  "organization_alias": "marketing_department",
  "budget_id": "98754244-3a9c-4b31-b2e9-c63edc8fd7eb",
  "metadata": {},
  "models": [
    "gpt-4"
  ],
  "created_by": "109010464461339474872",
  "updated_by": "109010464461339474872",
  "created_at": "2024-10-08T18:30:24.637000Z",
  "updated_at": "2024-10-08T18:30:24.637000Z"
}
```


### 2. Adding an `org_admin` to an Organization

Create a user (ishaan@berri.ai) as an `org_admin` for the `marketing_department` Organization (from [step 1](#1-creating-a-new-organization))

Users with the following roles can call `/organization/member_add`
- `proxy_admin`
- `org_admin` only within their own organization

```shell
curl -X POST 'http://0.0.0.0:4000/organization/member_add' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{"organization_id": "ad15e8ca-12ae-46f4-8659-d02debef1b23", "member": {"role": "org_admin", "user_id": "ishaan@berri.ai"}}'
```

Now a user with user_id = `ishaan@berri.ai` and role = `org_admin` has been created in the `marketing_department` Organization

Create a Virtual Key for user_id = `ishaan@berri.ai`. The User can then use the Virtual key for their Organization Admin Operations

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
        --header 'Authorization: Bearer sk-1234' \
        --header 'Content-Type: application/json' \
        --data '{
            "user_id": "ishaan@berri.ai"
    }'
```

Expected Response 

```json
{
  "models": [],
  "user_id": "ishaan@berri.ai",
  "key": "sk-7shH8TGMAofR4zQpAAo6kQ",
  "key_name": "sk-...o6kQ",
}
```

### 3. `Organization Admin` - Create a Team

The organization admin will use the virtual key created in [step 2](#2-adding-an-org_admin-to-an-organization) to create a `Team` within the `marketing_department` Organization

```shell
curl --location 'http://0.0.0.0:4000/team/new' \
    --header 'Authorization: Bearer sk-7shH8TGMAofR4zQpAAo6kQ' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_alias": "engineering_team",
        "organization_id": "ad15e8ca-12ae-46f4-8659-d02debef1b23"
    }'
```

This will create the team `engineering_team` within the `marketing_department` Organization

Expected Response 

```json
{
  "team_alias": "engineering_team",
  "team_id": "01044ee8-441b-45f4-be7d-c70e002722d8",
  "organization_id": "ad15e8ca-12ae-46f4-8659-d02debef1b23",
}
```


### 4. `Organization Admin` - Add a Team Admin

✨ **This is a Premium Feature**

The organization admin can now add a team admin who will manage the `engineering_team`. 

- We assign role=`admin` to make them a team admin for this specific team
- `team_id` is from [step 3](#3-organization-admin---create-a-team)

```shell
curl -X POST 'http://0.0.0.0:4000/team/member_add' \
    -H 'Authorization: Bearer sk-7shH8TGMAofR4zQpAAo6kQ' \
    -H 'Content-Type: application/json' \
    -d '{"team_id": "01044ee8-441b-45f4-be7d-c70e002722d8", "member": {"role": "admin", "user_id": "john@company.com"}}'
```

Now `john@company.com` is a team admin. They can manage the `engineering_team` - add members, update budgets, create keys - but they can't touch other teams.

Create a Virtual Key for the team admin:

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
    --header 'Authorization: Bearer sk-7shH8TGMAofR4zQpAAo6kQ' \
    --header 'Content-Type: application/json' \
    --data '{"user_id": "john@company.com"}'
```

Expected Response:

```json
{
  "models": [],
  "user_id": "john@company.com",
  "key": "sk-TeamAdminKey123",
  "key_name": "sk-...Key123"
}
```

### 5. `Team Admin` - Add Team Members

Now the team admin can use their key to add team members without needing to ask the org admin.

```shell
curl -X POST 'http://0.0.0.0:4000/team/member_add' \
    -H 'Authorization: Bearer sk-TeamAdminKey123' \
    -H 'Content-Type: application/json' \
    -d '{"team_id": "01044ee8-441b-45f4-be7d-c70e002722d8", "member": {"role": "user", "user_id": "krrish@berri.ai"}}'
```

The team admin can also create keys for their team members:

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
    --header 'Authorization: Bearer sk-TeamAdminKey123' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "krrish@berri.ai",
        "team_id": "01044ee8-441b-45f4-be7d-c70e002722d8"
    }'
```

### 6. `Team Admin` - Update Team Settings

The team admin can update team budgets and rate limits:

```shell
curl --location 'http://0.0.0.0:4000/team/update' \
    --header 'Authorization: Bearer sk-TeamAdminKey123' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "01044ee8-441b-45f4-be7d-c70e002722d8",
        "max_budget": 100,
        "rpm_limit": 1000
    }'
```

