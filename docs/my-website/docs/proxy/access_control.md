# Role-based Access Controls (RBAC)

Role-based access control (RBAC) is based on Organizations, Teams and Internal User Roles

- `Organizations` are the top-level entities that contain Teams.
- `Team` - A Team is a collection of multiple `Internal Users`
- `Internal Users` - users that can create keys, make LLM API calls, view usage on LiteLLM. Users can be on multiple teams.
- `Roles` define the permissions of an `Internal User`
- `Virtual Keys` - Keys are used for authentication to the LiteLLM API. Keys are tied to a `Internal User` and `Team` 

## Roles

| Role Type | Role Name | Permissions |
|-----------|-----------|-------------|
| **Admin** | `proxy_admin` | Admin over the platform |
| | `proxy_admin_viewer` | Can login, view all keys, view all spend. **Cannot** create keys/delete keys/add new users |
| **Organization** | `org_admin` | Admin over the organization. Can create teams and users within their organization |
| **Internal User** | `internal_user` | Can login, view/create/delete their own keys, view their spend. **Cannot** add new users |
| | `internal_user_viewer` | Can login, view their own keys, view their own spend. **Cannot** create/delete keys, add new users |

## Onboarding Organizations 

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


### `Organization Admin` - Add an `Internal User`

The organization admin will use the virtual key created in [step 2](#2-adding-an-org_admin-to-an-organization) to add an Internal User to the `engineering_team` Team. 

- We will assign role=`internal_user` so the user can create Virtual Keys for themselves
- `team_id` is from [step 3](#3-organization-admin---create-a-team)

```shell
curl -X POST 'http://0.0.0.0:4000/team/member_add' \
    -H 'Authorization: Bearer sk-1234' \
    -H 'Content-Type: application/json' \
    -d '{"team_id": "01044ee8-441b-45f4-be7d-c70e002722d8", "member": {"role": "internal_user", "user_id": "krrish@berri.ai"}}'

```

