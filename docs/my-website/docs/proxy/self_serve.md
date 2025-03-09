import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Internal User Self-Serve

## Allow users to create their own keys on [Proxy UI](./ui.md).

1. Add user with permissions to a team on proxy 

<Tabs>
<TabItem value="ui" label="UI">

Go to `Internal Users` -> `+New User`

<Image img={require('../../img/add_internal_user.png')}  style={{ width: '800px', height: 'auto' }} />

</TabItem>
<TabItem value="api" label="API">

Create a new Internal User on LiteLLM and assign them the role `internal_user`.

```bash
curl -X POST '<PROXY_BASE_URL>/user/new' \
-H 'Authorization: Bearer <PROXY_MASTER_KEY>' \
-H 'Content-Type: application/json' \
-D '{
    "user_email": "krrishdholakia@gmail.com",
    "user_role": "internal_user" # 👈 THIS ALLOWS USER TO CREATE/VIEW/DELETE THEIR OWN KEYS + SEE THEIR SPEND
}'
```

Expected Response 

```bash
{
    "user_id": "e9d45c7c-b20b-4ff8-ae76-3f479a7b1d7d", 👈 USE IN STEP 2
    "user_email": "<YOUR_USERS_EMAIL>",
    "user_role": "internal_user",
    ...
}
```

Here's the available UI roles for a LiteLLM Internal User: 

Admin Roles:
  - `proxy_admin`: admin over the platform
  - `proxy_admin_viewer`: can login, view all keys, view all spend. **Cannot** create/delete keys, add new users.

Internal User Roles:
  - `internal_user`: can login, view/create/delete their own keys, view their spend. **Cannot** add new users.
  - `internal_user_viewer`: can login, view their own keys, view their own spend. **Cannot** create/delete keys, add new users.

</TabItem>
</Tabs>

2. Share invitation link with user 

<Tabs>
<TabItem value="ui" label="UI">

Copy the invitation link with the user 

<Image img={require('../../img/invitation_link.png')}  style={{ width: '800px', height: 'auto' }} />

</TabItem>
<TabItem value="api" label="API">

```bash
curl -X POST '<PROXY_BASE_URL>/invitation/new' \
-H 'Authorization: Bearer <PROXY_MASTER_KEY>' \
-H 'Content-Type: application/json' \
-D '{
    "user_id": "e9d45c7c-b20b..." # 👈 USER ID FROM STEP 1
}'
```

Expected Response 

```bash
{
    "id": "a2f0918f-43b0-4770-a664-96ddd192966e",
    "user_id": "e9d45c7c-b20b..",
    "is_accepted": false,
    "accepted_at": null,
    "expires_at": "2024-06-13T00:02:16.454000Z", # 👈 VALID FOR 7d
    "created_at": "2024-06-06T00:02:16.454000Z",
    "created_by": "116544810872468347480",
    "updated_at": "2024-06-06T00:02:16.454000Z",
    "updated_by": "116544810872468347480"
}
```

Invitation Link: 

```bash
http://0.0.0.0:4000/ui/onboarding?id=a2f0918f-43b0-4770-a664-96ddd192966e

# <YOUR_PROXY_BASE_URL>/ui/onboarding?id=<id>
```

</TabItem>
</Tabs>

:::info

Use [Email Notifications](./email.md) to email users onboarding links 

:::

3. User logs in via email + password auth

<Image img={require('../../img/ui_clean_login.png')}  style={{ width: '500px', height: 'auto' }} />



:::info 

LiteLLM Enterprise: Enable [SSO login](./ui.md#setup-ssoauth-for-ui)

:::

4. User can now create their own keys


<Image img={require('../../img/ui_self_serve_create_key.png')}  style={{ width: '800px', height: 'auto' }} />

## Allow users to View Usage, Caching Analytics

1. Go to Internal Users -> +Invite User

Set their role to `Admin Viewer` - this means they can only view usage, caching analytics

<Image img={require('../../img/ui_invite_user.png')}  style={{ width: '800px', height: 'auto' }} />
<br />

2. Share invitation link with user


<Image img={require('../../img/ui_invite_link.png')}  style={{ width: '800px', height: 'auto' }} />
<br />

3. User logs in via email + password auth

<Image img={require('../../img/ui_clean_login.png')}  style={{ width: '500px', height: 'auto' }} />
<br />

4. User can now view Usage, Caching Analytics

<Image img={require('../../img/ui_usage.png')}  style={{ width: '800px', height: 'auto' }} />


## Available Roles
Here's the available UI roles for a LiteLLM Internal User: 

**Admin Roles:**
  - `proxy_admin`: admin over the platform
  - `proxy_admin_viewer`: can login, view all keys, view all spend. **Cannot** create/delete keys, add new users.

**Internal User Roles:**
  - `internal_user`: can login, view/create/delete their own keys, view their spend. **Cannot** add new users.
  - `internal_user_viewer`: can login, view their own keys, view their own spend. **Cannot** create/delete keys, add new users.

## Advanced
### Setting custom logout URLs

Set `PROXY_LOGOUT_URL` in your .env if you want users to get redirected to a specific URL when they click logout

```
export PROXY_LOGOUT_URL="https://www.google.com"
```

<Image img={require('../../img/ui_logout.png')}  style={{ width: '400px', height: 'auto' }} />


### Set max budget for internal users 

Automatically apply budget per internal user when they sign up. By default the table will be checked every 10 minutes, for users to reset. To modify this, [see this](./users.md#reset-budgets)

```yaml
litellm_settings:
  max_internal_user_budget: 10
  internal_user_budget_duration: "1mo" # reset every month
```

This sets a max budget of $10 USD for internal users when they sign up. 

This budget only applies to personal keys created by that user - seen under `Default Team` on the UI. 

<Image img={require('../../img/max_budget_for_internal_users.png')}  style={{ width: '500px', height: 'auto' }} />

This budget does not apply to keys created under non-default teams.


### Set max budget for teams

[**Go Here**](./team_budgets.md)

### Auto-add SSO users to teams

1. Specify the JWT field that contains the team ids, that the user belongs to. 

```yaml
general_settings:
  master_key: sk-1234
  litellm_jwtauth:
    team_ids_jwt_field: "groups" # 👈 CAN BE ANY FIELD
```

This is assuming your SSO token looks like this:
```
{
  ...,
  "groups": ["team_id_1", "team_id_2"]
}
```

2. Create the teams on LiteLLM 

```bash
curl -X POST '<PROXY_BASE_URL>/team/new' \
-H 'Authorization: Bearer <PROXY_MASTER_KEY>' \
-H 'Content-Type: application/json' \
-D '{
    "team_alias": "team_1",
    "team_id": "team_id_1" # 👈 MUST BE THE SAME AS THE SSO GROUP ID
}'
```

3. Test the SSO flow

Here's a walkthrough of [how it works](https://www.loom.com/share/8959be458edf41fd85937452c29a33f3?sid=7ebd6d37-569a-4023-866e-e0cde67cb23e)

### Restrict Users from creating personal keys 

This is useful if you only want users to create keys under a specific team. 

This will also prevent users from using their session tokens on the test keys chat pane. 

👉 [**See this**](./virtual_keys.md#restricting-key-generation)

## **All Settings for Self Serve / SSO Flow**

```yaml
litellm_settings:
  max_internal_user_budget: 10        # max budget for internal users
  internal_user_budget_duration: "1mo" # reset every month

  default_internal_user_params:    # Default Params used when a new user signs in Via SSO
    user_role: "internal_user"     # one of "internal_user", "internal_user_viewer", "proxy_admin", "proxy_admin_viewer". New SSO users not in litellm will be created as this user
    max_budget: 100                # Optional[float], optional): $100 budget for a new SSO sign in user
    budget_duration: 30d           # Optional[str], optional): 30 days budget_duration for a new SSO sign in user
    models: ["gpt-3.5-turbo"]      # Optional[List[str]], optional): models to be used by a new SSO sign in user


  upperbound_key_generate_params:    # Upperbound for /key/generate requests when self-serve flow is on
    max_budget: 100 # Optional[float], optional): upperbound of $100, for all /key/generate requests
    budget_duration: "10d" # Optional[str], optional): upperbound of 10 days for budget_duration values
    duration: "30d" # Optional[str], optional): upperbound of 30 days for all /key/generate requests
    max_parallel_requests: 1000 # (Optional[int], optional): Max number of requests that can be made in parallel. Defaults to None.
    tpm_limit: 1000 #(Optional[int], optional): Tpm limit. Defaults to None.
    rpm_limit: 1000 #(Optional[int], optional): Rpm limit. Defaults to None.

  key_generation_settings: # Restricts who can generate keys. [Further docs](./virtual_keys.md#restricting-key-generation)
    team_key_generation:
      allowed_team_member_roles: ["admin"]
    personal_key_generation: # maps to 'Default Team' on UI 
      allowed_user_roles: ["proxy_admin"]
```
