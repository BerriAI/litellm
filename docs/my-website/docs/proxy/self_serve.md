import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 🤗 UI - Self-Serve

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


