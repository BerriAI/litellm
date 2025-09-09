import Image from '@theme/IdealImage';


# SCIM with LiteLLM

✨ **Enterprise**: SCIM support requires a premium license.

Enables identity providers (Okta, Azure AD, OneLogin, etc.) to automate user and team (group) provisioning, updates, and deprovisioning on LiteLLM.

This tutorial will walk you through the steps to connect your IDP to LiteLLM SCIM Endpoints.

### Supported SSO Providers for SCIM
Below is a list of supported SSO providers for connecting to LiteLLM SCIM Endpoints.
- Microsoft Entra ID (Azure AD)
- Okta
- Google Workspace
- OneLogin
- Keycloak
- Auth0

### Supported Dynamic Operations

| Operation         | Supported | Details & Notes                                                                 |
|-------------------|-----------|-------------------------------------------------------------------------------|
| User Provisioning | Yes       | Automatically create users in LiteLLM when assigned in your IdP. If the user already exists, a conflict error is returned. If no role is specified, defaults to "viewer" or your configured default. |
| User Update       | Yes       | Update user attributes (name, email, teams, metadata) via SCIM PUT or PATCH. PATCH supports granular updates, such as changing display name or group membership. |
| Team Membership   | Yes       | Add or remove users from teams by updating group membership in your IdP. LiteLLM reflects these changes dynamically, updating the user’s access and permissions. |
| Team Provisioning | Yes       | SCIM can create, update, and delete teams (groups) in LiteLLM, including changing team names, adding/removing members, and storing/updating team metadata. |
| User Deletion     | Yes       | When a user is removed from your IdP or a group, LiteLLM will remove the user from all teams, **delete all API keys and tokens associated with the user**, and remove the user account from LiteLLM. |
| Error Handling    | Yes       | Returns standard SCIM error codes for conflicts, not found, etc.              |

> **Security Note:**
When a user is removed via SCIM, all their API keys and tokens are deleted immediately for security.

## Tutorial

## 1. Get your SCIM Tenant URL and Bearer Token

On LiteLLM, navigate to the Settings > Admin Settings > SCIM. On this page you will create a SCIM Token, this allows your IDP to authenticate to litellm `/scim` endpoints.

<Image img={require('../../img/scim_2.png')}  style={{ width: '800px', height: 'auto' }} />

## 2. Connect your IDP to LiteLLM SCIM Endpoints

On your IDP provider, navigate to your SSO application and select `Provisioning` > `New provisioning configuration`.

On this page, paste in your litellm scim tenant url and bearer token.

Once this is pasted in, click on `Test Connection` to ensure your IDP can authenticate to the LiteLLM SCIM endpoints.

<Image img={require('../../img/scim_4.png')}  style={{ width: '800px', height: 'auto' }} />


## 3. Test SCIM Connection

### 3.1 Assign the group to your LiteLLM Enterprise App

On your IDP Portal, navigate to `Enterprise Applications` > Select your litellm app 

<Image img={require('../../img/msft_enterprise_app.png')}  style={{ width: '800px', height: 'auto' }} />

<br />
<br />

Once you've selected your litellm app, click on `Users and Groups` > `Add user/group` 

<Image img={require('../../img/msft_enterprise_assign_group.png')}  style={{ width: '800px', height: 'auto' }} />

<br />

Now select the group you created in step 1.1. And add it to the LiteLLM Enterprise App. At this point we have added `Production LLM Evals Group` to the LiteLLM Enterprise App. The next step is having LiteLLM automatically create the `Production LLM Evals Group` on the LiteLLM DB when a new user signs in.

<Image img={require('../../img/msft_enterprise_select_group.png')}  style={{ width: '800px', height: 'auto' }} />


### 3.2 Sign in to LiteLLM UI via SSO

Sign into the LiteLLM UI via SSO. You should be redirected to the Entra ID SSO page. This SSO sign in flow will trigger LiteLLM to fetch the latest Groups and Members from Azure Entra ID.

<Image img={require('../../img/msft_sso_sign_in.png')}  style={{ width: '800px', height: 'auto' }} />

### 3.3 Check the new team on LiteLLM UI

On the LiteLLM UI, Navigate to `Teams`, You should see the new team `Production LLM Evals Group` auto-created on LiteLLM. 

<Image img={require('../../img/msft_auto_team.png')}  style={{ width: '900px', height: 'auto' }} />

---



---

## For Developers: Test Coverage

Our test suite covers:
- User creation (including conflict and default role logic)
- User updates (PUT/PATCH, including granular attribute and group changes)
- Team membership management (add/remove, atomic updates)
- Deprovisioning (user and key deletion)
- Group/team management (metadata serialization, membership as source of truth)
- Error handling (conflict, not found, etc.)

---




