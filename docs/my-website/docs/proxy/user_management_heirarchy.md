import Image from '@theme/IdealImage';


# User Management Hierarchy

<Image img={require('../../img/litellm_user_heirarchy.png')} style={{ width: '100%', maxWidth: '4000px' }} />

LiteLLM supports a hierarchy of users, teams, organizations, and budgets.

- Organizations can have multiple teams. [API Reference](https://litellm-api.up.railway.app/#/organization%20management)
- Teams can have multiple users. [API Reference](https://litellm-api.up.railway.app/#/team%20management)
- Users can have multiple keys, and be on multiple teams. [API Reference](https://litellm-api.up.railway.app/#/budget%20management)
- Keys can belong to either a team or a user. [API Reference](https://litellm-api.up.railway.app/#/end-user%20management)

## Entity Overview

| Entity | Database Table | Description | Key Relationships |
|--------|---------------|-------------|-------------------|
| **Organization** | `LiteLLM_OrganizationTable` | Top-level entity for grouping teams | Contains multiple teams |
| **Team** | `LiteLLM_TeamTable` | Group of users with shared settings and budgets | Belongs to an organization, contains multiple users via team membership |
| **Internal User** | `LiteLLM_UserTable` | Users who can create keys and make LLM API calls | Can own keys, can exist across multiple teams, can belong to multiple organizations |
| **API Key** | `LiteLLM_VerificationTokenTable` | Authentication credentials for API access | Can be owned by either a user or a team (or both) |


:::info

See [Access Control](./access_control) for more details on roles and permissions.
:::