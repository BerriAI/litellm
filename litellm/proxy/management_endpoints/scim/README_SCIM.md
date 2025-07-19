# SCIM v2 Integration for LiteLLM Proxy

This module provides SCIM v2 (System for Cross-domain Identity Management) endpoints for LiteLLM Proxy, allowing identity providers to manage users and teams (groups) within the LiteLLM ecosystem.

## Overview

SCIM is an open standard designed to simplify user management across different systems. This implementation allows compatible identity providers (like Okta, Azure AD, OneLogin, etc.) to automatically provision and deprovision users and groups in LiteLLM Proxy.

## Endpoints

The SCIM v2 API follows the standard specification with the following base URL:

```
/scim/v2
```

### User Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/Users` | GET | List all users with pagination support |
| `/Users/{user_id}` | GET | Get a specific user by ID |
| `/Users` | POST | Create a new user |
| `/Users/{user_id}` | PUT | Update an existing user |
| `/Users/{user_id}` | DELETE | Delete a user |

### Group Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/Groups` | GET | List all groups with pagination support |
| `/Groups/{group_id}` | GET | Get a specific group by ID |
| `/Groups` | POST | Create a new group |
| `/Groups/{group_id}` | PUT | Update an existing group |
| `/Groups/{group_id}` | DELETE | Delete a group |

## SCIM Schema

This implementation follows the standard SCIM v2 schema with the following mappings:

### Users

- SCIM User ID → LiteLLM `user_id`
- SCIM User Email → LiteLLM `user_email`
- SCIM User Group Memberships → LiteLLM User-Team relationships

### Groups

- SCIM Group ID → LiteLLM `team_id`
- SCIM Group Display Name → LiteLLM `team_alias`
- SCIM Group Members → LiteLLM Team members list

## Configuration

To enable SCIM in your identity provider, use the full URL to the SCIM endpoint:

```
https://your-litellm-proxy-url/scim/v2
```

Most identity providers will require authentication. You should use a valid LiteLLM API key with administrative privileges.

## Features

- Full CRUD operations for users and groups
- Pagination support 
- Basic filtering support
- Automatic synchronization of user-team relationships
- Proper status codes and error handling per SCIM specification


## Example Usage

### Listing Users

```
GET /scim/v2/Users?startIndex=1&count=10
```

### Creating a User

```json
POST /scim/v2/Users
{
  "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
  "userName": "john.doe@example.com",
  "active": true,
  "emails": [
    {
      "value": "john.doe@example.com",
      "primary": true
    }
  ]
}
```

### Adding a User to Groups

```json
PUT /scim/v2/Users/{user_id}
{
  "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
  "userName": "john.doe@example.com",
  "active": true,
  "emails": [
    {
      "value": "john.doe@example.com",
      "primary": true
    }
  ],
  "groups": [
    {
      "value": "team-123",
      "display": "Engineering Team"
    }
  ]
}
``` 