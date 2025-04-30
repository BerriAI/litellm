# JWT Authentication Default User Role

## Feature Description

This feature allows you to specify the default role that will be assigned to new users who are authenticated via JWT authentication. This ensures that every new user will have the appropriate constraints in place through the default budget, RPM, and budget duration settings based on their role.

## Implementation Details

The following changes were made to implement this feature:

1. Added a new `default_user_role` field to the `LiteLLM_JWTAuth` class in `litellm/proxy/_types.py`. This field defaults to `LitellmUserRoles.INTERNAL_USER`.

2. Modified the `get_user_object` function in `litellm/proxy/auth/auth_checks.py` to accept a `default_user_role` parameter that is used when upserting a new user.

3. Updated the `get_objects` method in `litellm/proxy/auth/handle_jwt.py` to pass the default role from the JWT auth configuration to the `get_user_object` function.

## Configuration

You can configure the default role for new users created via JWT authentication in your proxy configuration file (e.g., `config.yaml`):

```yaml
litellm_jwtauth:
  admin_jwt_scope: "litellm_proxy_admin"
  user_id_jwt_field: "sub"
  user_email_jwt_field: "email"
  user_id_upsert: true
  default_user_role: "internal_user"
```

## Available User Roles

The following roles can be specified:

- `proxy_admin`: Admin over the platform
- `proxy_admin_viewer`: Can login, view all own keys, view all spend
- `org_admin`: Admin over a specific organization
- `internal_user`: Can login, view/create/delete their own keys, view their spend (default)
- `internal_user_viewer`: Can login, view their own keys, view their own spend
- `team`: Used for JWT auth
- `customer`: External users

## Example Usage

### Example Configuration

```yaml
litellm_jwtauth:
  admin_jwt_scope: "litellm_proxy_admin"
  user_id_jwt_field: "sub"
  user_email_jwt_field: "email"
  user_id_upsert: true
  default_user_role: "internal_user"
  user_allowed_email_domain: "yourdomain.com"
```

### Behavior

When a user authenticates via JWT:

1. The proxy extracts the user ID from the JWT token (from the field specified in `user_id_jwt_field`).
2. If the user doesn't exist and `user_id_upsert` is `true`, a new user is created with:
   - The user ID from the token
   - The email from the token (if `user_email_jwt_field` is specified)
   - The role specified in `default_user_role` (defaults to `internal_user` if not specified)
3. The user is now authenticated with the appropriate role and corresponding permissions.
