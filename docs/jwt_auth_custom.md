# Custom JWT Authentication for LiteLLM Proxy

This guide shows you how to configure LiteLLM Proxy to authenticate requests using JWT tokens from an external authentication provider like Flask-Security, Auth0, or any OAuth2/OpenID Connect provider.

## Overview

LiteLLM's custom JWT authentication allows you to:
- Integrate with existing authentication systems
- Maintain user context for cost tracking and logging  
- Use external JWT providers without requiring LiteLLM Enterprise
- Leverage all of LiteLLM's features (budgets, rate limits, etc.)

## Configuration

### Step 1: Enable Custom JWT Authentication

Add the following to your `config.yaml`:

```yaml
general_settings:
  # Enable custom JWT authentication
  custom_auth: custom_jwt_auth.jwt_auth
  
  # JWT authentication settings
  jwt_settings:
    # Your JWT issuer (the 'iss' claim in the JWT)
    issuer: "https://your-auth-provider.com"
    
    # The audience for your LiteLLM proxy (the 'aud' claim in the JWT)  
    audience: "litellm-proxy"
    
    # URL to fetch JWKS (JSON Web Key Set) for signature verification
    public_key_url: "https://your-auth-provider.com/.well-known/jwks.json"
    
    # JWT algorithm (typically RS256 for RSA signatures)
    algorithm: "RS256"
    
    # Clock skew allowance in seconds (optional, default: 0)
    leeway: 30
    
    # Map JWT claims to LiteLLM user context
    user_claim_mappings:
      user_id: "sub"          # JWT 'sub' claim -> LiteLLM user_id
      user_email: "email"     # JWT 'email' claim -> LiteLLM user_email
      user_role: "role"       # JWT 'role' claim -> LiteLLM user_role
      team_id: "team"         # JWT 'team' claim -> LiteLLM team_id (optional)

model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
```

### Step 2: Configure Your Models

Add your models to the `model_list` section as usual. JWT authentication works with any LiteLLM-supported model.

### Step 3: Optional Database Setup

For cost tracking and user management, configure a database:

```yaml
general_settings:
  database_url: "postgresql://user:pass@localhost/litellm"

litellm_settings:
  success_callback: ["postgres"]
  failure_callback: ["postgres"]
```

## JWT Token Requirements

Your JWT tokens must include:

### Required Claims
- `iss` (issuer): Must match the `issuer` in your config
- `aud` (audience): Must match the `audience` in your config  
- `exp` (expiration): Token expiration time
- `sub` (subject): User identifier

### Optional Claims
- `email`: User email address
- `role`: User role for permission mapping
- `team`: Team/group identifier for cost tracking

### Example JWT Payload
```json
{
  "sub": "user123",
  "email": "user@example.com", 
  "role": "admin",
  "team": "engineering",
  "iss": "https://your-auth-provider.com",
  "aud": "litellm-proxy",
  "iat": 1701234567,
  "exp": 1701238167
}
```

## Role Mappings

The following external roles are automatically mapped to LiteLLM roles:

| External Role | LiteLLM Role | Permissions |
|---------------|--------------|-------------|
| `admin`, `proxy_admin` | `PROXY_ADMIN` | Full admin access |
| `user`, `internal_user` | `INTERNAL_USER` | Standard user access |
| `viewer`, `internal_user_viewer` | `INTERNAL_USER_VIEW_ONLY` | Read-only access |
| `team` | `TEAM` | Team-scoped access |
| `customer` | `CUSTOMER` | Customer access |
| Any other role | `INTERNAL_USER` | Default user access |

## Integration Examples

### Flask-Security Integration

If you're using Flask-Security, here's how to set it up:

#### 1. Configure Flask-Security to Issue JWTs

```python
from flask import Flask
from flask_security import Security, UserMixin, RoleMixin, jwt

app = Flask(__name__)

# Configure Flask-Security JWT settings
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_ALGORITHM'] = 'RS256'

# Generate RSA key pair for signing
private_key = ... # Your RSA private key
public_key = ... # Your RSA public key

@app.route('/auth/token', methods=['POST'])
@auth_required()
def get_jwt_token():
    # Create JWT with required claims
    additional_claims = {
        "email": current_user.email,
        "role": current_user.roles[0].name if current_user.roles else "user",
        "team": getattr(current_user, 'team', None)
    }
    
    token = jwt.encode_token(current_user, additional_claims)
    return {"access_token": token}
```

#### 2. Expose JWKS Endpoint

```python
@app.route('/.well-known/jwks.json')
def jwks():
    # Convert your public key to JWK format
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": "my-key-id", 
                "use": "sig",
                "alg": "RS256",
                "n": "...",  # Base64url-encoded modulus
                "e": "AQAB"  # Base64url-encoded exponent
            }
        ]
    }
```

### Using with LiteLLM

Once configured, authenticate requests to LiteLLM using the JWT:

```bash
# Get JWT from your auth provider
JWT_TOKEN=$(curl -X POST "https://your-auth-provider.com/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password"}' \
  | jq -r '.access_token')

# Use JWT with LiteLLM
curl -X POST "http://localhost:4000/v1/chat/completions" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Cost Tracking and User Management

With JWT authentication enabled, LiteLLM automatically tracks costs per user:

- **User-level tracking**: Based on the `sub` claim in the JWT
- **Team-level tracking**: Based on the configured team claim  
- **Email association**: Links costs to user email addresses
- **Role-based access**: Controls what endpoints users can access

You can view spending data through the LiteLLM admin UI or directly query the database.

## Security Considerations

### JWKS Caching
- Public keys are cached for 1 hour by default
- Cache automatically refreshes when keys expire
- Failed key fetches result in 503 errors

### Token Validation
- Full signature verification using RSA public keys
- Issuer and audience validation  
- Expiration time checking with configurable leeway
- Malformed tokens are rejected with 401 errors

### Best Practices
1. Use HTTPS for all communication
2. Set reasonable token expiration times (1-24 hours)
3. Rotate signing keys regularly
4. Monitor authentication logs for anomalies
5. Use strong RSA keys (2048-bit minimum)

## Troubleshooting

### Common Issues

#### "JWT settings not configured"
Ensure `jwt_settings` is properly defined in your `config.yaml`

#### "Unable to fetch JWT public keys"
- Check that `public_key_url` is accessible
- Verify JWKS endpoint returns valid JSON
- Ensure network connectivity between LiteLLM and auth provider

#### "Invalid token" errors
- Verify JWT is properly formatted (3 parts separated by dots)
- Check that issuer and audience claims match configuration
- Ensure token hasn't expired
- Verify JWT is signed with the correct private key

#### "Expected JWT token"  
- Ensure Authorization header uses format: `Bearer <jwt-token>`
- Check that token looks like a JWT (has 3 dot-separated parts)

### Debug Mode

Enable verbose logging to debug JWT authentication:

```yaml
litellm_settings:
  set_verbose: True
```

This will log JWT validation steps and any errors encountered.

## Advanced Configuration

### Custom Claim Mappings

You can map any JWT claim to user context:

```yaml
jwt_settings:
  user_claim_mappings:
    user_id: "employee_id"      # Use employee_id instead of sub
    user_email: "work_email"    # Use work_email instead of email
    user_role: "access_level"   # Use access_level instead of role
    team_id: "department"       # Use department as team
```

### Multiple Audiences

To support multiple audiences:

```yaml
jwt_settings:
  audience: ["litellm-proxy", "api-gateway", "mobile-app"]
```

### Custom Algorithm Support

LiteLLM supports various JWT algorithms:

```yaml
jwt_settings:
  algorithm: "RS256"  # RSA SHA-256 (recommended)
  # algorithm: "RS384"  # RSA SHA-384  
  # algorithm: "RS512"  # RSA SHA-512
  # algorithm: "ES256"  # ECDSA SHA-256
```

## Migration Guide

### From API Keys to JWT

If you're migrating from API key authentication:

1. Configure JWT authentication as described above
2. Test with a small subset of users
3. Update client applications to use JWTs instead of API keys
4. Gradually migrate all users
5. Disable API key authentication once migration is complete

### Maintaining Cost History

User costs are tracked by `user_id`. Ensure your JWT `sub` claim matches existing user IDs in your database to maintain cost history continuity. 