"""
Has all /sso/* routes

/sso/key/generate - handles user signing in with SSO and redirects to /sso/callback
/sso/callback - returns JWT Redirect Response that redirects to LiteLLM UI

/sso/debug/login - handles user signing in with SSO and redirects to /sso/debug/callback
/sso/debug/callback - returns the OpenID object returned by the SSO provider
"""
