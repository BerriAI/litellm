# SAML 2.0 with LiteLLM

:::info
SSO is free for 5 users, after this an Enterprise License is required.
:::

## Overview

SAML 2.0 SSO allows your users to authenticate with LiteLLM using your organization's identity provider (IdP) such as Okta, Azure AD, OneLogin, or any SAML 2.0 compliant provider.

## Configuration

### Required Environment Variables

```bash
# SAML Identity Provider Configuration
SAML_IDP_ENTITY_ID="https://your-idp.example.com"
SAML_IDP_SSO_URL="https://your-idp.example.com/sso/saml"
SAML_IDP_X509_CERT="MIIDqjCCAp...."

# LiteLLM configuration
PROXY_BASE_URL="https://proxy-base-url.com"
LITELLM_MASTER_KEY="sk-1234"
LITELLM_SALT_KEY="a-secure-hash" # this is to store credentials in your postgres db
DATABASE_URL="postgresql://..."

# [if setting up as an admin] SAML uses user_email here. find this in your dashboard upper right panel
PROXY_ADMIN_ID="ishaan@berri.ai"

# Enterprise License (for more than 5 users)
LITELLM_LICENSE="your-license-key"
```

### Optional Environment Variables

```bash
SAML_ENTITY_ID="litellm-proxy"  # find this here: {PROXY_BASE_URL}/sso/saml/metadata
SAML_ACS_PATH="/sso/saml/acs"  # Default: /sso/saml/acs (can be changed to /saml/acs if needed)
SAML_NAME_ID_FORMAT="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"  # Default

# SAML Attribute Mapping (customize based on IdP)

SAML_USER_ID_ATTRIBUTE="email"  # Default
SAML_USER_EMAIL_ATTRIBUTE="email"  # Default
SAML_USER_FIRST_NAME_ATTRIBUTE="firstName"  # Default
SAML_USER_LAST_NAME_ATTRIBUTE="lastName"  # Default
SAML_USER_DISPLAY_NAME_ATTRIBUTE="displayName"  # Optional

# Optional: Single Logout Url (should be set to proxy-base-url)
SAML_IDP_SLO_URL="https://your-idp.example.com/slo/saml"

```

Thats it! Once you configure these environment variables, head over to https://proxy-base-url and you'll be logged in via SAML 2.0