# Data Privacy and Security

## Security Measures

### LiteLLM Cloud

- We encrypt all data stored using your `LITELLM_MASTER_KEY` and in transit using TLS.
- Our database and application run on GCP, AWS infrastructure, partly managed by NeonDB.
    - US data region: Northern California (AWS/GCP `us-west-1`) & Virginia (AWS `us-east-1`)
    - EU data region Germany/Frankfurt (AWS/GCP `eu-central-1`)
- All users have access to SSO (Single Sign-On) through OAuth 2.0 with Google, Okta, Microsoft, KeyCloak. 
- Audit Logs with retention policy
- Control Allowed IP Addresses that can access your Cloud LiteLLM Instance

For security inquiries, please contact us at support@berri.ai

### Self-hosted Instances LiteLLM

- ** No data or telemetry is stored on LiteLLM Servers when you self host **
- For installation and configuration, see: [Self-hosting guided](../docs/proxy/deploy.md)
- **Telemetry** We run no telemetry when you self host LiteLLM

For security inquiries, please contact us at support@berri.ai

## Supported data regions for LiteLLM Cloud

LiteLLM supports the following data regions:

- US, Northern California (AWS/GCP `us-west-1`)
- Europe, Frankfurt, Germany (AWS/GCP `eu-central-1`)

All data, user accounts, and infrastructure are completely separated between these two regions

## Collection of personal data

### For Self-hosted LiteLLM Users:
- No personal data is collected or transmitted to LiteLLM servers when you self-host our software.
- Any data generated or processed remains entirely within your own infrastructure.

### For LiteLLM Cloud Users:
- LiteLLM Cloud tracks LLM usage data - We do not access or store the message / response content of your API requests or responses. You can see the [fields tracked here](https://github.com/BerriAI/litellm/blob/main/schema.prisma#L174)

**How to use and share the personal data**
- Only proxy admins can view their usage data, and they can only see the usage data of their organization.
- Proxy admins have the ability to invite other users / admins to their server to view their own usage data
- LiteLLM Cloud does not sell or share any usage data with any third parties.

## Cookies information, security and privacy

### For Self-hosted LiteLLM Users:
- Cookie data remains within your own infrastructure.
- LiteLLM uses minimal cookies, solely for the purpose of allowing Proxy users to access the LiteLLM Admin UI.
- These cookies are stored in your web browser after you log in.
- We do not use cookies for advertising, tracking, or any purpose beyond maintaining your login session.
- The only cookies used are essential for maintaining user authentication and session management for the app UI.
- Session cookies expire when you close your browser, logout or after 24 hours.
- LiteLLM does not use any third-party cookies.
- The Admin UI accesses the cookie to authenticate your login session.
- The cookie is stored as JWT and is not accessible to any other part of the system.
- We (LiteLLM) do not access or share this cookie data for any other purpose.


### For LiteLLM Cloud Users:
- LiteLLM uses minimal cookies, solely for the purpose of allowing Proxy users to access the LiteLLM Admin UI.
- These cookies are stored in your web browser after you log in.
- We do not use cookies for advertising, tracking, or any purpose beyond maintaining your login session.
- The only cookies used are essential for maintaining user authentication and session management for the app UI.
- Session cookies expire when you close your browser, logout or after 24 hours.
- LiteLLM does not use any third-party cookies.
- The Admin UI accesses the cookie to authenticate your login session.
- The cookie is stored as JWT and is not accessible to any other part of the system.
- We (LiteLLM) do not access or share this cookie data for any other purpose.

## Security Vulnerability Reporting Guidelines

We value the security community's role in protecting our systems and users. To report a security vulnerability:

- Email support@berri.ai with details
- Include steps to reproduce the issue
- Provide any relevant additional information

We'll review all reports promptly. Note that we don't currently offer a bug bounty program.

## Legal/Compliance FAQs

### Procurement Options

1. Invoicing
2. AWS Marketplace
3. Azure Marketplace

### Vendor Information

Legal Entity Name: Berrie AI Incorporated

Company Phone Number: 7708783106 

Number of employees in the company: 2

Number of employees in security team: 2

Point of contact email address for security incidents: krrish@berri.ai

Point of contact email address for general security-related questions: krrish@berri.ai 

Has the Vendor been audited / certified? Currently undergoing SOC-2 Certification from Drata 

Has an information security management system been implemented? Yes - [CodeQL](https://codeql.github.com/)

Is logging of key events - auth, creation, update changes occurring? Yes - we have [audit logs](https://docs.litellm.ai/docs/proxy/multiple_admins#1-switch-on-audit-logs)

Does the Vendor have an established Cybersecurity incident management program? No 

Not applicable - LiteLLM is self-hosted, this is the responsibility of the team hosting the proxy. We do provide [alerting](https://docs.litellm.ai/docs/proxy/alerting) and [monitoring](https://docs.litellm.ai/docs/proxy/prometheus) tools to help with this. 

Does the vendor have a vulnerability disclosure policy in place? [Yes](https://github.com/BerriAI/litellm?tab=security-ov-file#security-vulnerability-reporting-guidelines)

Does the vendor perform vulnerability scans? No 

Signer Name: Krish Amit Dholakia

Signer Email: krrish@berri.ai

