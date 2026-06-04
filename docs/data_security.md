# Data Privacy and Security

At LiteLLM, **safeguarding your data privacy and security** is our top priority. We recognize the critical importance of the data you share with us and handle it with the highest level of diligence.

## Security Measures

### Self-hosted Instances LiteLLM

- **No data or telemetry is stored on LiteLLM Servers when you self-host**
- For installation and configuration, see: [Self-hosting guide](../docs/proxy/deploy.md)
- **Telemetry**: We run no telemetry when you self-host LiteLLM

For security inquiries, please contact us at support@berri.ai

## **Security Certifications**

| **Certification** | **Status**                                                                                      |
|-------------------|-------------------------------------------------------------------------------------------------|
| SOC 2 Type I      | Certified. Report available upon request on Enterprise plan.                                                           |
| SOC 2 Type II     | Certified. Report available upon request on Enterprise plan.                   |
| ISO 27001          | Certified. Report available upon request on Enterprise                              |


## Collection of Personal Data

### For Self-hosted LiteLLM Users:
- No personal data is collected or transmitted to LiteLLM servers when you self-host our software.
- Any data generated or processed remains entirely within your own infrastructure.

## Cookies Information, Security, and Privacy

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

## Security Vulnerability Reporting Guidelines

We value the security community's role in protecting our systems and users. To report a security vulnerability:

- Email support@berri.ai with details
- Include steps to reproduce the issue
- Provide any relevant additional information

We'll review all reports promptly. Note that we don't currently offer a bug bounty program.

## Vulnerability Scanning

- LiteLLM runs [`grype`](https://github.com/anchore/grype) security scans on all built Docker images.
    - See [`grype litellm` check on ci/cd](https://github.com/BerriAI/litellm/blob/main/.circleci/config.yml#L1099). 
    - Current Status: ✅ Passing. 0 High/Critical severity vulnerabilities found.

## Legal/Compliance FAQs

### Procurement Options

1. Invoicing
2. AWS Marketplace
3. Azure Marketplace


### Vendor Information

Legal Entity Name: Berrie AI Incorporated

Point of contact email address for security incidents: krrish@berri.ai

Point of contact email address for general security-related questions: krrish@berri.ai 

Has the Vendor been audited / certified? 
- SOC 2 Type I. Certified. Report available upon request on Enterprise plan.
- SOC 2 Type II. In progress. Certificate available by April 15th, 2025.
- ISO 27001. Certified. Report available upon request on Enterprise plan.

Has an information security management system been implemented? 
- Yes - [CodeQL](https://codeql.github.com/) and a comprehensive ISMS covering multiple security domains.

Is logging of key events - auth, creation, update changes occurring? 
- Yes - we have [audit logs](https://docs.litellm.ai/docs/proxy/multiple_admins#1-switch-on-audit-logs)

Does the Vendor have an established Cybersecurity incident management program? 
- Yes, Incident Response Policy available upon request.


Does the vendor have a vulnerability disclosure policy in place? [Yes](https://github.com/BerriAI/litellm?tab=security-ov-file#security-vulnerability-reporting-guidelines)

Does the vendor perform vulnerability scans? 
- Yes, regular vulnerability scans are conducted as detailed in the [Vulnerability Scanning](#vulnerability-scanning) section.

Signer Name: Krish Amit Dholakia

Signer Email: krrish@berri.ai