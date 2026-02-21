# UI Troubleshooting

If you're experiencing issues with the LiteLLM Admin UI, please include the following information when reporting.

## 1. Steps to Reproduce

A clear, step-by-step description of how to trigger the issue (e.g., "Navigate to Settings → Team, click 'Create Team', fill in fields, click submit → error appears").

## 2. LiteLLM Version

The current version of LiteLLM you're running. Check via `litellm --version` or the UI's settings page.

## 3. Architecture & Deployment Setup

Distributed environments are a known source of UI issues. Please describe:

- **Number of LiteLLM instances/replicas** and how they are deployed (e.g., Kubernetes, Docker Compose, ECS)
- **Load balancer** type and configuration (e.g., ALB, Nginx, Cloudflare Tunnel) — include whether sticky sessions are enabled
- **How the UI is accessed** — directly via LiteLLM, through a reverse proxy, or behind an ingress controller
- **Any CDN or caching layers** between the user and the LiteLLM server

## 4. Network Tab Requests

Open your browser's Developer Tools (F12 → Network tab), reproduce the issue, and share:

- The **failing request(s)** — URL, method, status code, and response body
- **Screenshots or HAR export** of the relevant network activity
- Any **CORS or mixed-content errors** shown in the Console tab

## 5. Environment Variables

Non-sensitive environment variables related to the UI and proxy setup, such as:

- `LITELLM_MASTER_KEY`
- `PROXY_BASE_URL` / `LITELLM_PROXY_BASE_URL`
- `UI_BASE_PATH`
- Any SSO-related variables (e.g., `GOOGLE_CLIENT_ID`, `MICROSOFT_TENANT`)

Do **not** include passwords, secrets, or API keys.

## 6. Browser & Access Details

- **Browser** and version (e.g., Chrome 120, Firefox 121)
- **Access URL** used to reach the UI (redact sensitive parts)
- Whether the issue occurs for **all users or specific roles** (Admin, Internal User, etc.)

## 7. Screenshots or Screen Recordings

A screenshot or short screen recording of the issue is extremely helpful. Include any visible error messages, toasts, or unexpected behavior.
