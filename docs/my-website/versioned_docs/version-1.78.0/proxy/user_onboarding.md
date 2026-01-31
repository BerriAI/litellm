# User Onboarding Guide

A step-by-step guide to help admins onboard users to your LiteLLM proxy instance and help users get started with their API key.

---

## For Administrators

### Step 1: Create a User Account

You can create a user account via the Admin UI or using the API.

#### Admin UI
- Go to the (`/ui` endpoint)
- Navigate to the Internal Users section
- Click "Add User" and fill in the required details

#### API
```bash
curl -X POST http://localhost:4000/user/new \
  -H "Authorization: Bearer <admin-key>" \
  -H "Content-Type: application/json" \
  -d '{"user_email": "user@example.com"}'
```

---

### Step 2: Grant Access & Permissions

- Assign the user to a team (optional)
- Set budgets, rate limits, and allowed models as needed
- Generate an API key for the user (via UI or API)

#### **Generate API Key (API Example)**
```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer <admin-key>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<user-id>", "max_budget": 100}'
```

---

## For End Users

### Step 3: Validate Your API Key

Before making LLM calls, validate your key works by calling the `/v1/models` endpoint:

```bash
curl -X GET http://localhost:4000/v1/models \
  -H "Authorization: Bearer <your-api-key>"
```
- If your key is valid, you'll get a list of available models.
- If invalid, you'll get a 401 error.

---

### Step 4: Hello World - Make Your First LLM Call

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Troubleshooting
- If you get a 401 error, check with your admin that your key is active and you have access to the requested model.
- Use the `/v1/models` endpoint to quickly check if your key is valid without consuming LLM tokens.

---

## See Also
- [Proxy Quick Start](./quick_start.md)
- [User Management](./users.md)
- [Key Management](./key_management.md)
