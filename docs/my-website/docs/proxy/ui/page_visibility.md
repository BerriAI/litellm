import Image from '@theme/IdealImage';

# Control Page Visibility for Internal Users

Configure which navigation tabs and pages are visible to internal users (non-admin developers) in the LiteLLM UI.

Use this feature to simplify the UI and control which pages your internal users/developers can see when signing in.

## Overview

By default, all pages accessible to internal users are visible in the navigation sidebar. The page visibility control allows admins to restrict which pages internal users can see, creating a more focused and streamlined experience.


## Configure Page Visibility

### 1. Navigate to Settings

Click the **Settings** icon in the sidebar.

![Navigate to Settings](https://colony-recorder.s3.amazonaws.com/files/2026-01-28/cbb6f272-ab18-4996-b57d-7ed4aad721ea/ascreenshot_ab80f3175b1a41b0bdabdd2cd3980573_text_export.jpeg)

### 2. Go to Admin Settings

Click **Admin Settings** from the settings menu.

![Go to Admin Settings](https://colony-recorder.s3.amazonaws.com/files/2026-01-28/e2b327bf-1cfd-4519-a9ce-8a6ecb2de53a/ascreenshot_23bb1577b3f84d22be78e0faa58dee3d_text_export.jpeg)

### 3. Select UI Settings

Click **UI Settings** to access the page visibility controls.

![Select UI Settings](https://colony-recorder.s3.amazonaws.com/files/2026-01-28/fff0366a-4944-457a-8f6a-e22018dde108/ascreenshot_0e268e8651654e75bb9fb40d2ed366a9_text_export.jpeg)

### 4. Open Page Visibility Configuration

Click **Configure Page Visibility** to expand the configuration panel.

![Open Configuration](https://colony-recorder.s3.amazonaws.com/files/2026-01-28/3a4761d6-145a-4afd-8abf-d92744b9ac9f/ascreenshot_23c16eb79c32481887b879d961f1f00a_text_export.jpeg)

### 5. Select Pages to Make Visible

Check the boxes for the pages you want internal users to see. Pages are organized by category for easy navigation.

![Select Pages](https://colony-recorder.s3.amazonaws.com/files/2026-01-28/b9c96b54-6c20-484f-8b0b-3a86decb5717/ascreenshot_3347ade01ebe4ea390bc7b57e53db43f_text_export.jpeg)

**Available pages include:**
- Virtual Keys
- Playground
- Models + Endpoints
- Agents
- MCP Servers
- Search Tools
- Vector Stores
- Logs
- Teams
- Organizations
- Usage
- Budgets
- And more...

### 6. Save Your Configuration

Click **Save Page Visibility Settings** to apply the changes.

![Save Settings](https://colony-recorder.s3.amazonaws.com/files/2026-01-28/8a215378-44f5-4bb8-b984-06fa2aa03903/ascreenshot_44e7aeebe25a477ba92f73a3ed3df644_text_export.jpeg)

### 7. Verify Changes

Internal users will now only see the selected pages in their navigation sidebar.

![Verify Changes](https://colony-recorder.s3.amazonaws.com/files/2026-01-28/493a7718-b276-40b9-970f-5814054932d9/ascreenshot_ad23b8691f824095ba60256f91ad24f8_text_export.jpeg)

## Reset to Default

To restore all pages to internal users:

1. Open the Page Visibility configuration
2. Click **Reset to Default (All Pages)**
3. Click **Save Page Visibility Settings**

This will clear the restriction and show all accessible pages to internal users.

## API Configuration

You can also configure page visibility programmatically using the API:

### Get Current Settings

```bash
curl -X GET 'http://localhost:4000/ui_settings/get' \
  -H 'Authorization: Bearer <your-admin-key>'
```

### Update Page Visibility

```bash
curl -X PATCH 'http://localhost:4000/ui_settings/update' \
  -H 'Authorization: Bearer <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "enabled_ui_pages_internal_users": [
      "api-keys",
      "agents",
      "mcp-servers",
      "logs",
      "teams"
    ]
  }'
```

### Clear Page Visibility Restrictions

```bash
curl -X PATCH 'http://localhost:4000/ui_settings/update' \
  -H 'Authorization: Bearer <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "enabled_ui_pages_internal_users": null
  }'
```

