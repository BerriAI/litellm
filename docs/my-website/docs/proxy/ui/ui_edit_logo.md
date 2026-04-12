import Image from '@theme/IdealImage';

# Customize UI Logo

Personalize your LiteLLM dashboard by replacing the default logo with your own company branding. You can set a custom logo via the UI or the API.

## Via the UI

### 1. Navigate to Settings

Click the **Settings** icon in the sidebar.

![Navigate to Settings](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/57a15404-51f7-481e-9db2-cea94566d3ce/ascreenshot_7a348567c839448bb806fd71cf4abca0_text_export.jpeg)

### 2. Open UI Theme Settings

Click **UI Theme** from the settings menu.

![Open UI Theme](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/30663fe1-9f78-4496-96d4-c53513cbaf82/ascreenshot_ac1eb59eda0e423fbd0e7d3a6cabd4c7_text_export.jpeg)

### 3. Click the Logo URL Field

Click the **Logo URL** text field to start editing.

![Click Logo URL Field](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/069e8412-8ec1-4d36-ba38-6b2e2858a45a/ascreenshot_8fc7fb4a3af74815bc1b69a8554bc110_text_export.jpeg)

### 4. Find Your Logo Image

Open a new browser tab and find the logo image you want to use (e.g., search Google Images for your company logo).

![Find Logo Image](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/d9b55dac-bc4e-4728-b422-4afbc21f9034/ascreenshot_2a805f39c83d4b5e95f43495a6ea4e79_text_export.jpeg)

### 5. Right-Click on the Logo Image

Right-click the image you want to use as your logo.

![Right-Click Image](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/9d42d13e-6028-4710-acb2-c6af04a855c7/ascreenshot_0f21f29ba0e44132afe483a4b88e8b70_text_export.jpeg)

### 6. Copy the Image Address

Select **Copy Image Address** from the context menu to copy the URL.

![Copy Image Address](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/c25637be-383a-498b-ad11-eb1761d52757/ascreenshot_b237ee800979462189a02c1e1942ebf1_text_export.jpeg)

### 7. Switch Back to LiteLLM

Navigate back to the LiteLLM UI tab (e.g., press **Cmd + Left** or click the tab).

![Switch Back](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/f0647856-679c-4591-9ff7-7fd3cfbc70b4/ascreenshot_3ce46dae64c94891ac0983f5ed8f085a_text_export.jpeg)

### 8. Paste the Logo URL

Paste the copied image URL into the **Logo URL** field with **Cmd + V**.

![Paste URL](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/54dd30d9-7a88-41e8-a580-a6acf707c7fa/ascreenshot_8a772218ac0743d9ae8ffd3311eccd5a_text_export.jpeg)

### 9. Save Changes

Click **Save Changes** to apply your new logo.

![Save Changes](https://colony-recorder.s3.amazonaws.com/files/2026-03-13/4baf6494-d146-4600-b6f2-ef667338d580/ascreenshot_722cbcd568ec4267af5122b3958bb248_text_export.jpeg)

Your custom logo will now appear in the LiteLLM dashboard sidebar and login page.

## Via the API

### Set a Custom Logo

```bash
curl -X PATCH 'http://localhost:4000/settings/update/ui_theme_settings' \
  -H 'Authorization: Bearer <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "logo_url": "https://example.com/your-company-logo.png"
  }'
```

### Set a Custom Favicon

You can also customize the browser tab favicon:

```bash
curl -X PATCH 'http://localhost:4000/settings/update/ui_theme_settings' \
  -H 'Authorization: Bearer <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "logo_url": "https://example.com/your-company-logo.png",
    "favicon_url": "https://example.com/your-favicon.ico"
  }'
```

### Get Current Theme Settings

```bash
curl -X GET 'http://localhost:4000/settings/get/ui_theme_settings'
```

### Reset to Default Logo

Send an empty `logo_url` to restore the default LiteLLM logo:

```bash
curl -X PATCH 'http://localhost:4000/settings/update/ui_theme_settings' \
  -H 'Authorization: Bearer <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "logo_url": ""
  }'
```

## Via `proxy_config.yaml`

You can also set the logo URL in your proxy configuration file:

```yaml
litellm_settings:
  ui_theme_config:
    logo_url: "https://example.com/your-company-logo.png"
    favicon_url: "https://example.com/your-favicon.ico"  # optional
```

Or set it as an environment variable:

```yaml
environment_variables:
  UI_LOGO_PATH: "https://example.com/your-company-logo.png"
```

## Supported Logo Formats

| Format | Supported |
|--------|-----------|
| JPEG / JPG | Yes |
| PNG | Yes |
| SVG | Yes |
| ICO (favicon only) | Yes |
| HTTP/HTTPS URL | Yes |
| Local file path | Yes |
