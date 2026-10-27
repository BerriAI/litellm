INSERT INTO "LiteLLM_UISettings" (id, ui_settings, updated_at)
VALUES ('user_banner', $1, CURRENT_TIMESTAMP)
ON CONFLICT (id) DO UPDATE
SET ui_settings = EXCLUDED.ui_settings,
    updated_at = EXCLUDED.updated_at
