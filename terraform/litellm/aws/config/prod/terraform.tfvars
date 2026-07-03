region = "eu-central-1"
azs    = ["eu-central-1a", "eu-central-1b"]

tenant = "data-reply"
env    = "prod"

acm_certificate_domain_name = "litellm.datareply.de"
route53_zone_id             = "Z046271219OMVB01WCCVM"

s3_force_destroy    = false
skip_final_snapshot = false


# ---------- proxy_config (mirrors helm gateway.config.proxy_config) ----------
proxy_config = {
  model_list = [
    {
      model_name = "gpt-5.5"
      litellm_params = {
        model   = "openai/gpt-5.5"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "gpt-5.4"
      litellm_params = {
        model   = "openai/gpt-5.4"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "gpt-5.4-mini"
      litellm_params = {
        model   = "openai/gpt-5.4-mini"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "gpt-5.3-codex"
      litellm_params = {
        model   = "openai/gpt-5.3-codex"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "gpt-5.2"
      litellm_params = {
        model   = "openai/gpt-5.2"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
  ]
  general_settings = {
    master_key           = "os.environ/LITELLM_MASTER_KEY"
    database_url         = "os.environ/DATABASE_URL"
    alerting             = ["email", "slack_budget_alerts", "slack"]
    alert_types          = ["budget_alerts", "spend_reports"]
    alerting_threshold   = 300
    supported_db_objects = ["mcp"]
    alerting_args = {
      daily_report_frequency       = 43200 # 12 hours in seconds
      report_check_interval        = 3600  # 1 hour in seconds
      budget_alert_ttl             = 86400 # 24 hours in seconds
      outage_alert_ttl             = 60    # 1 minute in seconds
      region_outage_alert_ttl      = 60    # 1 minute in seconds
      minor_outage_alert_threshold = 5
      major_outage_alert_threshold = 10
      max_outage_alert_list_size   = 1000
      log_to_console               = false
    }
  },
  litellm_settings = {
    callbacks = ["smtp_email"]
    mcp_semantic_tool_filter = {

      enabled              = true
      embedding_model      = "text-embedding-3-small"
      top_k                = 5
      similarity_threshold = 0.3
    }
    redact_messages_in_exceptions = true
    ui_theme_config = {
      logo_url    = "https://www.reply.com/favicon.ico"
      favicon_url = "https://www.reply.com/favicon.ico"
    }
  },
  mcp_servers = {
    data_reply_sharepoint_server = {
      url           = ""
      transport     = "http"
      auth_type     = "oauth2"
      client_id     = "os.environ/SHAREPOINT_OAUTH_CREDENTIALS_CLIENT_ID"
      client_secret = "os.environ/SHAREPOINT_OAUTH_CREDENTIALS_CLIENT_SECRET"
    }
  }
}

# ---------- Extra env / secrets ----------
gateway_extra_env = {}

backend_extra_env = {
  SMTP_HOST         = "email-smtp.eu-central-1.amazonaws.com"
  SMTP_TLS          = "True"
  SMTP_PORT         = "587"
  SMTP_SENDER_EMAIL = "data.awsacccounts.management@reply.de"
  PROXY_BASE_URL    = "https://litellm.datareply.de"
  STORE_MODEL_IN_DB = true
  DISABLE_ADMIN_UI  = false
}

backend_extra_secrets = {
  SMTP_USERNAME = "arn:aws:secretsmanager:eu-central-1:751812493785:secret:data-reply/litellm/smtp/username-gg6WZa"
  SMTP_PASSWORD = "arn:aws:secretsmanager:eu-central-1:751812493785:secret:data-reply/litellm/smtp/password-QplKSB"
}

gateway_extra_secrets = {
  SLACK_WEBHOOK_URL = "arn:aws:secretsmanager:eu-central-1:751812493785:secret:data-reply/litellm/slack/webhook/reports-ELuU2k"
  OPENAI_API_KEY    = "arn:aws:secretsmanager:eu-central-1:751812493785:secret:data-reply/litellm/openai/api-key-tYAO39"
}

