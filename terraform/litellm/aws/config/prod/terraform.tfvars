region = "eu-central-1"
azs    = ["eu-central-1a", "eu-central-1b"]

tenant = "data-reply"
env    = "prod"

acm_certificate_domain_name = "litellm.datareply.de"

s3_force_destroy    = false
skip_final_snapshot = false


# ---------- proxy_config (mirrors helm gateway.config.proxy_config) ----------
proxy_config = {
  model_list = [
    {
      model_name = "GPT-5.5"
      litellm_params = {
        model   = "openai/gpt-5.5"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "GPT-5.4"
      litellm_params = {
        model   = "openai/gpt-5.4"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "GPT-5.4-Mini"
      litellm_params = {
        model   = "openai/gpt-5.4-mini"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "GPT-5.3-Codex"
      litellm_params = {
        model   = "openai/gpt-5.3-codex"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
    {
      model_name = "GPT-5.2"
      litellm_params = {
        model   = "openai/gpt-5.2"
        api_key = "os.environ/OPENAI_API_KEY"
      }
    },
  ]
  general_settings = {
    master_key         = "os.environ/LITELLM_MASTER_KEY"
    database_url       = "os.environ/DATABASE_URL"
    alerting           = ["email", "slack_budget_alerts", "slack"]
    alert_types        = ["budget_alerts", "spend_reports"]
    alerting_threshold = 300
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
  }
}

# ---------- Extra env / secrets ----------
gateway_extra_env = {
}

backend_extra_env = {
  SMTP_HOST         = "email-smtp.eu-central-1.amazonaws.com"
  SMTP_TLS          = "True"
  SMTP_PORT         = "587"
  SMTP_SENDER_EMAIL = "fa.siciliano@reply.de"
  PROXY_BASE_URL    = "http://datareply-litellm-dev-1078589364.eu-central-1.elb.amazonaws.com"
}

backend_extra_secrets = {
  "SMTP_USERNAME" = "arn:aws:secretsmanager:eu-central-1:863518425664:secret:datareply-litellm-dev-smtp-username-0kYsv6"
  "SMTP_PASSWORD" = "arn:aws:secretsmanager:eu-central-1:863518425664:secret:datareply-litellm-dev-smtp-password-6Qz5LJ"
}

gateway_extra_secrets = {
  SLACK_WEBHOOK_URL = "arn:aws:secretsmanager:eu-central-1:863518425664:secret:datareply-litellm-dev-slack-alert-webhook-BBpPFN"
  OPENAI_API_KEY    = "arn:aws:secretsmanager:eu-central-1:863518425664:secret:openai-api-key-6JziaK"
}

