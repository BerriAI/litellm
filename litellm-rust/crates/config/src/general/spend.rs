use crate::{Object, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SpendSettings {
    pub disable_spend_logs: Option<Spelled<bool>>,
    pub disable_spend_updates: Option<Spelled<bool>>,
    pub disable_reset_budget: Option<Spelled<bool>>,
    pub disable_error_logs: Option<Spelled<bool>>,
    pub store_prompts_in_spend_logs: Option<Spelled<bool>>,
    pub disable_requester_ip_address_logging: Option<Spelled<bool>>,
    pub spend_logs_metadata_fields: Option<SpendLogsMetadataFields>,
    pub proxy_batch_write_at: Option<Spelled<f64>>,
    pub proxy_batch_polling_interval: Option<Spelled<f64>>,
    pub proxy_budget_rescheduler_min_time: Option<Spelled<f64>>,
    pub proxy_budget_rescheduler_max_time: Option<Spelled<f64>>,
    pub disable_budget_reservation: Option<Spelled<bool>>,
    pub fail_closed_budget_enforcement: Option<Spelled<bool>>,
    pub apply_user_budget_to_team_keys: Option<Spelled<bool>>,
    pub enforce_fallback_budget: Option<Spelled<bool>>,
    pub track_unmanaged_batch_cost: Option<Spelled<bool>>,
    pub spend_report_frequency: Option<String>,
    pub spend_capture_rate_check: Option<SpendCaptureRateCheckSettings>,
    pub maximum_spend_logs_retention_period: Option<Spelled<i64>>,
    pub maximum_autorouter_session_retention_period: Option<Spelled<i64>>,
    pub maximum_health_check_retention_period: Option<Spelled<i64>>,
    pub maximum_daily_tag_spend_retention_period: Option<Spelled<i64>>,
    pub maximum_spend_logs_size: Option<Spelled<i64>>,
    pub maximum_spend_logs_cleanup_target_size: Option<Spelled<i64>>,
    pub maximum_spend_logs_size_check_interval: Option<String>,
    pub maximum_spend_logs_retention_interval: Option<String>,
    pub maximum_spend_logs_cleanup_cron: Option<String>,
    pub maximum_spend_logs_cleanup_batch_size: Option<Spelled<i64>>,
    pub maximum_spend_logs_cleanup_max_batches: Option<Spelled<i64>>,
    pub maximum_spend_logs_cleanup_run_budget: Option<Spelled<f64>>,
    pub maximum_spend_logs_cleanup_batch_timeout: Option<Spelled<f64>>,
    pub use_spend_logs_partitioning: Option<Spelled<bool>>,
    pub scheduled_job_stagger: Option<ScheduledJobStaggerSettings>,
    pub missing_session_id: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SpendLogsMetadataFields {
    pub include: Option<Box<[String]>>,
    pub exclude: Option<Box<[String]>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct SpendCaptureRateCheckSettings {
    pub providers: Option<Box<[String]>>,
    pub threshold: Option<Spelled<f64>>,
    pub lookback_days: Option<Spelled<i64>>,
    pub openai_project_ids: Option<Box<[String]>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ScheduledJobStaggerSettings {
    pub enabled: Option<Spelled<bool>>,
    pub window_seconds: Option<Spelled<i64>>,
    pub identity: Option<String>,
    pub offsets: Option<Object>,
}
