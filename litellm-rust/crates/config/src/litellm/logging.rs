use litellm_python_compat::read::one_or_many;

use crate::{Object, Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct LoggingSettings {
    #[serde(deserialize_with = "one_or_many")]
    pub callbacks: Option<Box<[Raw]>>,
    #[serde(deserialize_with = "one_or_many")]
    pub success_callback: Option<Box<[Raw]>>,
    #[serde(deserialize_with = "one_or_many")]
    pub failure_callback: Option<Box<[Raw]>>,
    pub service_callback: Option<Box<[String]>>,
    pub audit_log_callbacks: Option<Box<[String]>>,
    pub s3_callback_params: Option<Object>,
    pub s3_audit_callback_params: Option<Object>,
    pub store_audit_logs: Option<Spelled<bool>>,
    pub json_logs: Option<bool>,
    pub set_verbose: Option<bool>,
    pub turn_off_message_logging: Option<Spelled<bool>>,
    pub redact_messages_in_exceptions: Option<Spelled<bool>>,
    pub log_raw_request_response: Option<Spelled<bool>>,
    pub presidio_logging_only: Option<Spelled<bool>>,
}
