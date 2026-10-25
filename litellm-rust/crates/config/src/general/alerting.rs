use serde::Deserialize;

use crate::{Object, Spelled};

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Alerting {
    List(Box<[String]>),
    One(String),
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AlertingSettings {
    pub alerting: Option<Alerting>,
    pub alerting_threshold: Option<Spelled<f64>>,
    pub alert_types: Option<Box<[String]>>,
    pub alert_to_webhook_url: Option<Object>,
    pub alerting_args: Option<AlertingArgs>,
    pub alert_type_config: Option<Object>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AlertingArgs {
    pub daily_report_frequency: Option<Spelled<i64>>,
    pub report_check_interval: Option<Spelled<i64>>,
    pub budget_alert_ttl: Option<Spelled<i64>>,
    pub outage_alert_ttl: Option<Spelled<i64>>,
    pub region_outage_alert_ttl: Option<Spelled<i64>>,
    pub minor_outage_alert_threshold: Option<Spelled<i64>>,
    pub major_outage_alert_threshold: Option<Spelled<i64>>,
    pub max_outage_alert_list_size: Option<Spelled<i64>>,
    pub log_to_console: Option<Spelled<bool>>,
    pub daily_spend_per_user_threshold: Option<Spelled<f64>>,
    pub monthly_spend_per_user_threshold: Option<Spelled<f64>>,
    pub spend_anomaly_multiplier: Option<Spelled<f64>>,
    pub spend_anomaly_baseline_days: Option<Spelled<i64>>,
    pub spend_anomaly_min_spend: Option<Spelled<f64>>,
    pub user_spend_check_interval: Option<Spelled<i64>>,
}
