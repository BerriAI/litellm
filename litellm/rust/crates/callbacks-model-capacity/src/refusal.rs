use std::collections::BTreeMap;

use litellm_callbacks::{Abort, AbortError};
use litellm_python_compat::{
    format::percent, json::loaded_str, repr::float_repr, truthy::json_truthy,
};
use serde_json::{Value, json};

use crate::{
    error::CapacityError,
    model::{
        BorrowingLimits, GroupInfo, PRIORITY_CLASS_SHARE_AT_SATURATION,
        SATURATION_FALLBACK_DESCRIPTOR_KEY, Status, TeamShare,
    },
};

pub const RATE_LIMIT_ERROR: &str = "model_capacity_rate_limit";
pub const RERAISE: &str = "model_capacity_reraise";

#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub detail: Vec<(String, Value)>,
    pub headers: Vec<(String, Value)>,
    pub rate_limit_type: Option<String>,
    pub model: String,
    pub descriptor_key: Option<String>,
}

impl Refusal {
    pub fn header(&self, name: &str) -> Option<&Value> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }

    pub fn into_abort(self) -> Abort {
        let message = self
            .detail
            .iter()
            .find(|(key, _)| key == "error")
            .and_then(|(_, value)| value.as_str())
            .unwrap_or_default()
            .to_owned();
        let headers: BTreeMap<String, String> = self
            .headers
            .iter()
            .map(|(key, value)| (key.clone(), loaded_str(value)))
            .collect();
        let fields = json!({
            "detail": self.detail,
            "headers": self.headers,
            "rate_limit_type": self.rate_limit_type,
            "model": self.model,
            "descriptor_key": self.descriptor_key,
        });
        Abort {
            headers,
            error: Some(AbortError {
                kind: RATE_LIMIT_ERROR.into(),
                fields: fields.as_object().cloned().unwrap_or_default(),
            }),
            ..Abort::new(429, message)
        }
    }
}

pub fn reraise(token: u64, message: &str) -> Abort {
    Abort {
        error: Some(AbortError {
            kind: RERAISE.into(),
            fields: json!({"token": token})
                .as_object()
                .cloned()
                .unwrap_or_default(),
        }),
        ..Abort::new(500, message)
    }
}

pub fn unverifiable() -> Abort {
    Abort {
        error: Some(AbortError {
            kind: RERAISE.into(),
            fields: json!({"unverifiable": true})
                .as_object()
                .cloned()
                .unwrap_or_default(),
        }),
        ..Abort::new(503, CapacityError::Unverifiable.to_string())
    }
}

pub struct Context<'a> {
    pub model: &'a str,
    pub priority: Option<&'a Value>,
    pub saturation: f64,
    pub window_size: i64,
}

impl Context<'_> {
    fn priority_or_default(&self) -> Value {
        match self.priority {
            Some(priority) if json_truthy(priority) => priority.clone(),
            _ => Value::from("default"),
        }
    }

    fn headers(&self, status: &Status, extra: &[(&str, Value)]) -> Vec<(String, Value)> {
        let mut headers = vec![
            (
                "retry-after".to_owned(),
                Value::from(self.window_size.to_string()),
            ),
            (
                "rate_limit_type".to_owned(),
                Value::from(status.rate_limit_type.clone()),
            ),
        ];
        headers.extend(
            extra
                .iter()
                .map(|(key, value)| ((*key).to_owned(), value.clone())),
        );
        headers
    }

    fn refusal(&self, error: String, status: &Status, headers: Vec<(String, Value)>) -> Refusal {
        Refusal {
            detail: vec![("error".into(), Value::String(error))],
            headers,
            rate_limit_type: Some(status.rate_limit_type.clone()),
            model: self.model.to_owned(),
            descriptor_key: None,
        }
    }

    fn remaining(status: &Status) -> String {
        status.limit_remaining.as_ref().map_or_else(
            || "None".into(),
            |limit| loaded_str(&Value::Number(limit.clone())),
        )
    }

    fn saturation_headers(&self, rate_limited_by: Option<&str>) -> Vec<(&'static str, Value)> {
        let mut headers = Vec::new();
        if let Some(reason) = rate_limited_by {
            headers.push(("rate_limited_by", Value::from(reason)));
        }
        headers.push(("x-litellm-priority", self.priority_or_default()));
        headers.push((
            "x-litellm-saturation",
            Value::from(percent(self.saturation, 2)),
        ));
        headers
    }

    fn group_limits(group: &GroupInfo) -> (String, String) {
        let configured = |limit: Option<i64>| {
            limit.map_or_else(|| "not configured".to_owned(), |limit| limit.to_string())
        };
        (configured(group.tpm), configured(group.rpm))
    }

    pub fn model_capacity(&self, status: &Status, group: &GroupInfo) -> Refusal {
        let (tpm, rpm) = Self::group_limits(group);
        let error = format!(
            "Model capacity reached for {}. Priority: {}, Rate limit type: {}, Model TPM: {tpm}, Model RPM: {rpm}, Remaining: {}",
            self.model,
            loaded_str(self.priority.unwrap_or(&Value::Null)),
            status.rate_limit_type,
            Self::remaining(status),
        );
        let headers = self.headers(
            status,
            &[("x-litellm-priority", self.priority_or_default())],
        );
        self.refusal(error, status, headers)
    }

    pub fn priority_pool(&self, status: &Status, group: &GroupInfo) -> Refusal {
        let (tpm, rpm) = Self::group_limits(group);
        let error = format!(
            "Priority-based rate limit exceeded. Model: {}, Priority: {}, Rate limit type: {}, Model TPM: {tpm}, Model RPM: {rpm}, Remaining: {}, Model saturation: {}",
            self.model,
            loaded_str(self.priority.unwrap_or(&Value::Null)),
            status.rate_limit_type,
            Self::remaining(status),
            percent(self.saturation, 1),
        );
        let headers = self.headers(status, &self.saturation_headers(None));
        self.refusal(error, status, headers)
    }

    pub fn team_share(&self, status: &Status, borrower: &BorrowingLimits) -> Refusal {
        if status.descriptor_key != borrower.kind.max_borrow_multiple() {
            return self.team_limit(status, borrower);
        }
        self.team_borrow_ceiling(status, borrower)
    }

    pub fn saturation_fallback(&self, threshold: f64) -> Refusal {
        let error = format!(
            "Model {} is at {} of its capacity, at or above the saturation threshold of {}, trying fallbacks first. Priority: {}",
            self.model,
            percent(self.saturation, 0),
            percent(threshold, 0),
            loaded_str(self.priority.unwrap_or(&Value::Null)),
        );
        Refusal {
            detail: vec![("error".into(), Value::String(error))],
            headers: vec![
                (
                    "retry-after".into(),
                    Value::from(self.window_size.to_string()),
                ),
                ("x-litellm-priority".into(), self.priority_or_default()),
            ],
            rate_limit_type: None,
            model: self.model.to_owned(),
            descriptor_key: Some(SATURATION_FALLBACK_DESCRIPTOR_KEY.into()),
        }
    }

    fn team_limit_name(status: &Status) -> &'static str {
        if status.rate_limit_type == "requests" {
            "model_rpm_limit"
        } else {
            "model_tpm_limit"
        }
    }

    fn team_limit(&self, status: &Status, borrower: &BorrowingLimits) -> Refusal {
        let name = Self::team_limit_name(status);
        let label = borrower.kind.label();
        let error = format!(
            "{label} {name} reached for {} and the model is saturated, so there is no idle capacity to borrow. {label}: {}, Rate limit type: {}, {label} {name}: {}, Remaining: {}, Model saturation: {}",
            self.model,
            borrower.entity_id,
            status.rate_limit_type,
            loaded_str(&Value::Number(status.current_limit.clone())),
            Self::remaining(status),
            percent(self.saturation, 1),
        );
        let reason = borrower.kind.at_saturation();
        let headers = self.headers(status, &self.saturation_headers(Some(&reason)));
        self.refusal(error, status, headers)
    }

    fn team_borrow_ceiling(&self, status: &Status, borrower: &BorrowingLimits) -> Refusal {
        let name = Self::team_limit_name(status);
        let label = borrower.kind.label();
        let error = format!(
            "{label} {name} borrow cap reached for {}: the {} may use at most {}x its {name} while borrowing idle capacity. {label}: {}, Rate limit type: {}, {label} {name}: {}, Borrow cap: {}, Remaining: {}, Model saturation: {}",
            self.model,
            label.to_lowercase(),
            borrower
                .max_borrow_multiple
                .map_or_else(|| "None".into(), float_repr),
            borrower.entity_id,
            status.rate_limit_type,
            borrower
                .share(&status.rate_limit_type)
                .map_or_else(|| "None".into(), |share| share.to_string()),
            loaded_str(&Value::Number(status.current_limit.clone())),
            Self::remaining(status),
            percent(self.saturation, 1),
        );
        let reason = borrower.kind.max_borrow_multiple();
        let headers = self.headers(status, &self.saturation_headers(Some(&reason)));
        self.refusal(error, status, headers)
    }

    /// `holder` is the label and id of whose slice refused the request.
    pub fn class_share(&self, status: &Status, share: &TeamShare, holder: (&str, &str)) -> Refusal {
        let (label, holder_id) = holder;
        let error = format!(
            "Priority class share reached for {} and the model is saturated. {label}: {holder_id}, Priority: {}, Active callers in class: {}, Active classes: {}, Rate limit type: {}, Share: {}, Remaining: {}, Model saturation: {}",
            self.model,
            loaded_str(&self.priority_or_default()),
            share.active_teams_in_class,
            share.active_classes,
            status.rate_limit_type,
            loaded_str(&Value::Number(status.current_limit.clone())),
            Self::remaining(status),
            percent(self.saturation, 1),
        );
        let headers = self.headers(
            status,
            &self.saturation_headers(Some(PRIORITY_CLASS_SHARE_AT_SATURATION)),
        );
        self.refusal(error, status, headers)
    }

    pub fn unknown(&self, offending: Option<&Status>) -> Refusal {
        let descriptor_key = offending.map_or("unknown", |status| status.descriptor_key.as_str());
        let rate_limit_type = offending.map_or("unknown", |status| status.rate_limit_type.as_str());
        Refusal {
            detail: vec![
                ("error".into(), Value::from("Rate limit exceeded")),
                ("descriptor_key".into(), Value::from(descriptor_key)),
                ("rate_limit_type".into(), Value::from(rate_limit_type)),
            ],
            headers: vec![
                (
                    "retry-after".into(),
                    Value::from(self.window_size.to_string()),
                ),
                ("x-litellm-priority".into(), self.priority_or_default()),
            ],
            rate_limit_type: offending.map(|status| status.rate_limit_type.clone()),
            model: self.model.to_owned(),
            descriptor_key: None,
        }
    }
}
