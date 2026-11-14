use std::collections::BTreeMap;

use litellm_callbacks::{Abort, AbortError};
use serde_json::{Value, json};

use crate::{
    model::{
        BorrowingLimits, GroupInfo, PRIORITY_CLASS_SHARE_AT_SATURATION,
        SATURATION_FALLBACK_DESCRIPTOR_KEY, Status, TEAM_MODEL_LIMIT_AT_SATURATION,
        TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE, TeamShare,
    },
    pyfmt,
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
            .map(|(key, value)| (key.clone(), pyfmt::value(Some(value))))
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

pub struct Context<'a> {
    pub model: &'a str,
    pub priority: Option<&'a Value>,
    pub team_id: Option<&'a str>,
    pub saturation: f64,
    pub window_size: i64,
}

impl Context<'_> {
    fn priority_or_default(&self) -> Value {
        match self.priority {
            Some(priority) if pyfmt::truthy(Some(priority)) => priority.clone(),
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
        status
            .limit_remaining
            .as_ref()
            .map_or_else(|| "None".into(), pyfmt::number)
    }

    fn saturation_headers(&self, rate_limited_by: Option<&str>) -> Vec<(&'static str, Value)> {
        let mut headers = Vec::new();
        if let Some(reason) = rate_limited_by {
            headers.push(("rate_limited_by", Value::from(reason)));
        }
        headers.push(("x-litellm-priority", self.priority_or_default()));
        headers.push((
            "x-litellm-saturation",
            Value::from(pyfmt::percent(self.saturation, 2)),
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
            pyfmt::value(self.priority),
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
            pyfmt::value(self.priority),
            status.rate_limit_type,
            Self::remaining(status),
            pyfmt::percent(self.saturation, 1),
        );
        let headers = self.headers(status, &self.saturation_headers(None));
        self.refusal(error, status, headers)
    }

    pub fn team_share(&self, status: &Status, limits: &BorrowingLimits) -> Refusal {
        if status.descriptor_key != TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE {
            return self.team_limit(status);
        }
        self.team_borrow_ceiling(status, limits)
    }

    pub fn saturation_fallback(&self, threshold: f64) -> Refusal {
        let error = format!(
            "Model {} is at {} of its capacity, at or above the saturation threshold of {}, trying fallbacks first. Priority: {}",
            self.model,
            pyfmt::percent(self.saturation, 0),
            pyfmt::percent(threshold, 0),
            pyfmt::value(self.priority),
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

    fn team_limit(&self, status: &Status) -> Refusal {
        let name = Self::team_limit_name(status);
        let error = format!(
            "Team {name} reached for {} and the model is saturated, so there is no idle capacity to borrow. Team: {}, Rate limit type: {}, Team {name}: {}, Remaining: {}, Model saturation: {}",
            self.model,
            pyfmt::value(self.team_id.map(Value::from).as_ref()),
            status.rate_limit_type,
            pyfmt::number(&status.current_limit),
            Self::remaining(status),
            pyfmt::percent(self.saturation, 1),
        );
        let headers = self.headers(
            status,
            &self.saturation_headers(Some(TEAM_MODEL_LIMIT_AT_SATURATION)),
        );
        self.refusal(error, status, headers)
    }

    fn team_borrow_ceiling(&self, status: &Status, limits: &BorrowingLimits) -> Refusal {
        let name = Self::team_limit_name(status);
        let error = format!(
            "Team {name} borrow cap reached for {}: the team may use at most {}x its {name} while borrowing idle capacity. Team: {}, Rate limit type: {}, Team {name}: {}, Borrow cap: {}, Remaining: {}, Model saturation: {}",
            self.model,
            limits
                .max_borrow_multiple
                .map_or_else(|| "None".into(), pyfmt::float_repr),
            pyfmt::value(self.team_id.map(Value::from).as_ref()),
            status.rate_limit_type,
            pyfmt::optional_int(limits.share(&status.rate_limit_type)),
            pyfmt::number(&status.current_limit),
            Self::remaining(status),
            pyfmt::percent(self.saturation, 1),
        );
        let headers = self.headers(
            status,
            &self.saturation_headers(Some(TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE)),
        );
        self.refusal(error, status, headers)
    }

    pub fn class_share(&self, status: &Status, share: &TeamShare) -> Refusal {
        let error = format!(
            "Priority class share reached for {} and the model is saturated. Team: {}, Priority: {}, Active teams in class: {}, Active classes: {}, Rate limit type: {}, Team share: {}, Remaining: {}, Model saturation: {}",
            self.model,
            pyfmt::value(self.team_id.map(Value::from).as_ref()),
            pyfmt::value(Some(&self.priority_or_default())),
            share.active_teams_in_class,
            share.active_classes,
            status.rate_limit_type,
            pyfmt::number(&status.current_limit),
            Self::remaining(status),
            pyfmt::percent(self.saturation, 1),
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
