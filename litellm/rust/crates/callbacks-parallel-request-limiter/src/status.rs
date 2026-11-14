use litellm_redis_store::rate_limits::Counter;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Code {
    #[serde(rename = "OK")]
    Ok,
    #[serde(rename = "OVER_LIMIT")]
    OverLimit,
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum LimitKind {
    #[serde(rename = "requests")]
    #[strum(serialize = "requests")]
    Requests,
    #[serde(rename = "tokens")]
    #[strum(serialize = "tokens")]
    Tokens,
    #[serde(rename = "max_parallel_requests")]
    #[strum(serialize = "max_parallel_requests")]
    MaxParallelRequests,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Status {
    pub code: Code,
    pub current_limit: i64,
    pub limit_remaining: i64,
    pub rate_limit_type: LimitKind,
    pub descriptor_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub descriptor_value: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Redis,
    Local,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowIdentity {
    pub counter: Counter,
    pub window_start: String,
    pub backend: Backend,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    pub overall: Code,
    pub statuses: Vec<Status>,
    pub windows: Option<Vec<WindowIdentity>>,
}

impl Response {
    pub fn ok(statuses: Vec<Status>) -> Self {
        Self {
            overall: Code::Ok,
            statuses,
            windows: None,
        }
    }

    pub fn over(status: Status) -> Self {
        Self {
            overall: Code::OverLimit,
            statuses: vec![status],
            windows: None,
        }
    }

    pub fn is_over(&self) -> bool {
        self.overall == Code::OverLimit
    }

    pub fn windows(&self) -> &[WindowIdentity] {
        self.windows.as_deref().unwrap_or_default()
    }
}

pub fn identities(windows: impl IntoIterator<Item = WindowIdentity>) -> Vec<WindowIdentity> {
    windows.into_iter().fold(Vec::new(), |mut unique, window| {
        if !unique.contains(&window) {
            unique.push(window);
        }
        unique
    })
}

pub fn header_map(statuses: &[Status]) -> Map<String, Value> {
    statuses.iter().fold(Map::new(), |mut merged, status| {
        let prefix = format!("x-ratelimit-{}", status.descriptor_key);
        let kind: &str = status.rate_limit_type.into();
        merged.insert(
            format!("{prefix}-remaining-{kind}"),
            Value::from(status.limit_remaining),
        );
        merged.insert(
            format!("{prefix}-limit-{kind}"),
            Value::from(status.current_limit),
        );
        merged
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn status(key: &str, kind: LimitKind, remaining: i64, limit: i64) -> Status {
        Status {
            code: Code::Ok,
            current_limit: limit,
            limit_remaining: remaining,
            rate_limit_type: kind,
            descriptor_key: key.into(),
            descriptor_value: None,
        }
    }

    #[rstest]
    fn a_later_status_for_the_same_descriptor_and_kind_wins_like_the_dict_update() {
        let headers = header_map(&[
            status("api_key", LimitKind::Requests, 4, 5),
            status("api_key", LimitKind::Tokens, 90, 100),
            status("api_key", LimitKind::Requests, 2, 5),
        ]);
        assert_eq!(
            Value::Object(headers),
            json!({
                "x-ratelimit-api_key-remaining-requests": 2,
                "x-ratelimit-api_key-limit-requests": 5,
                "x-ratelimit-api_key-remaining-tokens": 90,
                "x-ratelimit-api_key-limit-tokens": 100,
            })
        );
    }

    #[rstest]
    fn a_status_serialises_the_python_typed_dict_shape() {
        let mut with_value = status("tag", LimitKind::MaxParallelRequests, 0, 1);
        with_value.descriptor_value = Some("t".into());
        assert_eq!(
            serde_json::to_value(&with_value).unwrap(),
            json!({"code": "OK", "current_limit": 1, "limit_remaining": 0,
                   "rate_limit_type": "max_parallel_requests", "descriptor_key": "tag", "descriptor_value": "t"})
        );
    }
}
