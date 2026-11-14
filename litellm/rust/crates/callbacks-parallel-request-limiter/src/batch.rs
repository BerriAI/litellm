use std::{collections::BTreeMap, time::Duration};

use litellm_redis_store::{
    Number,
    rate_limits::{Counter, Increment, Kind, Local},
};
use serde::{Deserialize, Serialize};

use crate::{
    descriptor::{Descriptor, PROJECT_ITPM, PROJECT_OTPM},
    engine::{GuardedOp, Op},
    enqueued::EnqueuedScope,
    status::{LimitKind, Status, WindowIdentity},
};

pub const TPD_SUFFIX: &str = "_tpd";
pub const TPD_WINDOW_SECONDS: u64 = 86_400;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchFacts {
    pub descriptors: Vec<Descriptor>,
    pub increments: Vec<(i64, i64)>,
    pub request_count: i64,
    pub total_tokens: i64,
    #[serde(default)]
    pub output_tokens: i64,
    #[serde(default)]
    pub per_model_usage: BTreeMap<String, BTreeMap<String, i64>>,
    #[serde(default)]
    pub enqueued_scopes: Vec<EnqueuedScope>,
}

pub struct Refusal {
    pub detail: String,
    pub retry_after: i64,
    pub reset_at: String,
    pub limit_type: &'static str,
}

impl BatchFacts {
    pub fn tpd_refunds(&self, windows: &[WindowIdentity]) -> Vec<Op> {
        if self.total_tokens <= 0 {
            return Vec::new();
        }
        let tpd: Vec<Counter> = self
            .descriptors
            .iter()
            .filter(|descriptor| descriptor.key.ends_with(TPD_SUFFIX))
            .map(|descriptor| Counter {
                scope: descriptor.scope(),
                kind: Kind::Tokens,
            })
            .collect();
        windows
            .iter()
            .filter(|window| tpd.contains(&window.counter))
            .map(|window| {
                Op::Guarded(GuardedOp {
                    increment: Increment {
                        counter: window.counter.clone(),
                        amount: Number::Int(-self.total_tokens),
                        ttl: Some(Duration::from_secs(TPD_WINDOW_SECONDS)),
                    },
                    expected_window_start: window.window_start.clone(),
                    backend: window.backend,
                })
            })
            .collect()
    }

    pub fn refusal(
        &self,
        status: &Status,
        now: f64,
        default_window: i64,
        window_start: Option<i64>,
    ) -> Refusal {
        let empty = Descriptor {
            key: String::new(),
            value: String::new(),
            rate_limit: None,
        };
        let descriptor = self
            .descriptors
            .iter()
            .find(|descriptor| {
                descriptor.key == status.descriptor_key
                    && status
                        .descriptor_value
                        .as_deref()
                        .is_none_or(|value| descriptor.value == value)
            })
            .or(self.descriptors.first())
            .unwrap_or(&empty);
        let window_size = descriptor
            .limits()
            .window_size
            .filter(|size| *size != 0)
            .unwrap_or(default_window);
        let reset = match window_start {
            Some(start) => (start + window_size) as f64,
            None => now + window_size as f64,
        };
        let retry_after = ((reset - now).trunc() as i64).max(0);
        let reset_at = format_reset(reset);
        let remaining = status.limit_remaining.max(0);
        let limit = status.current_limit;
        let (key, value) = (&descriptor.key, &descriptor.value);
        let limit_type: &'static str = status.rate_limit_type.into();
        let detail = if status.rate_limit_type == LimitKind::Requests {
            format!(
                "Batch rate limit exceeded for {key}: {value}. Batch contains {} requests but only {remaining} requests remaining out of {limit} RPM limit. Limit resets at: {reset_at}",
                self.request_count
            )
        } else {
            let label = if key.ends_with(TPD_SUFFIX) {
                "TPD"
            } else {
                "TPM"
            };
            format!(
                "Batch rate limit exceeded for {key}: {value}. Batch contains {} tokens but only {remaining} tokens remaining out of {limit} {label} limit. Limit resets at: {reset_at}",
                self.token_count(descriptor)
            )
        };
        Refusal {
            detail,
            retry_after,
            reset_at,
            limit_type,
        }
    }

    fn token_count(&self, descriptor: &Descriptor) -> i64 {
        let model_usage = descriptor
            .is_project_io()
            .then(|| {
                descriptor
                    .value
                    .split_once(':')
                    .map_or(descriptor.value.as_str(), |(_, model)| model)
            })
            .filter(|model| !model.is_empty())
            .and_then(|model| self.per_model_usage.get(model));
        let figure = |name: &str, fallback: i64| {
            model_usage
                .and_then(|usage| usage.get(name))
                .copied()
                .unwrap_or(fallback)
        };
        match descriptor.key.as_str() {
            PROJECT_OTPM => figure("output_tokens", self.output_tokens),
            PROJECT_ITPM => figure("total_tokens", self.total_tokens),
            _ => self.total_tokens,
        }
    }
}

pub fn window_start(stored: Option<Local>) -> Option<i64> {
    match stored? {
        Local::Number(Number::Int(value)) => Some(value),
        Local::Number(Number::Float(value)) => value.is_finite().then(|| value.trunc() as i64),
        Local::Text(text) => text
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .map(|value| value.trunc() as i64),
        Local::Slots(_) => None,
    }
}

pub fn format_reset(reset: f64) -> String {
    let micros = (reset * 1_000_000.0).round_ties_even() as i64;
    chrono::DateTime::from_timestamp_micros(micros)
        .map(|at| at.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use litellm_redis_store::rate_limits::Scope;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::status::{Backend, Code};

    fn facts(descriptors: serde_json::Value) -> BatchFacts {
        BatchFacts {
            descriptors: serde_json::from_value(descriptors).unwrap(),
            increments: Vec::new(),
            request_count: 3,
            total_tokens: 900,
            output_tokens: 300,
            per_model_usage: BTreeMap::from([(
                "gpt".to_owned(),
                BTreeMap::from([("total_tokens".to_owned(), 70)]),
            )]),
            enqueued_scopes: Vec::new(),
        }
    }

    fn status(key: &str, value: Option<&str>, kind: LimitKind) -> Status {
        Status {
            code: Code::OverLimit,
            current_limit: 100,
            limit_remaining: -4,
            rate_limit_type: kind,
            descriptor_key: key.into(),
            descriptor_value: value.map(Into::into),
        }
    }

    #[rstest]
    #[case::requests(
        json!([{"key": "api_key", "value": "k", "rate_limit": {"requests_per_unit": 100}}]),
        status("api_key", None, LimitKind::Requests), None,
        "Batch rate limit exceeded for api_key: k. Batch contains 3 requests but only 0 requests remaining out of 100 RPM limit. Limit resets at: 2023-11-14 22:14:20 UTC",
        60
    )]
    #[case::daily_tokens_from_the_window_start(
        json!([{"key": "api_key_tpd", "value": "k", "rate_limit": {"tokens_per_unit": 100, "window_size": 86400}}]),
        status("api_key_tpd", Some("k"), LimitKind::Tokens), Some(1_699_999_000),
        "Batch rate limit exceeded for api_key_tpd: k. Batch contains 900 tokens but only 0 tokens remaining out of 100 TPD limit. Limit resets at: 2023-11-15 21:56:40 UTC",
        85_399
    )]
    #[case::project_input_tokens_for_the_rows_model(
        json!([{"key": "model_per_project_itpm", "value": "p:other", "rate_limit": {"tokens_per_unit": 100}},
               {"key": "model_per_project_itpm", "value": "p:gpt", "rate_limit": {"tokens_per_unit": 100}}]),
        status("model_per_project_itpm", Some("p:gpt"), LimitKind::Tokens), None,
        "Batch rate limit exceeded for model_per_project_itpm: p:gpt. Batch contains 70 tokens but only 0 tokens remaining out of 100 TPM limit. Limit resets at: 2023-11-14 22:14:20 UTC",
        60
    )]
    #[case::project_output_tokens_fall_back_to_the_batch(
        json!([{"key": "model_per_project_otpm", "value": "p:gpt", "rate_limit": {"tokens_per_unit": 100}}]),
        status("model_per_project_otpm", Some("p:gpt"), LimitKind::Tokens), None,
        "Batch rate limit exceeded for model_per_project_otpm: p:gpt. Batch contains 300 tokens but only 0 tokens remaining out of 100 TPM limit. Limit resets at: 2023-11-14 22:14:20 UTC",
        60
    )]
    fn the_refusal_reads_like_the_python_batch_limiter(
        #[case] descriptors: serde_json::Value,
        #[case] over: Status,
        #[case] window_start: Option<i64>,
        #[case] detail: &str,
        #[case] retry_after: i64,
    ) {
        let refusal = facts(descriptors).refusal(&over, 1_700_000_000.5, 60, window_start);
        assert_eq!(
            (refusal.detail.as_str(), refusal.retry_after),
            (detail, retry_after)
        );
    }

    #[rstest]
    #[case::int(Some(Local::Number(Number::Int(5))), Some(5))]
    #[case::float(Some(Local::Number(Number::Float(5.9))), Some(5))]
    #[case::text(Some(Local::Text("12.5".into())), Some(12))]
    #[case::garbage(Some(Local::Text("x".into())), None)]
    #[case::missing(None, None)]
    fn the_window_start_reads_like_int_of_float(
        #[case] stored: Option<Local>,
        #[case] expected: Option<i64>,
    ) {
        assert_eq!(window_start(stored), expected);
    }

    #[rstest]
    fn only_the_daily_counters_are_refunded() {
        let batch = facts(json!([
            {"key": "api_key_tpd", "value": "k", "rate_limit": {"tokens_per_unit": 100}},
            {"key": "team", "value": "t", "rate_limit": {"tokens_per_unit": 100}}
        ]));
        let window = |descriptor: &str| WindowIdentity {
            counter: Counter {
                scope: Scope {
                    descriptor: descriptor.into(),
                    value: if descriptor == "team" { "t" } else { "k" }.into(),
                },
                kind: Kind::Tokens,
            },
            window_start: "1700000000".into(),
            backend: Backend::Local,
        };
        let refunds = batch.tpd_refunds(&[window("api_key_tpd"), window("team")]);
        assert_eq!(
            refunds,
            vec![Op::Guarded(GuardedOp {
                increment: Increment {
                    counter: window("api_key_tpd").counter,
                    amount: Number::Int(-900),
                    ttl: Some(Duration::from_secs(TPD_WINDOW_SECONDS)),
                },
                expected_window_start: "1700000000".into(),
                backend: Backend::Local,
            })]
        );
    }
}
