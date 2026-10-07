use std::collections::HashMap;

use serde::Deserialize;

use crate::{failure::ExceptionClass, pyrepr::PyNumber};

/// `litellm.types.router.RetryPolicy`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub struct RetryPolicy {
    pub bad_request_error_retries: Option<u32>,
    pub authentication_error_retries: Option<u32>,
    pub timeout_error_retries: Option<u32>,
    pub rate_limit_error_retries: Option<u32>,
    pub content_policy_violation_error_retries: Option<u32>,
    pub internal_server_error_retries: Option<u32>,
    pub service_unavailable_error_retries: Option<u32>,
    pub not_found_error_retries: Option<u32>,
    pub default_retries: Option<u32>,
}

impl RetryPolicy {
    /// `get_num_retries_from_retry_policy`: a 404 prefers `NotFoundErrorRetries`, then the
    /// exception's classes are walked most specific first.
    pub fn retries_for(&self, classes: &[ExceptionClass], status_code: Option<i64>) -> Option<u32> {
        let not_found = (status_code == Some(404))
            .then_some(self.not_found_error_retries)
            .flatten();
        not_found
            .into_iter()
            .chain(classes.iter().filter_map(|class| self.by_class(*class)))
            .next()
            .or(self.default_retries)
    }

    fn by_class(&self, class: ExceptionClass) -> Option<u32> {
        match class {
            ExceptionClass::Authentication => self.authentication_error_retries,
            ExceptionClass::Timeout => self.timeout_error_retries,
            ExceptionClass::RateLimit => self.rate_limit_error_retries,
            ExceptionClass::ContentPolicyViolation => self.content_policy_violation_error_retries,
            ExceptionClass::BadRequest => self.bad_request_error_retries,
            ExceptionClass::ServiceUnavailable => self.service_unavailable_error_retries,
            ExceptionClass::InternalServer => self.internal_server_error_retries,
            _ => None,
        }
    }
}

/// One entry of a router fallback list: `{"group": ["a", "b"]}` (only the first key counts,
/// as in Python) or a bare `"group"`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FallbackEntry {
    Keyed { key: String, targets: Vec<String> },
    Bare(String),
}

impl<'de> Deserialize<'de> for FallbackEntry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bare(String),
            Keyed(serde_json::Map<String, serde_json::Value>),
        }
        match Raw::deserialize(deserializer)? {
            Raw::Bare(group) => Ok(Self::Bare(group)),
            Raw::Keyed(map) => {
                let (key, value) = map
                    .into_iter()
                    .next()
                    .ok_or_else(|| serde::de::Error::custom("empty fallback entry"))?;
                let targets = Vec::<String>::deserialize(value).map_err(|_| {
                    serde::de::Error::custom("fallback targets other than model group names")
                })?;
                Ok(Self::Keyed { key, targets })
            }
        }
    }
}

pub type Fallbacks = Vec<FallbackEntry>;

/// Constants Python reads from `litellm.constants`, which operators can override through the
/// environment, so they cross with the settings instead of being hardcoded here.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Tunables {
    pub initial_retry_delay: f64,
    pub max_retry_delay: f64,
    pub jitter: f64,
    pub failure_threshold_percent: f64,
    pub failure_threshold_minimum_requests: u64,
    pub single_deployment_traffic_failure_threshold: u64,
    pub default_cooldown_time: f64,
    pub cooldown_redis_read_interval: f64,
}

impl Default for Tunables {
    fn default() -> Self {
        Self {
            initial_retry_delay: 0.5,
            max_retry_delay: 8.0,
            jitter: 0.75,
            failure_threshold_percent: 0.5,
            failure_threshold_minimum_requests: 5,
            single_deployment_traffic_failure_threshold: 1000,
            default_cooldown_time: 5.0,
            cooldown_redis_read_interval: 1.0,
        }
    }
}

/// The router attributes the hot path reads, already resolved the way `PythonRouter.__init__`
/// resolves them (global defaults, `or` fallbacks), so `None` and `[]` keep their Python meaning.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Settings {
    pub num_retries: u32,
    pub retry_after: f64,
    pub max_fallbacks: u32,
    pub fallbacks: Option<Fallbacks>,
    pub context_window_fallbacks: Option<Fallbacks>,
    pub content_policy_fallbacks: Option<Fallbacks>,
    pub retry_policy: Option<RetryPolicy>,
    pub model_group_retry_policy: Option<HashMap<String, RetryPolicy>>,
    pub allowed_fails: Option<u32>,
    pub allowed_fails_set_on_router: bool,
    pub cooldown_time: PyNumber,
    pub disable_cooldowns: bool,
    pub enable_pre_call_checks: bool,
    /// A `fallback_access_check` or `fallback_budget_check` is set, so each cross-group
    /// fallback target is put to the host before it is tried.
    #[serde(default)]
    pub fallback_checks: bool,
    #[serde(default)]
    pub tunables: Tunables,
}

impl Settings {
    /// `resolve_retry_policy`: the group's own policy wins over the router-wide one.
    pub fn retry_policy_for(&self, model_group: &str) -> Option<&RetryPolicy> {
        self.model_group_retry_policy
            .as_ref()
            .and_then(|policies| policies.get(model_group))
            .or(self.retry_policy.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{FallbackEntry, RetryPolicy};
    use crate::failure::ExceptionClass;

    #[rstest]
    fn fallback_entries_read_python_shapes() {
        let entries: Vec<FallbackEntry> =
            serde_json::from_str(r#"[{"a": ["b", "c"], "ignored": ["d"]}, "e", {"*": []}]"#)
                .unwrap();
        assert_eq!(
            entries,
            [
                FallbackEntry::Keyed {
                    key: "a".into(),
                    targets: vec!["b".into(), "c".into()]
                },
                FallbackEntry::Bare("e".into()),
                FallbackEntry::Keyed {
                    key: "*".into(),
                    targets: vec![]
                },
            ]
        );
    }

    #[rstest]
    fn dict_fallback_targets_are_rejected() {
        assert!(serde_json::from_str::<FallbackEntry>(r#"{"a": [{"model": "b"}]}"#).is_err());
    }

    #[rstest]
    #[case::not_found_status_wins(Some(404), &[ExceptionClass::RateLimit], Some(9))]
    #[case::most_specific_class_first(None, &[ExceptionClass::ContentPolicyViolation, ExceptionClass::BadRequest], Some(3))]
    #[case::falls_through_unset_classes(None, &[ExceptionClass::Timeout, ExceptionClass::BadRequest], Some(1))]
    #[case::default_when_nothing_matches(None, &[ExceptionClass::NotFound], Some(7))]
    fn retry_policy_walks_classes(
        #[case] status: Option<i64>,
        #[case] classes: &[ExceptionClass],
        #[case] expected: Option<u32>,
    ) {
        let policy = RetryPolicy {
            not_found_error_retries: Some(9),
            content_policy_violation_error_retries: Some(3),
            bad_request_error_retries: Some(1),
            default_retries: Some(7),
            ..RetryPolicy::default()
        };
        assert_eq!(policy.retries_for(classes, status), expected);
    }
}
