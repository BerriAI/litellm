//! What `deployment_callback_on_failure` and `async_deployment_callback_on_failure` do after an
//! attempt fails: count the failure, decide on a cooldown, record failure-side usage.

use crate::{
    failure::Classified,
    pyrepr::PyNumber,
    settings::Settings,
    snapshot::{RoutedDeployment, Snapshot},
    store::{Cooldown, Store},
};

const COUNTER_TTL: f64 = 60.0;
const USAGE_TTL: i64 = 60;
const MASK_VISIBLE_PREFIX: usize = 50;

/// `litellm._should_retry`.
pub fn status_is_retryable(status: i64) -> bool {
    matches!(status, 408 | 409 | 429) || status >= 500
}

pub fn successes_key(id: &str) -> String {
    format!("{id}:successes")
}

pub fn fails_key(id: &str) -> String {
    format!("{id}:fails")
}

/// `RouterCacheEnum.RPM` for the minute `now` falls in, in UTC.
pub fn rpm_key(id: &str, model: &str, now: f64) -> String {
    let minute_of_day = (now.rem_euclid(86_400.0) / 60.0) as u64;
    format!(
        "global_router:{id}:{model}:rpm:{:02}-{:02}",
        minute_of_day / 60,
        minute_of_day % 60
    )
}

/// `SensitiveDataMasker(visible_prefix=50, visible_suffix=0)._mask_value`.
pub fn mask(message: &str) -> String {
    let length = message.chars().count();
    if length <= MASK_VISIBLE_PREFIX {
        return message.to_owned();
    }
    message
        .chars()
        .take(MASK_VISIBLE_PREFIX)
        .chain(std::iter::repeat_n('*', length - MASK_VISIBLE_PREFIX))
        .collect()
}

pub async fn record_success(store: &Store, deployment_id: &str) {
    store.increment_local(&successes_key(deployment_id), COUNTER_TTL);
}

/// Both failure callbacks for one failed attempt, in the order Python runs them.
pub async fn record_failure(
    snapshot: &Snapshot,
    store: &Store,
    deployment: &RoutedDeployment,
    failure: &Classified,
) {
    if !failure.exempt_from_cooldown {
        let cooldown_time = cooldown_time(&snapshot.settings, deployment, failure);
        store.increment_local(&fails_key(&deployment.id), COUNTER_TTL);
        set_cooldown(snapshot, store, &deployment.id, failure, cooldown_time).await;
    }
    store
        .increment_usage(
            &rpm_key(&deployment.id, &deployment.model, store.now()),
            1.0,
            USAGE_TTL,
        )
        .await;
}

/// `_trigger_cooldown_for_failed_deployment`: the failure count and cooldown decision without
/// the failure-side usage.
pub async fn record_hop_failure(
    snapshot: &Snapshot,
    store: &Store,
    deployment: &RoutedDeployment,
    failure: &Classified,
) {
    let cooldown_time = cooldown_time(&snapshot.settings, deployment, failure);
    store.increment_local(&fails_key(&deployment.id), COUNTER_TTL);
    set_cooldown(snapshot, store, &deployment.id, failure, cooldown_time).await;
}

/// Deployment config, then the response's Retry-After, then the router default.
fn cooldown_time(
    settings: &Settings,
    deployment: &RoutedDeployment,
    failure: &Classified,
) -> PyNumber {
    deployment
        .cooldown_time
        .filter(|time| time.as_f64() >= 0.0)
        .or(failure
            .cooldown_retry_after
            .filter(|seconds| *seconds >= 0)
            .map(PyNumber::Int))
        .unwrap_or(settings.cooldown_time)
}

/// `_set_cooldown_deployments` for the configurations the first increment serves (no
/// deployment-level or router-level allowed-fails policy, no routing groups or team models).
async fn set_cooldown(
    snapshot: &Snapshot,
    store: &Store,
    deployment_id: &str,
    failure: &Classified,
    cooldown_time: PyNumber,
) -> bool {
    let Some(group_size) = snapshot.group_size_of(deployment_id) else {
        return false;
    };
    let settings = &snapshot.settings;
    if (cooldown_time.as_f64() - 0.0).abs() <= 1e-9
        || settings.disable_cooldowns
        || !is_cooldown_required(failure)
    {
        return false;
    }
    let status = failure.status_code.unwrap_or(500);
    let single = group_size == 1;
    let should = if settings.allowed_fails_set_on_router {
        exceeds_allowed_fails(settings, store, deployment_id).await
    } else {
        base_case(settings, store, deployment_id, status, single)
    };
    if should {
        store
            .set_cooldown(
                deployment_id,
                Cooldown {
                    exception_received: mask(&failure.message),
                    status_code: status.to_string(),
                    timestamp: store.now(),
                    cooldown_time,
                },
            )
            .await;
    }
    should
}

/// `_is_cooldown_required`: connection errors and status-less failures never cool down; of
/// the 4xx statuses only 401, 404, 408 and 429 do.
fn is_cooldown_required(failure: &Classified) -> bool {
    if failure.message.contains("APIConnectionError") {
        return false;
    }
    match failure.status_code {
        None => false,
        Some(status) if (400..500).contains(&status) => matches!(status, 401 | 404 | 408 | 429),
        Some(_) => true,
    }
}

fn base_case(
    settings: &Settings,
    store: &Store,
    deployment_id: &str,
    status: i64,
    single: bool,
) -> bool {
    let successes = store.local_count(&successes_key(deployment_id));
    let fails = store.local_count(&fails_key(deployment_id));
    let total = successes + fails;
    let percent_fails = if total > 0.0 { fails / total } else { 0.0 };
    let tunables = &settings.tunables;
    (status == 429 && !single)
        || (percent_fails == 1.0
            && total >= tunables.single_deployment_traffic_failure_threshold as f64)
        || (percent_fails > tunables.failure_threshold_percent
            && total >= tunables.failure_threshold_minimum_requests as f64
            && !single)
        || !status_is_retryable(status)
}

/// `should_cooldown_based_on_allowed_fails_policy` with only `router.allowed_fails` set: a
/// fleet-wide counter whose TTL is the router cooldown time.
async fn exceeds_allowed_fails(settings: &Settings, store: &Store, deployment_id: &str) -> bool {
    let ttl = match settings.cooldown_time.as_f64() {
        time if time != 0.0 => time,
        _ => settings.tunables.default_cooldown_time,
    };
    let updated = store
        .increment_shared(&format!("deployment:{deployment_id}:allowed_fails"), ttl)
        .await;
    updated > i64::from(settings.allowed_fails.unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;

    use super::{mask, record_failure, rpm_key};
    use crate::{
        failure::Classified,
        pyrepr::PyNumber,
        settings::Settings,
        snapshot::{RoutedDeployment, Snapshot},
        store::{Store, SystemClock},
    };

    fn deployment(id: &str, group: &str) -> RoutedDeployment {
        RoutedDeployment {
            id: id.into(),
            model_name: group.into(),
            model: format!("openai/{id}"),
            weight: None,
            cooldown_time: None,
            num_retries: None,
            supports_web_search: None,
        }
    }

    fn snapshot(deployments: Vec<RoutedDeployment>) -> Snapshot {
        Snapshot::new(
            deployments,
            Settings {
                cooldown_time: PyNumber::Int(5),
                ..Settings::default()
            },
            vec![],
        )
    }

    fn failure(status: Option<i64>, message: &str) -> Classified {
        Classified {
            status_code: status,
            message: message.into(),
            ..Classified::default()
        }
    }

    #[rstest]
    #[case::rate_limit_with_alternatives(2, Some(429), "", true)]
    #[case::rate_limit_single_deployment(1, Some(429), "", false)]
    #[case::auth_single_deployment(1, Some(401), "", true)]
    #[case::not_found(1, Some(404), "", true)]
    #[case::bad_request(2, Some(400), "", false)]
    #[case::server_error_first_failure(2, Some(500), "", false)]
    #[case::connection_error(2, Some(500), "litellm.APIConnectionError: x", false)]
    #[case::no_status(2, None, "", false)]
    #[tokio::test]
    async fn first_failure_cooldown_follows_the_python_base_case(
        #[case] group_size: usize,
        #[case] status: Option<i64>,
        #[case] message: &str,
        #[case] cooled: bool,
    ) {
        let deployments: Vec<_> = (0..group_size)
            .map(|index| deployment(&format!("d{index}"), "g"))
            .collect();
        let snapshot = snapshot(deployments);
        let store = Store::new(Arc::new(SystemClock), None, 1.0);
        let target = snapshot.by_id("d0").unwrap().clone();

        record_failure(&snapshot, &store, &target, &failure(status, message)).await;

        assert_eq!(store.active_cooldowns(&["d0"]).await.len() == 1, cooled);
        assert_eq!(store.local_count("d0:fails"), 1.0);
    }

    #[rstest]
    #[tokio::test]
    async fn exempt_failures_count_usage_but_not_fails() {
        let snapshot = snapshot(vec![deployment("d0", "g")]);
        let store = Store::new(Arc::new(SystemClock), None, 1.0);
        let target = snapshot.by_id("d0").unwrap().clone();
        let exempt = Classified {
            exempt_from_cooldown: true,
            ..failure(Some(401), "")
        };

        record_failure(&snapshot, &store, &target, &exempt).await;

        assert!(store.active_cooldowns(&["d0"]).await.is_empty());
        assert_eq!(store.local_count("d0:fails"), 0.0);
        assert_eq!(
            store.local_count(&rpm_key("d0", "openai/d0", store.now())),
            1.0
        );
    }

    #[rstest]
    #[case::short("short", "short")]
    #[case::long(&"a".repeat(52), &format!("{}**", "a".repeat(50)))]
    #[case::counts_characters(&"é".repeat(51), &format!("{}*", "é".repeat(50)))]
    fn mask_keeps_the_first_fifty_characters(#[case] message: &str, #[case] expected: &str) {
        assert_eq!(mask(message), expected);
    }

    #[rstest]
    fn rpm_key_uses_the_utc_minute() {
        assert_eq!(
            rpm_key(
                "i",
                "m",
                86_400.0 * 3.0 + 3_600.0 * 13.0 + 60.0 * 7.0 + 59.9
            ),
            "global_router:i:m:rpm:13-07"
        );
    }
}
