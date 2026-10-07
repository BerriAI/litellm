//! One attempt's deployment pick: `_common_checks_available_deployment`, the healthy-deployment
//! filters the first increment serves, then `simple_shuffle`.

use std::sync::Mutex;

use crate::{
    failure::Rejection,
    random::PythonRandom,
    settings::FallbackEntry,
    snapshot::{RoutedDeployment, Snapshot},
    store::Store,
};

/// The request facts selection reads.
#[derive(Clone, Debug, Default)]
pub struct SelectionRequest {
    /// The request asked for a `web_search` or `web_search_preview` tool.
    pub web_search: bool,
    /// `_retry_skipped_deployment_ids` from the retry layer.
    pub retry_skipped: Vec<String>,
}

pub async fn pick<'a>(
    snapshot: &'a Snapshot,
    store: &Store,
    random: &Mutex<PythonRandom>,
    model: &str,
    request: &SelectionRequest,
) -> Result<&'a RoutedDeployment, Rejection> {
    let (group, candidates) = match resolve(snapshot, model)? {
        Resolved::Deployment(deployment) => return Ok(deployment),
        Resolved::Group(group, candidates) => (group, candidates),
    };
    let candidates: Vec<&RoutedDeployment> = candidates
        .into_iter()
        .filter(|deployment| !request.web_search || deployment.supports_web_search != Some(false))
        .collect();
    let ids: Vec<&str> = candidates
        .iter()
        .map(|deployment| deployment.id.as_str())
        .collect();
    let cooling = store.active_cooldowns(&ids).await;
    let healthy: Vec<&RoutedDeployment> = candidates
        .into_iter()
        .filter(|deployment| !cooling.contains(&deployment.id))
        .collect();
    let not_skipped: Vec<&RoutedDeployment> = healthy
        .iter()
        .copied()
        .filter(|deployment| !request.retry_skipped.contains(&deployment.id))
        .collect();
    let healthy = if not_skipped.is_empty() {
        healthy
    } else {
        not_skipped
    };
    if healthy.is_empty() {
        return Err(no_deployments_available(snapshot, store, &group).await);
    }
    let index = shuffle(
        &healthy,
        &mut random
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    Ok(healthy[index])
}

enum Resolved<'a> {
    Deployment(&'a RoutedDeployment),
    Group(String, Vec<&'a RoutedDeployment>),
}

fn resolve<'a>(snapshot: &'a Snapshot, model: &str) -> Result<Resolved<'a>, Rejection> {
    if !snapshot.has_group(model)
        && let Some(deployment) = snapshot.by_id(model)
    {
        return Ok(Resolved::Deployment(deployment));
    }
    let group = snapshot.group(model);
    if !group.is_empty() {
        return Ok(Resolved::Group(model.to_owned(), group));
    }
    let by_model = snapshot.by_litellm_model(model);
    if !by_model.is_empty() {
        return Ok(Resolved::Group(model.to_owned(), by_model));
    }
    let default_group =
        first_default_fallback(snapshot.settings.fallbacks.as_deref().unwrap_or_default());
    let model = default_group.unwrap_or(model);
    let group = snapshot.group(model);
    if group.is_empty() {
        return Err(Rejection::NoHealthyDeployments {
            model: model.to_owned(),
        });
    }
    Ok(Resolved::Group(model.to_owned(), group))
}

/// `_get_first_default_fallback`.
fn first_default_fallback(fallbacks: &[FallbackEntry]) -> Option<&str> {
    fallbacks.iter().find_map(|entry| match entry {
        FallbackEntry::Keyed { key, targets } if key == "*" => targets.first().map(String::as_str),
        _ => None,
    })
}

/// `async_raise_no_deployment_exception`.
pub async fn no_deployments_available(
    snapshot: &Snapshot,
    store: &Store,
    model: &str,
) -> Rejection {
    let model_ids: Vec<String> = snapshot
        .group(model)
        .iter()
        .map(|deployment| deployment.id.clone())
        .collect();
    let group_ids: Vec<&str> = model_ids.iter().map(String::as_str).collect();
    let minimum = store
        .cooldowns(&group_ids)
        .await
        .into_iter()
        .flatten()
        .map(|cooldown| cooldown.cooldown_time)
        .reduce(|lowest, time| {
            if time.as_f64() < lowest.as_f64() {
                time
            } else {
                lowest
            }
        });
    let all_ids: Vec<&str> = snapshot.ids().collect();
    Rejection::NoDeploymentsAvailable {
        model: model.to_owned(),
        cooldown_time: minimum
            .filter(|time| time.as_f64() != 0.0)
            .unwrap_or(snapshot.settings.cooldown_time),
        cooldown_list: store.active_cooldowns(&all_ids).await,
        model_ids,
        enable_pre_call_checks: snapshot.settings.enable_pre_call_checks,
    }
}

/// `simple_shuffle` with `litellm_params.weight` as the only weight source.
fn shuffle(healthy: &[&RoutedDeployment], random: &mut PythonRandom) -> usize {
    let weights: Vec<f64> = healthy
        .iter()
        .map(|deployment| deployment.weight.unwrap_or(0.0))
        .collect();
    let largest = weights.iter().copied().fold(0.0_f64, f64::max);
    if largest > 0.0 {
        let normalized: Vec<f64> = weights.iter().map(|weight| weight / largest).collect();
        if normalized.iter().sum::<f64>() > 0.0
            && let Some(index) = random.weighted_choice(&normalized)
        {
            return index;
        }
    }
    random.choice(healthy.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use rstest::rstest;

    use super::{SelectionRequest, pick};
    use crate::{
        failure::Rejection,
        pyrepr::PyNumber,
        random::PythonRandom,
        settings::{FallbackEntry, Settings},
        snapshot::{RoutedDeployment, Snapshot},
        store::{Cooldown, Store, SystemClock},
    };

    fn deployment(id: &str, group: &str, weight: Option<f64>) -> RoutedDeployment {
        RoutedDeployment {
            id: id.into(),
            model_name: group.into(),
            model: format!("openai/{id}"),
            weight,
            cooldown_time: None,
            num_retries: None,
            supports_web_search: None,
        }
    }

    fn snapshot(
        deployments: Vec<RoutedDeployment>,
        fallbacks: Option<Vec<FallbackEntry>>,
    ) -> Snapshot {
        Snapshot::new(
            deployments,
            Settings {
                cooldown_time: PyNumber::Int(5),
                fallbacks,
                ..Settings::default()
            },
            vec![],
        )
    }

    async fn cool(store: &Store, id: &str, time: PyNumber) {
        store
            .set_cooldown(
                id,
                Cooldown {
                    exception_received: String::new(),
                    status_code: "429".into(),
                    timestamp: store.now(),
                    cooldown_time: time,
                },
            )
            .await;
    }

    async fn picks(
        snapshot: &Snapshot,
        store: &Store,
        model: &str,
        seed: u64,
        count: usize,
    ) -> Vec<String> {
        let random = Mutex::new(PythonRandom::seeded(seed));
        let mut picked = Vec::new();
        for _ in 0..count {
            let deployment = pick(
                snapshot,
                store,
                &random,
                model,
                &SelectionRequest::default(),
            )
            .await
            .unwrap();
            picked.push(deployment.id.clone());
        }
        picked
    }

    fn store() -> Store {
        Store::new(Arc::new(SystemClock), None, 1.0)
    }

    #[rstest]
    #[tokio::test]
    async fn unweighted_pick_is_python_random_choice() {
        let snapshot = snapshot(
            ["a", "b", "c", "d", "e"]
                .iter()
                .map(|id| deployment(id, "g", None))
                .collect(),
            None,
        );
        assert_eq!(
            picks(&snapshot, &store(), "g", 42, 5).await,
            ["a", "a", "c", "b", "b"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn weighted_pick_is_python_random_choices_over_normalized_weights() {
        let snapshot = snapshot(
            vec![
                deployment("a", "g", Some(4.0)),
                deployment("b", "g", Some(2.0)),
                deployment("c", "g", Some(1.0)),
            ],
            None,
        );
        assert_eq!(
            picks(&snapshot, &store(), "g", 7, 6).await,
            ["a", "a", "b", "a", "a", "a"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn cooling_deployments_are_never_picked() {
        let snapshot = snapshot(
            vec![deployment("a", "g", None), deployment("b", "g", None)],
            None,
        );
        let store = store();
        cool(&store, "a", PyNumber::Int(60)).await;
        assert!(
            picks(&snapshot, &store, "g", 1, 20)
                .await
                .iter()
                .all(|id| id == "b")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn every_deployment_cooling_rejects_with_the_lowest_cooldown() {
        let snapshot = snapshot(
            vec![deployment("a", "g", None), deployment("b", "g", None)],
            None,
        );
        let store = store();
        cool(&store, "a", PyNumber::Int(60)).await;
        cool(&store, "b", PyNumber::Float(30.5)).await;
        let rejection = pick(
            &snapshot,
            &store,
            &Mutex::new(PythonRandom::seeded(1)),
            "g",
            &SelectionRequest::default(),
        )
        .await
        .unwrap_err();
        assert_eq!(
            rejection,
            Rejection::NoDeploymentsAvailable {
                model: "g".into(),
                cooldown_time: PyNumber::Float(30.5),
                cooldown_list: vec!["a".into(), "b".into()],
                model_ids: vec!["a".into(), "b".into()],
                enable_pre_call_checks: false,
            }
        );
    }

    #[rstest]
    #[case::deployment_id("b", Ok("b"))]
    #[case::litellm_model_name("openai/a", Ok("a"))]
    #[case::unknown("nope", Err(Rejection::NoHealthyDeployments { model: "nope".into() }))]
    #[tokio::test]
    async fn non_group_models_resolve_like_python(
        #[case] model: &str,
        #[case] expected: Result<&str, Rejection>,
    ) {
        let snapshot = snapshot(
            vec![deployment("a", "g", None), deployment("b", "h", None)],
            None,
        );
        let result = pick(
            &snapshot,
            &store(),
            &Mutex::new(PythonRandom::seeded(1)),
            model,
            &SelectionRequest::default(),
        )
        .await
        .map(|deployment| deployment.id.as_str());
        assert_eq!(result, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn unknown_model_uses_the_first_default_fallback_group() {
        let snapshot = snapshot(
            vec![deployment("a", "g", None), deployment("b", "h", None)],
            Some(vec![FallbackEntry::Keyed {
                key: "*".into(),
                targets: vec!["h".into()],
            }]),
        );
        assert_eq!(picks(&snapshot, &store(), "nope", 1, 1).await, ["b"]);
    }

    #[rstest]
    #[tokio::test]
    async fn retry_skip_is_ignored_when_it_would_leave_nothing() {
        let snapshot = snapshot(
            vec![deployment("a", "g", None), deployment("b", "g", None)],
            None,
        );
        let random = Mutex::new(PythonRandom::seeded(3));
        let skip_a = SelectionRequest {
            retry_skipped: vec!["a".into()],
            ..SelectionRequest::default()
        };
        let skip_all = SelectionRequest {
            retry_skipped: vec!["a".into(), "b".into()],
            ..SelectionRequest::default()
        };
        let store = store();
        for _ in 0..10 {
            assert_eq!(
                pick(&snapshot, &store, &random, "g", &skip_a)
                    .await
                    .unwrap()
                    .id,
                "b"
            );
            assert!(
                pick(&snapshot, &store, &random, "g", &skip_all)
                    .await
                    .is_ok()
            );
        }
    }
}
