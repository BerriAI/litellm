mod deployment;
mod error;
mod group;
mod health;
mod policy;

use std::{collections::HashMap, time::Instant};

use litellm_config::{Config, Model};

pub use deployment::Deployment;
pub use error::Error;
pub use group::ModelGroup;
pub use health::{Failure, InFlight};
pub use policy::{CooldownPolicy, RoutingPolicy, RoutingStrategy};

use group::Member;

/// The model groups of a config and the health of each deployment in them.
#[derive(Debug)]
pub struct Router {
    groups: HashMap<String, ModelGroup>,
    policy: RoutingPolicy,
    epoch: Instant,
}

impl Default for Router {
    fn default() -> Self {
        Self::new(std::iter::empty(), RoutingPolicy::default())
    }
}

/// One call to make, on `deployment`. `last` is set when no retry can follow it.
pub struct Attempt<'a> {
    pub deployment: &'a Deployment,
    pub last: bool,
}

/// A successful call, and the in-flight mark its deployment keeps until the guard drops.
pub struct Routed<T> {
    pub value: T,
    pub in_flight: InFlight,
}

impl Router {
    pub fn from_config(config: &Config) -> Result<Self, Error> {
        let policy =
            RoutingPolicy::from_settings(&config.router_settings, &config.litellm_settings)?;
        Self::with_policy(&config.model_list, policy)
    }

    pub fn from_model_list(model_list: &[Model]) -> Result<Self, Error> {
        Self::with_policy(model_list, RoutingPolicy::default())
    }

    fn with_policy(model_list: &[Model], policy: RoutingPolicy) -> Result<Self, Error> {
        let deployments = model_list
            .iter()
            .map(|model| {
                let deployment = Deployment::from_params(&model.litellm_params)?;
                Ok((
                    model.model_name.clone(),
                    Deployment {
                        timeout: deployment.timeout.or(policy.timeout),
                        team_id: owner_team(model),
                        ..deployment
                    },
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self::new(deployments, policy))
    }

    pub fn new(
        deployments: impl IntoIterator<Item = (String, Deployment)>,
        policy: RoutingPolicy,
    ) -> Self {
        let mut members: HashMap<String, Vec<Member>> = HashMap::new();
        for (name, deployment) in deployments {
            members
                .entry(name)
                .or_default()
                .push(Member::new(deployment));
        }
        Self {
            groups: members
                .into_iter()
                .map(|(name, members)| {
                    let group = ModelGroup::new(name.as_str().into(), members.into());
                    (name, group)
                })
                .collect(),
            policy,
            epoch: Instant::now(),
        }
    }

    pub fn group(&self, model_name: &str) -> Option<&ModelGroup> {
        self.groups.get(model_name)
    }

    pub fn policy(&self) -> &RoutingPolicy {
        &self.policy
    }

    /// One deployment of `group` for a call made without retries.
    pub fn select<'a>(&self, group: &'a ModelGroup) -> &'a Deployment {
        let index = group
            .choose(self.policy.strategy, self.now(), &[], rand::random())
            .unwrap_or_default();
        &group.member(index).deployment
    }

    /// Like [`Router::select`], among the deployments `usable` accepts; `None` when it accepts
    /// none of them.
    pub fn select_where<'a>(
        &self,
        group: &'a ModelGroup,
        usable: impl Fn(&Deployment) -> bool,
    ) -> Option<&'a Deployment> {
        let excluded = excluded(group, usable);
        group
            .choose(self.policy.strategy, self.now(), &excluded, rand::random())
            .map(|index| &group.member(index).deployment)
    }

    /// Like [`Router::route`], among the deployments `usable` accepts, so neither the first
    /// attempt nor a retry reaches another. `None`, without calling, when it accepts none.
    pub async fn route_where<'a, T, E, F, Fut, C>(
        &self,
        group: &'a ModelGroup,
        usable: impl Fn(&Deployment) -> bool,
        call: F,
        classify: C,
    ) -> Option<Result<Routed<T>, E>>
    where
        F: FnMut(Attempt<'a>) -> Fut,
        Fut: Future<Output = Result<T, E>>,
        C: Fn(&E) -> Option<Failure>,
    {
        let excluded = excluded(group, usable);
        if excluded.len() >= group.len() {
            return None;
        }
        Some(self.route_among(group, excluded, call, classify).await)
    }

    /// Runs `call` on a deployment of `group` and, while `classify` names the error a
    /// [`Failure`] and `num_retries` allows, again on a deployment not yet tried. Every
    /// failure counts toward its deployment's cooldown. A call that returned `Ok` is never
    /// repeated, so a stream that already started is not retried.
    pub async fn route<'a, T, E, F, Fut, C>(
        &self,
        group: &'a ModelGroup,
        call: F,
        classify: C,
    ) -> Result<Routed<T>, E>
    where
        F: FnMut(Attempt<'a>) -> Fut,
        Fut: Future<Output = Result<T, E>>,
        C: Fn(&E) -> Option<Failure>,
    {
        self.route_among(group, Vec::new(), call, classify).await
    }

    /// Routes over the deployments of `group` not in `excluded`, which leaves at least one.
    async fn route_among<'a, T, E, F, Fut, C>(
        &self,
        group: &'a ModelGroup,
        excluded: Vec<usize>,
        mut call: F,
        classify: C,
    ) -> Result<Routed<T>, E>
    where
        F: FnMut(Attempt<'a>) -> Fut,
        Fut: Future<Output = Result<T, E>>,
        C: Fn(&E) -> Option<Failure>,
    {
        let attempts = usize::try_from(self.policy.num_retries)
            .unwrap_or(usize::MAX)
            .saturating_add(1);
        let skipped = excluded.len();
        let mut tried = excluded;
        let mut index = group
            .choose(self.policy.strategy, self.now(), &tried, rand::random())
            .unwrap_or_default();
        loop {
            tried.push(index);
            let member = group.member(index);
            let in_flight = InFlight::enter(&member.health);
            let attempt = Attempt {
                deployment: &member.deployment,
                last: tried.len() - skipped >= attempts || tried.len() >= group.len(),
            };
            let last = attempt.last;
            let error = match call(attempt).await {
                Ok(value) => {
                    member
                        .health
                        .record_success(self.now(), &self.policy.cooldown);
                    return Ok(Routed { value, in_flight });
                }
                Err(error) => error,
            };
            drop(in_flight);
            let Some(failure) = classify(&error) else {
                return Err(error);
            };
            if member.health.record_failure(
                self.now(),
                failure,
                &self.policy.cooldown,
                group.len() > 1,
            ) {
                tracing::warn!(
                    model_group = group.name(),
                    deployment = index,
                    cooldown_secs = self.policy.cooldown.cooldown_time.as_secs_f64(),
                    "deployment cooling down"
                );
            }
            let next = (!last)
                .then(|| group.choose(self.policy.strategy, self.now(), &tried, rand::random()))
                .flatten();
            let Some(next) = next else {
                return Err(error);
            };
            tracing::debug!(
                model_group = group.name(),
                from = index,
                to = next,
                ?failure,
                "retrying on another deployment"
            );
            index = next;
        }
    }

    fn now(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// The indices of the deployments in `group` that `usable` rejects.
fn excluded(group: &ModelGroup, usable: impl Fn(&Deployment) -> bool) -> Vec<usize> {
    (0..group.len())
        .filter(|index| !usable(&group.member(*index).deployment))
        .collect()
}

/// `model_info.team_id`. A value that is not a string still marks the deployment as owned, by
/// a team no key can name, so it is never shared by mistake.
fn owner_team(model: &Model) -> Option<String> {
    match model.model_info.get("team_id")? {
        litellm_config::Value::Null => None,
        litellm_config::Value::String(team_id) => Some(team_id.clone()),
        other => Some(format!("{other:?}")),
    }
}

impl FromIterator<(String, Deployment)> for Router {
    fn from_iter<I: IntoIterator<Item = (String, Deployment)>>(entries: I) -> Self {
        Self::new(entries, RoutingPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap, time::Duration};

    use rstest::rstest;

    use super::*;

    type Outcome = Result<&'static str, Option<Failure>>;

    fn router(models: &[&str], policy: RoutingPolicy) -> Router {
        Router::new(
            models.iter().map(|model| {
                (
                    "group".to_owned(),
                    Deployment {
                        model: (*model).to_owned(),
                        ..Deployment::default()
                    },
                )
            }),
            policy,
        )
    }

    fn retries(num_retries: u32) -> RoutingPolicy {
        RoutingPolicy {
            num_retries,
            ..RoutingPolicy::default()
        }
    }

    /// Routes one request where `outcome` decides each deployment's answer, and returns the
    /// deployments tried in order with the `last` flag each attempt carried.
    async fn send(
        router: &Router,
        outcome: impl Fn(&str) -> Outcome,
    ) -> (
        Vec<(String, bool)>,
        Result<Routed<&'static str>, Option<Failure>>,
    ) {
        let tried = RefCell::new(Vec::new());
        let group = router.group("group").unwrap();
        let result = router
            .route(
                group,
                |attempt| {
                    tried
                        .borrow_mut()
                        .push((attempt.deployment.model.clone(), attempt.last));
                    let result = outcome(&attempt.deployment.model);
                    async move { result }
                },
                |failure| *failure,
            )
            .await;
        (tried.into_inner(), result)
    }

    #[rstest]
    #[case::rate_limited(Failure::RateLimited)]
    #[case::unavailable(Failure::Unavailable)]
    #[case::rejected(Failure::Rejected)]
    #[tokio::test]
    async fn a_failure_retries_on_another_deployment(#[case] failure: Failure) {
        let router = router(&["a", "b"], retries(2));
        let first = RefCell::new(None);

        let (tried, result) = send(&router, |model| {
            if first.borrow().is_none() {
                first.replace(Some(model.to_owned()));
                return Err(Some(failure));
            }
            Ok("served")
        })
        .await;

        assert_eq!(tried.len(), 2);
        assert_ne!(tried[0].0, tried[1].0);
        assert_eq!(result.ok().map(|routed| routed.value), Some("served"));
    }

    fn owned_by(teams: &[(&str, Option<&str>)]) -> Router {
        Router::new(
            teams.iter().map(|(model, team_id)| {
                (
                    "group".to_owned(),
                    Deployment {
                        model: (*model).to_owned(),
                        team_id: team_id.map(str::to_owned),
                        ..Deployment::default()
                    },
                )
            }),
            retries(5),
        )
    }

    #[rstest]
    #[case::team_a(Some("team-a"), Some(vec!["a", "shared"]))]
    #[case::teamless(None, Some(vec!["shared"]))]
    #[case::team_c(Some("team-c"), Some(vec!["shared"]))]
    #[tokio::test]
    async fn route_where_never_reaches_a_rejected_deployment(
        #[case] team: Option<&str>,
        #[case] expected: Option<Vec<&str>>,
    ) {
        let router = owned_by(&[
            ("shared", None),
            ("a", Some("team-a")),
            ("b", Some("team-b")),
        ]);
        let group = router.group("group").unwrap();
        let usable = |deployment: &Deployment| {
            deployment.team_id.is_none() || deployment.team_id.as_deref() == team
        };
        let tried = RefCell::new(Vec::new());

        let result = router
            .route_where(
                group,
                usable,
                |attempt| {
                    tried
                        .borrow_mut()
                        .push((attempt.deployment.model.clone(), attempt.last));
                    async { Err::<(), _>(Some(Failure::Unavailable)) }
                },
                |failure| *failure,
            )
            .await;

        let tried = tried.into_inner();
        let mut models: Vec<&str> = tried.iter().map(|(model, _)| model.as_str()).collect();
        models.sort_unstable();
        assert_eq!(Some(models), expected);
        assert_eq!(tried.last().map(|(_, last)| *last), Some(true));
        assert!(matches!(result, Some(Err(Some(Failure::Unavailable)))));
        for _ in 0..50 {
            let selected = router.select_where(group, usable).map(|d| d.model.as_str());
            assert!(selected.is_some_and(|model| model != "b"), "{selected:?}");
        }
    }

    #[rstest]
    #[tokio::test]
    async fn route_where_without_a_usable_deployment_calls_nothing() {
        let router = owned_by(&[("a", Some("team-a"))]);
        let group = router.group("group").unwrap();
        let calls = RefCell::new(0);

        let result = router
            .route_where(
                group,
                |deployment| deployment.team_id.is_none(),
                |_| {
                    *calls.borrow_mut() += 1;
                    async { Ok::<_, Option<Failure>>("served") }
                },
                |failure| *failure,
            )
            .await;

        assert!(result.is_none());
        assert_eq!(calls.into_inner(), 0);
        assert!(
            router
                .select_where(group, |deployment| deployment.team_id.is_none())
                .is_none()
        );
    }

    #[rstest]
    #[case::string("team_id: team-a", Some("team-a"))]
    #[case::null("team_id: null", None)]
    #[case::absent("id: x", None)]
    #[case::number("team_id: 7", Some("Number(7)"))]
    fn reads_the_owning_team_from_model_info(#[case] info: &str, #[case] expected: Option<&str>) {
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: m\n    litellm_params:\n      model: openai/m\n    model_info:\n      {info}\n"
        ))
        .unwrap();
        let router = Router::from_config(&config).unwrap();
        let group = router.group("m").unwrap();
        assert_eq!(
            group.deployments().next().unwrap().team_id.as_deref(),
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_unclassified_error_is_not_retried() {
        let router = router(&["a", "b", "c"], retries(2));

        let (tried, result) = send(&router, |_| Err(None)).await;

        assert_eq!(tried.len(), 1);
        assert_eq!(result.err(), Some(None));
    }

    #[rstest]
    #[case::bounded_by_num_retries(&["a", "b", "c", "d"], 1, 2)]
    #[case::bounded_by_deployments(&["a", "b"], 5, 2)]
    #[case::no_retries(&["a", "b"], 0, 1)]
    #[tokio::test]
    async fn retries_stop_at_num_retries_or_untried_deployments(
        #[case] models: &[&str],
        #[case] num_retries: u32,
        #[case] attempts: usize,
    ) {
        let router = router(models, retries(num_retries));

        let (tried, result) = send(&router, |_| Err(Some(Failure::Unavailable))).await;

        let distinct: BTreeMap<_, _> = tried.iter().cloned().collect();
        assert_eq!((tried.len(), distinct.len()), (attempts, attempts));
        assert_eq!(tried.last().map(|(_, last)| *last), Some(true));
        assert_eq!(result.err(), Some(Some(Failure::Unavailable)));
    }

    #[rstest]
    #[case::single_deployment(&["a"], 2, vec![true])]
    #[case::three_deployments(&["a", "b", "c"], 2, vec![false, false, true])]
    #[tokio::test]
    async fn last_marks_the_final_possible_attempt(
        #[case] models: &[&str],
        #[case] num_retries: u32,
        #[case] expected: Vec<bool>,
    ) {
        let router = router(models, retries(num_retries));

        let (tried, _) = send(&router, |_| Err(Some(Failure::Unavailable))).await;

        assert_eq!(
            tried.into_iter().map(|(_, last)| last).collect::<Vec<_>>(),
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_cooling_deployment_is_skipped_by_later_requests() {
        let router = router(
            &["bad", "good"],
            RoutingPolicy {
                cooldown: CooldownPolicy {
                    allowed_fails: Some(0),
                    cooldown_time: Duration::from_secs(60),
                },
                ..retries(1)
            },
        );
        let outcome = |model: &str| match model {
            "bad" => Err(Some(Failure::RateLimited)),
            _ => Ok("served"),
        };
        while send(&router, outcome).await.0[0].0 != "bad" {}

        let (tried, result) = send(&router, outcome).await;

        assert_eq!(tried, vec![("good".to_owned(), false)]);
        assert!(result.is_ok());
    }

    #[rstest]
    #[tokio::test]
    async fn every_deployment_cooling_still_serves() {
        let router = router(
            &["a"],
            RoutingPolicy {
                cooldown: CooldownPolicy {
                    allowed_fails: Some(0),
                    cooldown_time: Duration::from_secs(60),
                },
                ..retries(0)
            },
        );
        let _ = send(&router, |_| Err(Some(Failure::Unavailable))).await;

        let (tried, result) = send(&router, |_| Ok("served")).await;

        assert_eq!(tried.len(), 1);
        assert!(result.is_ok());
    }

    #[rstest]
    #[tokio::test]
    async fn least_busy_avoids_a_deployment_holding_a_call() {
        let router = router(
            &["a", "b"],
            RoutingPolicy {
                strategy: RoutingStrategy::LeastBusy,
                ..retries(0)
            },
        );
        let (first, held) = send(&router, |_| Ok("streaming")).await;

        let (second, _) = send(&router, |_| Ok("served")).await;

        assert_ne!(first[0].0, second[0].0);
        drop(held);
    }

    #[rstest]
    #[tokio::test]
    async fn simple_shuffle_spreads_requests_over_the_group() {
        let router = router(&["a", "b", "c", "d"], retries(0));
        let mut hits = BTreeMap::<String, usize>::new();
        for _ in 0..4_000 {
            let (tried, _) = send(&router, |_| Ok("served")).await;
            *hits.entry(tried[0].0.clone()).or_default() += 1;
        }

        assert_eq!(hits.len(), 4);
        assert!(
            hits.values().all(|count| (700..1300).contains(count)),
            "{hits:?}"
        );
    }
}
