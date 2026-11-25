//! The post-call spend counter fan-out's decisions: which `spend:*` counters one charged request
//! bumps.
//!
//! [`plan`] is pure. It names every counter a charge touches, grouped into the scopes whose failure
//! rules [`crate::charge`] keeps, and the auth cache entries a cold counter falls back to. Counters
//! held by a budget reservation are named too; the charge skips them once the reconcile settled
//! them, and charges them directly when it could not.

use std::collections::HashSet;

use aws_lc_rs::digest::{SHA256, digest};
use litellm_redis_store::{auth_cache::AuthObject, spend_counters::SpendCounter};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::charge::{CounterStep, WindowSource};

/// Reads a `spend:*` key as the counter it names, refusing a key that names none.
pub(crate) fn counter_key<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<SpendCounter, D::Error> {
    let key = String::deserialize(deserializer)?;
    SpendCounter::parse(&key).map_err(D::Error::custom)
}

/// [`counter_key`] for a list of keys.
pub(crate) fn counter_keys<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<SpendCounter>, D::Error> {
    Vec::<String>::deserialize(deserializer)?
        .iter()
        .map(|key| SpendCounter::parse(key).map_err(D::Error::custom))
        .collect()
}

/// Writes a counter as its `spend:*` key, the form [`counter_key`] reads.
pub(crate) fn counter_key_text<S: Serializer>(
    counter: &SpendCounter,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&counter.key())
}

/// [`counter_key_text`] for a list of counters.
pub(crate) fn counter_keys_text<S: Serializer>(
    counters: &[SpendCounter],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(counters.iter().map(SpendCounter::key))
}

/// A customer hierarchy budget counter the request was admitted against. Only the ones with a
/// `window_duration` are window counters the fan-out charges.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CustomerWindow {
    #[serde(deserialize_with = "counter_key", serialize_with = "counter_key_text")]
    pub counter_key: SpendCounter,
    pub entity_type: String,
    pub entity_id: String,
    #[serde(default)]
    pub window_duration: Option<String>,
    /// The start of the window the request falls in, ISO 8601; `None` when it is not valid.
    #[serde(default)]
    pub window_start: Option<String>,
}

/// A customer scope's lifetime or period counter and the spend its auth snapshot recorded.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CustomerScopeSpend {
    #[serde(deserialize_with = "counter_key", serialize_with = "counter_key_text")]
    pub counter_key: SpendCounter,
    pub spend: f64,
}

/// An entity whose budget windows live on its cached auth object. The strum string is the entity
/// label the window reseed reads spend logs by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, strum::IntoStaticStr)]
pub(crate) enum WindowedEntity {
    #[serde(rename = "key")]
    #[strum(serialize = "Key")]
    Key,
    #[serde(rename = "team")]
    #[strum(serialize = "Team")]
    Team,
    #[serde(rename = "user")]
    #[strum(serialize = "User")]
    User,
}

impl WindowedEntity {
    /// The counter of the entity's budget window named by `duration`.
    pub(crate) fn window(self, entity_id: &str, duration: &str) -> SpendCounter {
        let window = duration.to_owned();
        match self {
            Self::Key => SpendCounter::Key {
                token: entity_id.to_owned(),
                window: Some(window),
            },
            Self::Team => SpendCounter::Team {
                team_id: entity_id.to_owned(),
                window: Some(window),
            },
            Self::User => SpendCounter::UserWindow {
                user_id: entity_id.to_owned(),
                window,
            },
        }
    }
}

/// One step of a scope.
///
/// `Counter` warms a lifetime counter and charges it, falling back to the auth cache entries in
/// `sources`. `CachedWindows` reads the entity's budget windows from the auth cache and charges
/// each, or defers to the stored row when the entity is not cached. `CustomerScope` charges the
/// customer hierarchy counter at that index of the step's `customer_scopes`, seeded from its
/// admitted spend when cold. `CustomerWindow` charges the customer counter at that index of the
/// step's `customer_windows`. `Committed` seeds and charges each twin, paired with the counter it
/// twins.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Task {
    Counter {
        counter: SpendCounter,
        sources: Vec<AuthObject>,
    },
    CachedWindows(WindowSource),
    CustomerScope(usize),
    CustomerWindow(usize),
    Committed(Vec<(SpendCounter, SpendCounter)>),
}

/// A group of tasks for one entity. A sequential scope stops at its first failure and drops what it
/// already prepared. A concurrent scope runs every task and keeps each failure as its own item.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Scope {
    pub(crate) concurrent: bool,
    pub(crate) tasks: Vec<Task>,
}

impl Scope {
    fn sequential(tasks: impl IntoIterator<Item = Task>) -> Self {
        Self {
            concurrent: false,
            tasks: tasks.into_iter().collect(),
        }
    }

    pub(crate) fn concurrent(tasks: impl IntoIterator<Item = Task>) -> Self {
        Self {
            concurrent: true,
            tasks: tasks.into_iter().collect(),
        }
    }
}

/// Each tracked counter with the committed twin that takes this charge, none unless the cost is
/// positive.
fn committed_twins(
    tracked: &[SpendCounter],
    response_cost: Option<f64>,
) -> Vec<(SpendCounter, SpendCounter)> {
    match response_cost {
        Some(cost) if cost > 0.0 => tracked
            .iter()
            .filter_map(|base| Some((base.clone(), base.clone().committed()?)))
            .collect(),
        _ => Vec::new(),
    }
}

fn named(values: Option<&[Option<String>]>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .unwrap_or_default()
        .iter()
        .flatten()
        .filter(|name| !name.is_empty() && seen.insert(name.as_str()))
        .cloned()
        .collect()
}

/// The cost the fan-out charges, `None` when there is no cost or it is zero.
pub(crate) fn charged(response_cost: Option<f64>) -> Option<f64> {
    response_cost.filter(|cost| *cost != 0.0)
}

fn hashed_token(token: &str) -> String {
    if token.starts_with("sk-") {
        hex::encode(digest(&SHA256, token.as_bytes()))
    } else {
        token.to_owned()
    }
}

/// The label a customer window's spend is read by: the entity type, capitalized for the key, team
/// and user entities.
pub(crate) fn entity_label(entity_type: &str) -> String {
    match entity_type {
        "key" => "Key",
        "team" => "Team",
        "user" => "User",
        other => other,
    }
    .to_owned()
}

fn counter(counter: SpendCounter, sources: Vec<AuthObject>) -> Task {
    Task::Counter { counter, sources }
}

/// The scopes and tasks for one charge, in the order their increments join the pipeline. A scope
/// with no task is left out.
pub(crate) fn plan(step: &CounterStep) -> Vec<Scope> {
    let customer_scopes = (0..step.customer_scopes.len()).map(Task::CustomerScope);
    let customer_windows = step
        .customer_windows
        .iter()
        .enumerate()
        .filter(|(_, window)| window.window_duration.is_some())
        .map(|(index, _)| Task::CustomerWindow(index));
    let mut scopes = vec![Scope::concurrent(customer_scopes.chain(customer_windows))];
    if let Some(token) = &step.token {
        let hashed = hashed_token(token);
        scopes.push(Scope::sequential([
            counter(
                SpendCounter::Key {
                    token: hashed.clone(),
                    window: None,
                },
                vec![AuthObject::Key(hashed.clone())],
            ),
            Task::CachedWindows(WindowSource {
                entity_type: WindowedEntity::Key,
                entity_id: hashed.clone(),
                cache_key: hashed,
            }),
        ]));
    }
    if let Some(team_id) = &step.team_id {
        let cache_key = format!("team_id:{team_id}");
        scopes.push(Scope::sequential([
            counter(
                SpendCounter::Team {
                    team_id: team_id.clone(),
                    window: None,
                },
                vec![AuthObject::Team(team_id.clone())],
            ),
            Task::CachedWindows(WindowSource {
                entity_type: WindowedEntity::Team,
                entity_id: team_id.clone(),
                cache_key,
            }),
        ]));
    }
    if let (Some(user_id), Some(team_id)) = (&step.user_id, &step.team_id) {
        scopes.push(Scope::sequential([counter(
            SpendCounter::TeamMember {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            },
            vec![AuthObject::TeamMembershipReservation {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            }],
        )]));
    }
    if let Some(user_id) = &step.user_id {
        scopes.push(Scope::sequential([
            counter(
                SpendCounter::User(user_id.clone()),
                vec![AuthObject::User(user_id.clone())],
            ),
            Task::CachedWindows(WindowSource {
                entity_type: WindowedEntity::User,
                entity_id: user_id.clone(),
                cache_key: user_id.clone(),
            }),
        ]));
    }
    let end_user = step.end_user_id.iter().map(|end_user| {
        counter(
            SpendCounter::EndUser(end_user.clone()),
            vec![AuthObject::EndUser(end_user.clone())],
        )
    });
    let tags = named(step.tags.as_deref()).into_iter().map(|tag| {
        counter(
            SpendCounter::Tag(tag.clone()),
            vec![AuthObject::Tag(tag.clone())],
        )
    });
    scopes.push(Scope::concurrent(end_user.chain(tags)));
    scopes.push(Scope::concurrent(
        named(step.model_access_groups.as_deref())
            .into_iter()
            .map(|group| {
                counter(
                    SpendCounter::ModelAccessGroup(group.clone()),
                    vec![AuthObject::ModelAccessGroup(group.clone())],
                )
            }),
    ));
    if let Some(org_id) = &step.org_id {
        scopes.push(Scope::sequential([counter(
            SpendCounter::Org(org_id.clone()),
            vec![
                AuthObject::OrgWithBudget(org_id.clone()),
                AuthObject::Org(org_id.clone()),
            ],
        )]));
    }
    if let Some(project_id) = &step.project_id {
        scopes.push(Scope::sequential([counter(
            SpendCounter::Project(project_id.clone()),
            vec![AuthObject::Project(project_id.clone())],
        )]));
    }
    let twins = committed_twins(&step.committed_tracked, step.response_cost);
    if !twins.is_empty() {
        scopes.push(Scope::sequential([Task::Committed(twins)]));
    }
    scopes.retain(|scope| !scope.tasks.is_empty());
    scopes
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn keys(scopes: &[Scope]) -> Vec<Vec<String>> {
        scopes
            .iter()
            .map(|scope| {
                scope
                    .tasks
                    .iter()
                    .map(|task| match task {
                        Task::Counter { counter, .. } => counter.key(),
                        Task::CustomerScope(index) => format!("customer_scope:{index}"),
                        Task::CustomerWindow(index) => format!("customer:{index}"),
                        Task::CachedWindows(source) => format!("windows:{}", source.cache_key),
                        Task::Committed(twins) => twins
                            .iter()
                            .map(|(_, twin)| twin.key())
                            .collect::<Vec<_>>()
                            .join(","),
                    })
                    .collect()
            })
            .collect()
    }

    fn parsed(key: &str) -> SpendCounter {
        SpendCounter::parse(key).unwrap()
    }

    fn customer_window(key: &str, duration: Option<&str>) -> CustomerWindow {
        CustomerWindow {
            counter_key: parsed(key),
            entity_type: "customer_org".into(),
            entity_id: "c".into(),
            window_duration: duration.map(str::to_owned),
            window_start: None,
        }
    }

    fn full_step() -> CounterStep {
        CounterStep {
            token: Some("hashed".into()),
            team_id: Some("t1".into()),
            user_id: Some("u1".into()),
            org_id: Some("o1".into()),
            end_user_id: Some("eu1".into()),
            project_id: Some("p1".into()),
            tags: Some(vec![
                Some("a".into()),
                Some(String::new()),
                None,
                Some("b".into()),
                Some("a".into()),
            ]),
            model_access_groups: Some(vec![Some("g".into()), None, Some("g".into())]),
            response_cost: Some(1.5),
            committed_tracked: vec![parsed("spend:key:hashed")],
            customer_windows: vec![
                customer_window("spend:customer_org:c:window:1mo@x", Some("1mo")),
                customer_window("spend:customer_org:c:lifetime", None),
            ],
            customer_scopes: vec![CustomerScopeSpend {
                counter_key: parsed("spend:customer_org:c:lifetime"),
                spend: 2.0,
            }],
            ..CounterStep::default()
        }
    }

    #[test]
    fn plan_orders_scopes_and_counters_like_the_python_fan_out() {
        assert_eq!(
            keys(&plan(&full_step())),
            vec![
                vec!["customer_scope:0".to_owned(), "customer:0".into()],
                vec!["spend:key:hashed".into(), "windows:hashed".into()],
                vec!["spend:team:t1".into(), "windows:team_id:t1".into()],
                vec!["spend:team_member:u1:t1".into()],
                vec!["spend:user:u1".into(), "windows:u1".into()],
                vec![
                    "spend:end_user:eu1".into(),
                    "spend:tag:a".into(),
                    "spend:tag:b".into()
                ],
                vec!["spend:model_access_group:g".into()],
                vec!["spend:org:o1".into()],
                vec!["spend:project:p1".into()],
                vec!["spend:committed:key:hashed".into()],
            ]
        );
    }

    #[rstest]
    fn source_cache_keys_name_the_cached_auth_objects() {
        let sources: Vec<Vec<String>> = plan(&full_step())
            .iter()
            .flat_map(|scope| &scope.tasks)
            .filter_map(|task| match task {
                Task::Counter { sources, .. } => {
                    Some(sources.iter().map(AuthObject::key).collect())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            sources,
            vec![
                vec!["hashed".to_owned()],
                vec!["team_id:t1".into()],
                vec!["team_membership:u1:t1".into()],
                vec!["u1".into()],
                vec!["end_user_id:eu1".into()],
                vec!["tag:a".into()],
                vec!["tag:b".into()],
                vec!["model_access_group:g".into()],
                vec!["org_id:o1:with_budget".into(), "org_id:o1".into()],
                vec!["project_id:p1".into()],
            ]
        );
    }

    #[rstest]
    fn a_raw_key_is_hashed_before_it_names_a_counter() {
        let scopes = plan(&CounterStep {
            token: Some("sk-raw-key".into()),
            ..CounterStep::default()
        });
        let hashed = "7e829077433c1b2fa414ae023e40914235063fcb5cf3cb0797e835c509a9b02a".to_owned();
        assert_eq!(
            scopes[0].tasks[0],
            Task::Counter {
                counter: SpendCounter::Key {
                    token: hashed.clone(),
                    window: None,
                },
                sources: vec![AuthObject::Key(hashed)],
            }
        );
    }

    #[rstest]
    #[case::empty_tags(Some(vec![]), None)]
    #[case::empty_groups(None, Some(vec![]))]
    #[case::unnamed(Some(vec![None, Some(String::new())]), Some(vec![None]))]
    #[case::none(None, None)]
    fn a_scope_with_nothing_to_charge_is_left_out(
        #[case] tags: Option<Vec<Option<String>>>,
        #[case] model_access_groups: Option<Vec<Option<String>>>,
    ) {
        let scopes = plan(&CounterStep {
            tags,
            model_access_groups,
            ..CounterStep::default()
        });
        assert!(scopes.is_empty());
    }

    #[rstest]
    #[case::charged(Some(0.5), vec!["spend:committed:key:k", "spend:committed:org:o"])]
    #[case::zero(Some(0.0), vec![])]
    #[case::negative(Some(-1.0), vec![])]
    #[case::unknown(None, vec![])]
    fn committed_twins_only_take_a_positive_charge(
        #[case] cost: Option<f64>,
        #[case] expected: Vec<&str>,
    ) {
        let tracked = [
            parsed("spend:key:k"),
            parsed("spend:tag:t"),
            parsed("spend:org:o"),
        ];
        let twins: Vec<String> = committed_twins(&tracked, cost)
            .iter()
            .map(|(_, twin)| twin.key())
            .collect();
        assert_eq!(twins, expected);
    }

    #[rstest]
    #[case::none(None, None)]
    #[case::zero(Some(0.0), None)]
    #[case::negative_zero(Some(-0.0), None)]
    #[case::refund(Some(-2.0), Some(-2.0))]
    #[case::cost(Some(1e-9), Some(1e-9))]
    fn only_a_nonzero_cost_is_charged(#[case] cost: Option<f64>, #[case] expected: Option<f64>) {
        assert_eq!(charged(cost), expected);
    }

    #[rstest]
    #[case::key(WindowedEntity::Key, "spend:key:id:window:1d", "Key")]
    #[case::team(WindowedEntity::Team, "spend:team:id:window:1d", "Team")]
    #[case::user(WindowedEntity::User, "spend:user_window:id:1d", "User")]
    fn each_budget_window_names_its_counter(
        #[case] entity: WindowedEntity,
        #[case] key: &str,
        #[case] label: &str,
    ) {
        assert_eq!(entity.window("id", "1d").key(), key);
        assert_eq!(<&str>::from(entity), label);
    }

    #[rstest]
    #[case::customer_window(
        r#"{"counter_key": "spend:customer_org:c:window:1mo", "entity_type": "customer_org", "entity_id": "c"}"#,
        true
    )]
    #[case::not_a_counter(
        r#"{"counter_key": "budget:c", "entity_type": "customer_org", "entity_id": "c"}"#,
        false
    )]
    fn a_customer_window_arrives_with_its_parsed_counter(#[case] json: &str, #[case] ok: bool) {
        let window = serde_json::from_str::<CustomerWindow>(json);
        assert_eq!(window.is_ok(), ok);
    }
}
