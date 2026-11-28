//! The budget checks the proxy runs while it authenticates a call, for a native call. Each budget
//! the caller holds is read as `get_current_spend` reads it, decided by
//! `litellm_callbacks_budgets::decide` as Python decides it, and the first budget the caller is
//! over refuses the call with Python's 422 `budget_exceeded` error, in Python's order: the key
//! path checks the member's own budget and then the key's, then `common_checks` the project's,
//! then the team's, organization's, user's, team member's and end user's.
//!
//! A call to a free model is checked against no budget, as `request_skips_budget_checks`
//! (`litellm/proxy/auth/auth_checks.py`) skips it. The model is the one the body names before
//! anything routes it, as `get_model_from_request` reads it, and [`FreeModels`] says whether it is
//! free. An unknown model, or one whose rates are missing, is checked as paid.
//! No throttle or alert is decided, since a config that sets `budget_exceeded_throttle_percentage`
//! or `alerting` goes to Python, and a decision of either fails the call rather than letting it
//! through unchecked.
//!
//! Only the budgets `budgets::counters` plans are checked. A `budget_limits` window, soft budget,
//! tag, model access group, per-model budget, temporary increase, throttle, budget alert or the
//! proxy-wide budget is never checked here, since the key column, request or setting that asks
//! for one sends the call to Python first.

use std::sync::Arc;

use arc_swap::ArcSwap;
use axum::http::StatusCode;
use futures_util::future::{BoxFuture, join_all};
use litellm_callbacks::{
    Abort, AbortError, BudgetPlanCounter, Callback, CallbackError, Field, FieldSet, Hook,
    HookInput, HookOutput, Manifest, Patch, Selection, Subscription, Verdict, typed,
};
use litellm_callbacks_budgets::{
    Admission, BudgetCounterSpend, BudgetPlan, BudgetReservation, Budgets, Check, Decision,
    PlannedCounter, PyNumber, SpendReader, decide, post_call,
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::Credential;
use litellm_gateway_inference::{CallType, Refusal, record};
use litellm_model_catalog::CostMap;
use litellm_redis_store::{
    Number,
    spend_counters::{LostSeed, SpendCounter},
};
use litellm_router_core::Router;
use parking_lot::Mutex;
use serde_json::{Map, Value};

use super::{Registered, budget_price, budgets, hosted};
use crate::host::{FreeModels, NativeRows};

pub(crate) const NAME: &str = "budget_check";
/// `ProxyErrorTypes.budget_exceeded`.
const BUDGET_EXCEEDED: &str = "budget_exceeded";
/// `litellm.budget_exceeded_status_code`. A config that sets it goes to Python.
const BUDGET_EXCEEDED_STATUS: u16 = 422;

/// Refuses a native call whose caller is over one of its budgets.
pub struct BudgetCheck {
    manifest: Manifest,
    spend: SpendReader,
    free: Arc<FreeModels>,
    reserve: Option<Arc<Reserve>>,
}

/// The live routing and spend services used by pre-call reservations.
pub(super) struct Reserve {
    pub(super) router: Router,
    pub(super) prices: Arc<ArcSwap<CostMap>>,
    pub(super) budgets: Arc<Budgets>,
    pub(super) host: Arc<dyn hosted::Answers>,
}

/// Owns a hold until the spend log finalizes it or the request is dropped.
struct Hold {
    reservation: Option<BudgetReservation>,
    budget: Arc<Reserve>,
    disconnected: bool,
}

impl Hold {
    async fn finish(mut self, actual: f64) -> Option<BudgetReservation> {
        let budget = self.budget.clone();
        let mut reservation = self.reservation.take()?;
        hosted::run(budget.host.as_ref(), {
            let budgets = budget.budgets.clone();
            move |host| async move {
                if let Err(error) =
                    post_call::settle_cost(&budgets, &host, &mut reservation, &[], actual).await
                {
                    tracing::error!(%error, "Failed to settle request budget reservation");
                    post_call::invalidate(budgets.store(), &reservation).await;
                    reservation.finalized = Some(true);
                }
                reservation
            }
        })
        .await
        .ok()
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        let Some(reservation) = self.reservation.take() else {
            return;
        };
        let actual = if self.disconnected {
            reservation.input_cost.unwrap_or_default()
        } else {
            0.0
        };
        let hold = Self {
            reservation: Some(reservation),
            budget: self.budget.clone(),
            disconnected: false,
        };
        litellm_call_scope::spawn_in_call(hold.finish(actual));
    }
}

/// What a budget is checked as. The variants are in the order Python checks them.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Entity {
    /// `team_member_budget`: the member's own positive budget, checked before the key's.
    OwnMember {
        user_id: String,
        team_id: String,
    },
    Key {
        token: String,
        alias: Option<String>,
        name: Option<String>,
    },
    Project(String),
    Team(String),
    Organization(String),
    User(String),
    /// `_check_team_member_budget`: the member's own budget or its team's default.
    Member {
        user_id: String,
        team_id: String,
    },
    EndUser(String),
}

impl Entity {
    fn check(&self, spend: f64, max_budget: f64) -> Check {
        let (spend, max_budget) = (PyNumber::Float(spend), PyNumber::Float(max_budget));
        match self.clone() {
            Self::OwnMember { user_id, team_id } => Check::TeamMemberPositive {
                spend,
                max_budget,
                user_id: Some(user_id),
                team_id: Some(team_id),
            },
            Self::Key { token, alias, name } => Check::KeyMax {
                spend,
                max_budget,
                token: Some(token),
                key_alias: alias,
                key_name: name,
                throttle_pct: None,
            },
            Self::Project(project_id) => Check::ProjectMax {
                spend,
                max_budget,
                project_id: Some(project_id),
            },
            Self::Team(team_id) => Check::TeamMax {
                spend,
                max_budget,
                team_id: Some(team_id),
            },
            Self::Organization(org_id) => Check::Organization {
                spend,
                max_budget,
                org_id,
            },
            Self::User(user_id) => Check::User {
                spend,
                max_budget,
                user_id: Some(user_id),
            },
            Self::Member { user_id, team_id } => Check::TeamMember {
                spend,
                max_budget,
                user_id: Some(user_id),
                team_id: Some(team_id),
            },
            Self::EndUser(end_user_id) => Check::EndUser {
                spend,
                max_budget,
                end_user_id: Some(end_user_id),
            },
        }
    }
}

/// One budget the caller holds: the counter its spend is read from, and what it is checked as.
struct Held {
    planned: PlannedCounter,
    counter: SpendCounter,
    fallback_spend: f64,
    max_budget: f64,
    entities: Vec<Entity>,
}

/// The budgets one request's caller is checked against.
pub struct Budgeted {
    held: Vec<Held>,
    reservation: Arc<Mutex<Option<Hold>>>,
    team_id: Option<String>,
}

impl Budgeted {
    pub(super) fn response_headers(&self) {
        if let Some(hold) = self.reservation.lock().as_mut() {
            hold.disconnected = true;
        }
    }

    pub(super) async fn finish(&self, finalized: bool, actual: f64) {
        let hold = self.reservation.lock().take();
        if let Some(mut hold) = hold {
            if finalized {
                hold.reservation = None;
            } else if let Ok(Some(reservation)) =
                litellm_call_scope::spawn_in_call(hold.finish(actual)).await
            {
                record::write(|entries| entries.budget_reservation = Some(reservation));
            }
        }
    }
}

fn entities(identity: &Authenticated<NativeRows>, counter: &SpendCounter) -> Vec<Entity> {
    match counter.clone() {
        SpendCounter::Key {
            token,
            window: None,
        } => {
            let key = match &identity.credential {
                Credential::Key { key } => Some(key),
                Credential::MasterKey => None,
            };
            vec![Entity::Key {
                token,
                alias: key.and_then(|key| key.alias()).map(str::to_owned),
                name: key.and_then(|key| key.name()).map(str::to_owned),
            }]
        }
        SpendCounter::Team {
            team_id,
            window: None,
        } => vec![Entity::Team(team_id)],
        SpendCounter::TeamMember { user_id, team_id } => {
            let own = identity
                .affiliation
                .holder
                .membership()
                .and_then(|membership| membership.budget.max)
                .is_some_and(|max_budget| max_budget > 0.0);
            let member = Entity::Member {
                user_id: user_id.clone(),
                team_id: team_id.clone(),
            };
            if own {
                vec![Entity::OwnMember { user_id, team_id }, member]
            } else {
                vec![member]
            }
        }
        SpendCounter::User(user_id) => vec![Entity::User(user_id)],
        SpendCounter::EndUser(end_user_id) => vec![Entity::EndUser(end_user_id)],
        SpendCounter::Org(org_id) => vec![Entity::Organization(org_id)],
        SpendCounter::Project(project_id) => vec![Entity::Project(project_id)],
        _ => Vec::new(),
    }
}

fn budgeted(identity: &Authenticated<NativeRows>) -> Budgeted {
    let now = jiff::Timestamp::now();
    let held = budgets::counters(identity, budgets::Settings::default(), now)
        .into_iter()
        .filter_map(|planned| {
            let counter = SpendCounter::parse(&planned.counter_key).ok()?;
            Some(Held {
                entities: entities(identity, &counter),
                counter,
                fallback_spend: planned.fallback_spend,
                max_budget: planned.max_budget,
                planned,
            })
        })
        .collect();
    Budgeted {
        held,
        reservation: Default::default(),
        team_id: identity
            .affiliation
            .holder
            .team()
            .map(|team| team.id().to_owned()),
    }
}

/// Python's `BudgetExceededError` for an over-budget caller, as an abort.
fn over_budget(fields: Map<String, Value>) -> Abort {
    let message = fields
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    Abort {
        code: Some(BUDGET_EXCEEDED_STATUS.to_string()),
        error: Some(AbortError {
            kind: BUDGET_EXCEEDED.to_owned(),
            fields,
        }),
        ..Abort::new(BUDGET_EXCEEDED_STATUS, message)
    }
}

/// The `ProxyException` Python answers an over-budget caller with, for an abort this callback
/// made.
pub(crate) fn refusal(abort: &Abort) -> Option<Refusal> {
    abort
        .error
        .as_ref()
        .filter(|error| error.kind == BUDGET_EXCEEDED)?;
    Some(Refusal {
        kind: Some(BUDGET_EXCEEDED.to_owned()),
        code: abort.code.clone(),
        at_authentication: true,
        ..Refusal::new(
            StatusCode::from_u16(abort.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            abort.message.clone(),
        )
    })
}

/// What `decision` does to the call, `None` for one that lets it go on to the next budget.
fn verdict(decision: Decision) -> Option<HookOutput> {
    match decision {
        Decision::Skip | Decision::Pass => None,
        Decision::Refuse(refusal) => Some(match serde_json::to_value(refusal) {
            Ok(Value::Object(fields)) => Ok(Some(Verdict::Abort(over_budget(fields)))),
            Ok(_) => Err(CallbackError::Failed(
                "a budget refusal is not an object".to_owned(),
            )),
            Err(error) => Err(CallbackError::Failed(error.to_string())),
        }),
        Decision::Throttle { .. } | Decision::Alert => Some(Err(CallbackError::Failed(format!(
            "the gateway does not answer a budget decision of {decision:?}"
        )))),
    }
}

fn manifest() -> Manifest {
    Manifest {
        selection: Selection::named(NAME),
        subscriptions: vec![Subscription {
            reads: FieldSet::of(&[Field::Model, Field::RequestBody]),
            writes: FieldSet::of(&[Field::BudgetReservation]),
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        }],
        has_request_state: true,
    }
}

impl BudgetCheck {
    pub fn new(spend: SpendReader, free: Arc<FreeModels>) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            spend,
            free,
            reserve: None,
        })
    }

    pub(super) fn with_reservation(
        spend: SpendReader,
        free: Arc<FreeModels>,
        reserve: Reserve,
    ) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            spend,
            free,
            reserve: Some(Arc::new(reserve)),
        })
    }

    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        Registered::for_caller(self.clone(), budgeted)
    }

    async fn pre_call(
        &self,
        model: Option<&str>,
        body: Option<&Map<String, Value>>,
        budgeted: &Budgeted,
    ) -> HookOutput {
        if model.is_some_and(|model| self.free.is_free(model)) {
            return Ok(None);
        }
        let spends = join_all(budgeted.held.iter().map(|held| {
            self.spend
                .current(&held.counter, held.fallback_spend, Some(held.max_budget))
        }))
        .await;
        let mut checks = budgeted
            .held
            .iter()
            .zip(spends.iter().copied())
            .flat_map(|(held, spend)| {
                held.entities
                    .iter()
                    .map(move |entity| (entity, spend, held.max_budget))
            })
            .collect::<Vec<_>>();
        checks.sort_by_key(|&(entity, ..)| entity);
        if let Some(outcome) = checks.into_iter().find_map(|(entity, spend, max_budget)| {
            verdict(decide(&entity.check(spend, max_budget)))
        }) {
            return outcome;
        }
        let Some((reserve, body)) = self.reserve.as_ref().zip(body) else {
            return Ok(None);
        };
        if budgeted.held.is_empty()
            || record::read(|entries| entries.call_type) != Some(Some(CallType::AnthropicMessages))
        {
            return Ok(None);
        }
        let Some((bound, input_tokens)) = budget_price::estimate(
            &reserve.router,
            &reserve.prices.load_full(),
            body,
            budgeted.team_id.as_deref(),
        )
        .await
        else {
            return Ok(None);
        };
        let Some(max) = bound.max.filter(|max| max.is_finite() && *max > 0.0) else {
            return Ok(None);
        };
        let pairs = budgeted
            .held
            .iter()
            .zip(spends)
            .filter(|(held, _)| {
                matches!(
                    held.counter,
                    SpendCounter::TeamMember { .. } | SpendCounter::User(_)
                )
            })
            .map(|(held, spend)| (held.counter.clone(), held.planned.clone(), spend))
            .collect::<Vec<_>>();
        if pairs.is_empty() {
            return Ok(None);
        }
        let reserve = reserve.clone();
        let held = budgeted.reservation.clone();
        let admission = litellm_call_scope::spawn_in_call(async move {
            for (counter, _, spend) in &pairs {
                if *spend > 0.0 {
                    let _ = reserve
                        .budgets
                        .store()
                        .seed(counter, Number::Float(*spend), LostSeed::Answer)
                        .await;
                }
            }
            let observed = pairs
                .iter()
                .map(|(_, plan, spend)| BudgetCounterSpend {
                    counter_key: Some(plan.counter_key.clone()),
                    spend: Some(*spend),
                })
                .collect::<Vec<_>>();
            let plan = BudgetPlan {
                counters: Some(
                    pairs
                        .into_iter()
                        .map(|(_, plan, _)| BudgetPlanCounter {
                            counter_key: Some(plan.counter_key),
                            max_budget: Some(plan.max_budget),
                            fallback_spend: Some(plan.fallback_spend),
                            entity_type: Some(plan.entity_type),
                            entity_id: Some(plan.entity_id),
                            spend_log_entity_id: plan.spend_log_entity_id,
                        })
                        .collect(),
                ),
                reservation_cost: Some(max),
                input_cost: Some(bound.input_floor.unwrap_or_default()),
                input_tokens,
                status_code: Some(BUDGET_EXCEEDED_STATUS),
                ..BudgetPlan::default()
            };
            let admission = reserve.budgets.admit(plan, &observed).await?;
            if let Admission::Held { reservation } = &admission {
                *held.lock() = Some(Hold {
                    budget: reserve,
                    reservation: Some(reservation.clone()),
                    disconnected: false,
                });
            }
            Ok::<_, String>(admission)
        })
        .await
        .map_err(|error| CallbackError::Failed(error.to_string()))?
        .map_err(CallbackError::Failed)?;
        match admission {
            Admission::Skipped => Ok(None),
            Admission::Held { reservation } => {
                let value = serde_json::to_value(reservation)
                    .map_err(|error| CallbackError::Failed(error.to_string()))?;
                Ok(Some(Verdict::Patch(
                    Patch::default().set(Field::BudgetReservation, value),
                )))
            }
            Admission::Refused {
                kind: BUDGET_EXCEEDED,
                fields,
            } => Ok(Some(Verdict::Abort(over_budget(fields)))),
            Admission::Refused { kind, .. } => Err(CallbackError::Failed(kind.to_owned())),
        }
    }
}

impl Callback for BudgetCheck {
    type State = Budgeted;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Budgeted>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match (input.hook, state) {
                (Hook::PreCall, Some(budgeted)) => {
                    let model = input.fields.get::<typed::Model>()?;
                    let body = input.fields.get::<typed::RequestBody>()?;
                    self.pre_call(
                        model.as_deref(),
                        body.as_ref().and_then(Value::as_object),
                        budgeted,
                    )
                    .await
                }
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use litellm_callbacks_budgets::RowSpend;
    use litellm_gateway_auth_core::{
        Affiliation, Caller, Holder, Named, Team,
        record::{
            Budget, EndUserRecord, MembershipRecord, OrgRecord, ProjectRecord, TeamRecord,
            UserRecord,
        },
    };
    use litellm_gateway_inference::{
        CallHooks, Error, HookError,
        record::{self, CallRecord},
    };
    use litellm_redis_store::spend_counters::SpendCounters;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::{callbacks::Callbacks, host::Key};

    /// A caller's budgets have no rows, so each counter starts from the spend its plan read.
    struct NoRow;

    impl RowSpend for NoRow {
        fn stored<'a>(&'a self, _: &'a SpendCounter) -> BoxFuture<'a, Option<f64>> {
            Box::pin(async { None })
        }
    }

    const MAX: f64 = 10.0;

    fn budget(max: f64, spend: f64) -> Budget {
        Budget {
            max: Some(max),
            spend,
        }
    }

    /// The budget each of a caller's entities holds, `None` for one it holds none of. A caller
    /// with a user budget holds a personal key, since a team key skips its owner's budget.
    #[derive(Clone, Copy, Default)]
    struct Holds {
        key: Option<Budget>,
        team: Option<Budget>,
        own_member: Option<Budget>,
        member_default: Option<Budget>,
        user: Option<Budget>,
        end_user: Option<Budget>,
        org: Option<Budget>,
        project: Option<Budget>,
    }

    fn key(held: Budget) -> Holds {
        Holds {
            key: Some(held),
            ..Holds::default()
        }
    }

    fn team(held: Budget) -> Holds {
        Holds {
            team: Some(held),
            ..Holds::default()
        }
    }

    fn own_member(held: Budget) -> Holds {
        Holds {
            own_member: Some(held),
            ..Holds::default()
        }
    }

    fn member_default(held: Budget) -> Holds {
        Holds {
            member_default: Some(held),
            ..Holds::default()
        }
    }

    fn user(held: Budget) -> Holds {
        Holds {
            user: Some(held),
            ..Holds::default()
        }
    }

    fn end_user(held: Budget) -> Holds {
        Holds {
            end_user: Some(held),
            ..Holds::default()
        }
    }

    fn org(held: Budget) -> Holds {
        Holds {
            org: Some(held),
            ..Holds::default()
        }
    }

    fn project(held: Budget) -> Holds {
        Holds {
            project: Some(held),
            ..Holds::default()
        }
    }

    fn caller(holds: Holds) -> Authenticated<NativeRows> {
        caller_with(Key::stored("hash", None), holds)
    }

    fn caller_with(key: Key, holds: Holds) -> Authenticated<NativeRows> {
        let stored_user = Named::Stored(UserRecord {
            user_id: "u1".to_owned(),
            budget: holds.user.unwrap_or_default(),
            ..UserRecord::default()
        });
        let holder = match holds.user {
            Some(_) => Holder::User(stored_user),
            None => Holder::Member {
                user: stored_user,
                team: Team::Row(TeamRecord {
                    team_id: "t1".to_owned(),
                    budget: holds.team.unwrap_or_default(),
                    member_max_budget: holds.member_default.and_then(|held| held.max),
                    ..TeamRecord::default()
                }),
                membership: Some(MembershipRecord {
                    user_id: "u1".to_owned(),
                    team_id: "t1".to_owned(),
                    budget: holds.own_member.unwrap_or(Budget {
                        max: None,
                        spend: holds.member_default.map_or(0.0, |held| held.spend),
                    }),
                    ..MembershipRecord::default()
                }),
            },
        };
        Authenticated::from(Caller {
            credential: Credential::Key {
                key: key.with_budget(holds.key.unwrap_or_default()),
            },
            affiliation: Affiliation {
                holder,
                org: holds.org.map(|budget| {
                    Named::Stored(OrgRecord {
                        organization_id: "o1".to_owned(),
                        budget,
                        ..OrgRecord::default()
                    })
                }),
                project: holds.project.map(|budget| {
                    Named::Stored(ProjectRecord {
                        project_id: "p1".to_owned(),
                        budget,
                        ..ProjectRecord::default()
                    })
                }),
            },
            end_user: holds
                .end_user
                .map_or_else(EndUserRecord::default, |budget| EndUserRecord {
                    end_user_id: Some("e1".to_owned()),
                    found: true,
                    budget,
                    ..EndUserRecord::default()
                }),
            customer_hierarchy: None,
        })
    }

    fn check(store: &Arc<SpendCounters>, free: FreeModels) -> Arc<BudgetCheck> {
        BudgetCheck::new(
            SpendReader::new(store.clone(), Arc::new(NoRow)),
            Arc::new(free),
        )
    }

    /// The status, entity type and message the call is refused with.
    fn refused(abort: &Abort) -> (StatusCode, String, String) {
        let refusal = refusal(abort).expect("a budget refusal");
        let entity_type = abort
            .error
            .as_ref()
            .and_then(|error| error.fields.get("entity_type"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        (refusal.status, entity_type, refusal.message)
    }

    /// What one pre-call check of `identity` refuses a call to a paid model with, `None` when it
    /// lets it on.
    async fn checked(identity: &Authenticated<NativeRows>) -> Option<(StatusCode, String, String)> {
        let check = check(&Arc::new(SpendCounters::new(None)), FreeModels::default());
        let verdict = check
            .pre_call(Some("held-model"), None, &budgeted(identity))
            .await
            .unwrap();
        match verdict {
            None => None,
            Some(Verdict::Abort(abort)) => Some(refused(&abort)),
            Some(other) => panic!("the check answered {other:?}"),
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Spent {
        Under,
        Equal,
        Over,
    }

    #[rstest]
    #[case::key(key, "key", "Key=key ", true)]
    #[case::team(team, "team", "Team=t1 ", false)]
    #[case::own_member(own_member, "team_member", "TeamMember=u1:t1 ", true)]
    #[case::member_default(member_default, "team_member", "User=u1 in Team=t1 ", true)]
    #[case::user(user, "user", "User=u1 over budget", true)]
    #[case::end_user(end_user, "end_user", "End User=e1 over budget", false)]
    #[case::organization(org, "organization", "Organization=o1 ", true)]
    #[case::project(project, "project", "Project=p1 ", true)]
    #[tokio::test(flavor = "multi_thread")]
    async fn each_budget_refuses_by_pythons_comparison(
        #[case] holds: fn(Budget) -> Holds,
        #[case] entity_type: &str,
        #[case] named: &str,
        #[case] refuses_at_its_budget: bool,
        #[values(Spent::Under, Spent::Equal, Spent::Over)] spent: Spent,
    ) {
        let (spend, refuses) = match spent {
            Spent::Under => (MAX - 0.5, false),
            Spent::Equal => (MAX, refuses_at_its_budget),
            Spent::Over => (MAX + 0.5, true),
        };

        let refusal = checked(&caller(holds(budget(MAX, spend)))).await;

        assert_eq!(
            refusal.as_ref().map(|(status, kind, message)| (
                *status,
                kind.as_str(),
                message.contains(named)
            )),
            refuses.then_some((StatusCode::UNPROCESSABLE_ENTITY, entity_type, true)),
            "{refusal:?}"
        );
    }

    #[rstest]
    #[case::a_key_at_zero(key(budget(0.0, 0.0)), Some("key"))]
    #[case::a_user_below_zero(user(budget(-1.0, 0.0)), Some("user"))]
    #[case::an_organization_at_zero(org(budget(0.0, 0.0)), Some("organization"))]
    #[case::a_team_at_zero(team(budget(0.0, 0.0)), None)]
    #[case::an_end_user_at_zero(end_user(budget(0.0, 0.0)), None)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_budget_of_zero_or_less_is_checked(
        #[case] holds: Holds,
        #[case] refused_as: Option<&str>,
    ) {
        let refusal = checked(&caller(holds)).await;

        assert_eq!(
            refusal.as_ref().map(|(_, kind, _)| kind.as_str()),
            refused_as
        );
    }

    #[rstest]
    #[case::a_key_at_nan(key(budget(f64::NAN, 100.0)))]
    #[case::a_key_at_infinity(key(budget(f64::INFINITY, 100.0)))]
    #[case::a_team_at_nan(team(budget(f64::NAN, 100.0)))]
    #[case::an_own_member_budget_at_nan(own_member(budget(f64::NAN, 100.0)))]
    #[case::a_member_default_at_infinity(member_default(budget(f64::INFINITY, 100.0)))]
    #[case::a_user_at_infinity(user(budget(f64::INFINITY, 100.0)))]
    #[case::an_end_user_at_nan(end_user(budget(f64::NAN, 100.0)))]
    #[case::an_organization_at_infinity(org(budget(f64::INFINITY, 100.0)))]
    #[case::a_project_at_infinity(project(budget(f64::INFINITY, 100.0)))]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_budget_that_is_not_a_finite_number_refuses_nothing(#[case] holds: Holds) {
        assert_eq!(checked(&caller(holds)).await, None);
    }

    fn over() -> Option<Budget> {
        Some(budget(MAX, MAX + 1.0))
    }

    #[rstest]
    #[case::the_own_member_budget_before_the_key(Holds { own_member: over(), key: over(), ..Holds::default() }, "TeamMember=u1:t1 ")]
    #[case::the_key_before_the_project(Holds { project: over(), key: over(), ..Holds::default() }, "Key=key ")]
    #[case::the_project_before_the_team(Holds { team: over(), project: over(), ..Holds::default() }, "Project=p1 ")]
    #[case::the_team_before_the_organization(Holds { org: over(), team: over(), ..Holds::default() }, "Team=t1 ")]
    #[case::the_organization_before_the_member(Holds { member_default: over(), org: over(), ..Holds::default() }, "Organization=o1 ")]
    #[case::the_organization_before_the_user(Holds { user: over(), org: over(), ..Holds::default() }, "Organization=o1 ")]
    #[case::the_user_before_the_end_user(Holds { end_user: over(), user: over(), ..Holds::default() }, "User=u1 over budget")]
    #[case::the_member_before_the_end_user(Holds { end_user: over(), member_default: over(), ..Holds::default() }, "User=u1 in Team=t1 ")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_caller_over_two_budgets_is_refused_by_the_one_python_checks_first(
        #[case] holds: Holds,
        #[case] named: &str,
    ) {
        let refusal = checked(&caller(holds)).await;

        assert!(
            refusal
                .as_ref()
                .is_some_and(|(.., message)| message.contains(named)),
            "{refusal:?}"
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_key_is_named_by_its_alias_and_masked_name() {
        let identity = caller_with(
            Key::stored("hash", Some("my-alias")).with_name("sk-...abcd"),
            key(budget(MAX, MAX)),
        );

        let refusal = checked(&identity).await;

        assert_eq!(
            refusal.map(|(.., message)| message),
            Some(
                "Budget has been exceeded! Key=my-alias (sk-...abcd) Current cost: 10.0, Max budget: 10.0"
                    .to_owned()
            )
        );
    }

    const PRICED: &str = "model_list:
  - model_name: held-model
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
  - model_name: declared-free
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
      output_cost_per_token: 0
  - model_name: free-input-only
    litellm_params:
      model: anthropic/held-model
      api_key: provider-key
    model_info:
      input_cost_per_token: 0
  - model_name: free-by-the-cost-map
    litellm_params:
      model: anthropic/free-model
      api_key: provider-key
";

    /// What the callback stage, with the check registered as the gateway registers it, refuses a
    /// call to `model` from a caller with `key_budget` with.
    async fn refused_by_the_stage(key_budget: Budget, model: &str) -> Option<Refusal> {
        let store = Arc::new(SpendCounters::new(None));
        let free = FreeModels::of(PRICED).await;
        let callbacks = Arc::new(Callbacks::new(vec![check(&store, free).registered()]).unwrap());
        let Value::Object(body) = json!({"model": model, "max_tokens": 16, "messages": []}) else {
            unreachable!("the body is an object")
        };
        let body: Map<String, Value> = body;
        record::scope(CallRecord::new(), async {
            let call = callbacks.begin(&caller(key(key_budget)));
            match call.pre_call(body).await {
                Ok(_) => None,
                Err(HookError::Refused(refusal)) => Some(refusal),
                Err(HookError::Failed(error)) => panic!("the call failed: {error}"),
            }
        })
        .await
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_stage_refuses_with_pythons_budget_exceeded_error() {
        let refusal = refused_by_the_stage(budget(MAX, MAX), "held-model").await;

        assert_eq!(
            refusal.map(|refusal| (refusal.status, Error::Refused(refusal).body(None).to_string())),
            Some((
                StatusCode::UNPROCESSABLE_ENTITY,
                r#"{"type":"error","error":{"type":"invalid_request_error","message":"Budget has been exceeded! Key=key Current cost: 10.0, Max budget: 10.0","litellm_error_type":"budget_exceeded"}}"#.to_owned(),
            ))
        );
    }

    #[rstest]
    #[case::a_free_model_is_served("declared-free", false)]
    #[case::an_unknown_model_is_checked_as_paid("no-such-model", true)]
    #[case::a_free_input_with_a_paid_output_is_paid("free-input-only", true)]
    #[case::a_model_free_only_in_the_cost_map_is_paid("free-by-the-cost-map", true)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_caller_over_its_budget_is_refused_unless_the_model_is_free(
        #[case] model: &str,
        #[case] refused: bool,
    ) {
        let refusal = refused_by_the_stage(budget(MAX, MAX + 1.0), model).await;

        assert_eq!(refusal.is_some(), refused, "{refusal:?}");
    }

    #[derive(Clone, Copy)]
    enum Outcome {
        Success,
        Failure,
        Disconnect,
    }

    #[rstest]
    #[case::owner_success(false, Outcome::Success)]
    #[case::member_success(true, Outcome::Success)]
    #[case::owner_failure(false, Outcome::Failure)]
    #[case::member_failure(true, Outcome::Failure)]
    #[case::owner_disconnect(false, Outcome::Disconnect)]
    #[case::member_disconnect(true, Outcome::Disconnect)]
    #[tokio::test]
    async fn a_request_holds_then_reconciles_its_shared_counter(
        #[case] member: bool,
        #[case] outcome: Outcome,
    ) {
        let store = Arc::new(SpendCounters::new(None));
        let config = litellm_config::Config::from_value(
            serde_yaml_ng::from_str(
                "model_list:
  - model_name: metered
    litellm_params: {model: anthropic/claude-test}
    model_info: {input_cost_per_token: 0.01, output_cost_per_token: 0.02}",
            )
            .unwrap(),
        )
        .unwrap();
        let check = BudgetCheck::with_reservation(
            SpendReader::new(store.clone(), Arc::new(NoRow)),
            Arc::new(FreeModels::default()),
            Reserve {
                router: crate::boot::router(&config).await.unwrap(),
                prices: Arc::new(ArcSwap::from_pointee(CostMap::bundled())),
                budgets: Arc::new(Budgets::new(store.clone())),
                host: Arc::new(crate::host::RustSpendLogHost::new(None)),
            },
        );
        let callbacks = Arc::new(Callbacks::new(vec![check.registered()]).unwrap());
        let known_spend = 2.0;
        let budget = budget(MAX, known_spend);
        let (caller, counter) = if member {
            (
                caller(own_member(budget)),
                SpendCounter::TeamMember {
                    user_id: "u1".to_owned(),
                    team_id: "t1".to_owned(),
                },
            )
        } else {
            (caller(user(budget)), SpendCounter::User("u1".to_owned()))
        };
        let read = || async {
            store
                .read(&counter)
                .await
                .unwrap()
                .value()
                .unwrap()
                .as_f64()
        };
        let actual = record::scope(CallRecord::new(), async {
            record::write(|entries| entries.call_type = Some(CallType::AnthropicMessages));
            let call = callbacks.begin(&caller);
            let body = json!({
                "model": "metered",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello there"}],
            })
            .as_object()
            .unwrap()
            .clone();
            call.pre_call(body.clone()).await.unwrap();
            let hold = record::read(|entries| entries.budget_reservation.clone())
                .flatten()
                .expect("the call reserved before dispatch");
            let input_tokens = crate::prompt_estimate::Countable::of(
                &litellm_gateway_inference::routing::Prompt::Messages(&body),
            )
            .and_then(|prompt| prompt.tokens())
            .unwrap();
            let input_cost = input_tokens as f64 * 0.01;
            assert!((read().await - (known_spend + input_cost + 16.0 * 0.02)).abs() < 1e-9);

            let actual = match outcome {
                Outcome::Success => {
                    record::write(|entries| {
                        entries.attempt = Some(litellm_gateway_inference::record::AttemptRecord {
                            cost: Some(litellm_host::interceptors::Cost::Reported {
                                amount: serde_json::Number::from_f64(0.25).unwrap(),
                            }),
                            ..Default::default()
                        });
                    });
                    call.succeeded(litellm_gateway_inference::ServedCall {
                        model_group: "metered".to_owned(),
                        deployment: None,
                        response_id: None,
                        usage: None,
                        first_event: None,
                    })
                    .await;
                    0.25
                }
                Outcome::Failure => {
                    call.failed().await;
                    0.0
                }
                Outcome::Disconnect => {
                    call.response_headers().await;
                    drop(call);
                    input_cost
                }
            };
            assert!((hold.input_cost.unwrap() - input_cost).abs() < 1e-9);
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while (read().await - (known_spend + actual)).abs() >= 1e-9 {
                    if !matches!(outcome, Outcome::Disconnect) {
                        panic!("the terminal hook did not reconcile the hold");
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("disconnect settled the input cost");
            actual
        })
        .await;
        assert!((read().await - (known_spend + actual)).abs() < 1e-9);
    }
}
