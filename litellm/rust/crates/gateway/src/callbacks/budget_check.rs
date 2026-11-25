//! The budget checks the proxy runs while it authenticates a call, before the budget reservation,
//! for a native `/v1/messages` call. Each budget the caller holds is read as `get_current_spend`
//! reads it, decided by `litellm_callbacks_budgets::decide` as Python decides it, and the first
//! budget the caller is over refuses the call with Python's budget error, in Python's order: the
//! key path checks the member's own budget and then the key's, then `common_checks` the project's,
//! then the team's, organization's, user's, team member's and end user's.
//!
//! A call to a free model is checked against no budget, as `request_skips_budget_checks` skips it.
//! No throttle or alert is decided, since a config that sets `budget_exceeded_throttle_percentage`
//! or `alerting` goes to Python, and a decision of either fails the call rather than letting it
//! through unchecked.
//!
//! Only the budgets `budgets::counters` plans are checked. A `budget_limits` window, soft budget,
//! tag, model access group, per-model budget, temporary increase, throttle, budget alert or the
//! proxy-wide budget is never checked here, since the key column, request or setting that asks
//! for one sends the call to Python first.

use std::sync::Arc;

use futures_util::future::{BoxFuture, join_all};
use litellm_callbacks::{
    Callback, CallbackError, Hook, HookInput, HookOutput, Manifest, Selection, Subscription,
    Verdict,
};
use litellm_callbacks_budgets::{Check, Decision, PyNumber, SpendReader, decide};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::Credential;
use litellm_redis_store::spend_counters::SpendCounter;
use serde_json::Value;

use super::{Registered, budget_reservation, budgets};
use crate::{Admitted, host::NativeRows};

pub(crate) const NAME: &str = "budget_check";

/// Refuses a native call whose caller is over one of its budgets, before anything is held.
pub struct BudgetCheck {
    manifest: Manifest,
    spend: SpendReader,
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
    counter: SpendCounter,
    fallback_spend: f64,
    max_budget: f64,
    entities: Vec<Entity>,
}

/// The budgets one request's caller is checked against, none for a call to a free model.
pub struct Budgeted {
    held: Vec<Held>,
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
                .belongs
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

fn budgeted(identity: &Authenticated<NativeRows>, admitted: Option<&Admitted>) -> Budgeted {
    if admitted.is_some_and(|admitted| admitted.free) {
        return Budgeted { held: Vec::new() };
    }
    let held = budgets::counters(identity, budgets::Settings::default())
        .into_iter()
        .filter_map(|planned| {
            let counter = SpendCounter::parse(&planned.counter_key).ok()?;
            Some(Held {
                entities: entities(identity, &counter),
                counter,
                fallback_spend: planned.fallback_spend,
                max_budget: planned.max_budget,
            })
        })
        .collect();
    Budgeted { held }
}

/// What `decision` does to the call, `None` for one that lets it go on to the next budget.
fn verdict(decision: Decision) -> Option<HookOutput> {
    match decision {
        Decision::Skip | Decision::Pass => None,
        Decision::Refuse(refusal) => Some(match serde_json::to_value(refusal) {
            Ok(Value::Object(fields)) => Ok(Some(Verdict::Abort(budget_reservation::over_budget(
                fields,
            )))),
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
            may_abort: true,
            ..Subscription::on(Hook::PreCall)
        }],
        has_request_state: true,
    }
}

impl BudgetCheck {
    pub fn new(spend: SpendReader) -> Arc<Self> {
        Arc::new(Self {
            manifest: manifest(),
            spend,
        })
    }

    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        Registered::for_caller(self.clone(), budgeted)
    }

    async fn pre_call(&self, budgeted: &Budgeted) -> HookOutput {
        let spends = join_all(budgeted.held.iter().map(|held| {
            self.spend
                .current(&held.counter, held.fallback_spend, Some(held.max_budget))
        }))
        .await;
        let mut checks = budgeted
            .held
            .iter()
            .zip(spends)
            .flat_map(|(held, spend)| {
                held.entities
                    .iter()
                    .map(move |entity| (entity, spend, held.max_budget))
            })
            .collect::<Vec<_>>();
        checks.sort_by_key(|&(entity, ..)| entity);
        checks
            .into_iter()
            .find_map(|(entity, spend, max_budget)| {
                verdict(decide(&entity.check(spend, max_budget)))
            })
            .unwrap_or(Ok(None))
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
                (Hook::PreCall, Some(budgeted)) => self.pre_call(budgeted).await,
                _ => Ok(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use litellm_callbacks::Abort;
    use litellm_callbacks_budgets::{Budgets, RowSpend};
    use litellm_gateway_auth_core::{
        Belongs, Caller, Holder, Named, Team,
        record::{
            Budget, EndUserRecord, MembershipRecord, OrgRecord, ProjectRecord, TeamRecord,
            UserRecord,
        },
    };
    use litellm_gateway_inference::{
        CallHooks, HookError,
        record::{self, CallRecord},
    };
    use litellm_redis_store::spend_counters::SpendCounters;
    use rstest::rstest;
    use serde_json::{Map, json};

    use super::*;
    use crate::{
        callbacks::{Callbacks, budget_reservation::Reserver},
        host::{Hold, Key},
    };

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
            belongs: Belongs {
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
        })
    }

    fn check(store: &Arc<SpendCounters>) -> Arc<BudgetCheck> {
        BudgetCheck::new(SpendReader::new(store.clone(), Arc::new(NoRow)))
    }

    /// The status, entity type and message the call is refused with.
    fn refused(abort: &Abort) -> (StatusCode, String, String) {
        let refusal = budget_reservation::refusal(abort).expect("a budget refusal");
        let entity_type = abort
            .error
            .as_ref()
            .and_then(|error| error.fields.get("entity_type"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        (refusal.status, entity_type, refusal.message)
    }

    /// What one pre-call check of `identity` refuses the call with, `None` when it lets it on.
    async fn checked(
        identity: &Authenticated<NativeRows>,
        admitted: Option<Admitted>,
    ) -> Option<(StatusCode, String, String)> {
        let check = check(&Arc::new(SpendCounters::new(None)));
        let verdict = check
            .pre_call(&budgeted(identity, admitted.as_ref()))
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

        let refusal = checked(&caller(holds(budget(MAX, spend))), None).await;

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
    async fn a_budget_of_zero_or_less_is_checked_although_nothing_is_reserved_against_it(
        #[case] holds: Holds,
        #[case] refused_as: Option<&str>,
    ) {
        let refusal = checked(&caller(holds), None).await;

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
        assert_eq!(checked(&caller(holds), None).await, None);
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
        let refusal = checked(&caller(holds), None).await;

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

        let refusal = checked(&identity, None).await;

        assert_eq!(
            refusal.map(|(.., message)| message),
            Some(
                "Budget has been exceeded! Key=my-alias (sk-...abcd) Current cost: 10.0, Max budget: 10.0"
                    .to_owned()
            )
        );
    }

    #[rstest]
    #[case::a_free_model(Some(Admitted { hold: None, free: true }), false)]
    #[case::a_priced_model(Some(Admitted { hold: None, free: false }), true)]
    #[case::a_call_admit_did_not_see(None, true)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_to_a_free_model_is_checked_against_no_budget(
        #[case] admitted: Option<Admitted>,
        #[case] refuses: bool,
    ) {
        let identity = caller(Holds {
            key: over(),
            team: over(),
            ..Holds::default()
        });

        assert_eq!(checked(&identity, admitted).await.is_some(), refuses);
    }

    const HELD: f64 = 1.0;

    /// What a call from a caller with `key_budget` comes to with the checks and then the
    /// reservation registered, as the gateway registers them: what it is refused with, and what
    /// the key's counter holds once the pre-call hook ends.
    async fn checked_then_held(
        key_budget: Budget,
        disabled: bool,
    ) -> (Option<String>, Option<f64>) {
        let store = Arc::new(SpendCounters::new(None));
        let reserver = Reserver::new(
            Arc::new(Budgets::new(store.clone())),
            Arc::new(NoRow),
            disabled,
        );
        let callbacks = Arc::new(
            Callbacks::new(vec![check(&store).registered(), reserver.registered()]).unwrap(),
        );
        let admitted = Admitted {
            hold: Some(Hold {
                reservation_cost: HELD,
                input_cost: HELD / 2.0,
                input_tokens: 9,
            }),
            free: false,
        };
        let Value::Object(body) = json!({"model": "held-model", "max_tokens": 16, "messages": []})
        else {
            unreachable!("the body is an object")
        };
        let body: Map<String, Value> = body;
        record::scope(CallRecord::new(), async {
            let call = callbacks.begin(&caller(key(key_budget)), Some(&admitted));
            let refused = match call.pre_call(body).await {
                Ok(_) => None,
                Err(HookError::Refused(refusal)) => Some(refusal.message),
                Err(HookError::Failed(error)) => panic!("the call failed: {error}"),
            };
            let counter = SpendCounter::Key {
                token: "hash".to_owned(),
                window: None,
            };
            let held = store.get_local(&counter).unwrap().map(|held| held.as_f64());
            (refused, held)
        })
        .await
    }

    const AT_ITS_BUDGET: &str =
        "Budget has been exceeded! Key=key Current cost: 10.0, Max budget: 10.0";

    #[rstest]
    #[case::under_its_budget_it_is_held(budget(MAX, 9.0), false, None, Some(9.0 + HELD))]
    #[case::at_its_budget_the_check_refuses_before_any_hold(
        budget(MAX, MAX),
        false,
        Some(AT_ITS_BUDGET),
        Some(MAX)
    )]
    #[case::at_a_budget_of_zero_the_check_refuses_what_nothing_holds(
        budget(0.0, 0.0),
        false,
        Some("Budget has been exceeded! Key=key Current cost: 0.0, Max budget: 0.0"),
        None
    )]
    #[case::without_reservation_nothing_is_held(budget(MAX, 9.0), true, None, Some(9.0))]
    #[case::without_reservation_the_check_still_refuses(
        budget(MAX, MAX),
        true,
        Some(AT_ITS_BUDGET),
        Some(MAX)
    )]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_is_checked_and_then_held_unless_reservation_is_disabled(
        #[case] key_budget: Budget,
        #[case] disabled: bool,
        #[case] refused: Option<&str>,
        #[case] held: Option<f64>,
    ) {
        assert_eq!(
            checked_then_held(key_budget, disabled).await,
            (refused.map(str::to_owned), held)
        );
    }
}
