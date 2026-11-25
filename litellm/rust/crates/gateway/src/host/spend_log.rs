//! The gateway's answers to the spend log callback's host calls, the counterpart of the proxy's
//! `SpendLogHost` (`rust_spend_logger.py`) and the counter step's `_CounterHost`.
//!
//! A caller the gateway serves is the master key or a key that names no team, owner, project,
//! organization, budget or access group, since `host/auth` declines any other. So the key's
//! identity is already whole and its charge debits no budget window, customer, access group or
//! committed counter. A lifetime counter's stored spend is read from its row (`Rows`), which a
//! budget reservation's cold counters start from too. The work Python does after a row is
//! written or charged, its alerts and the logging payload's cost have no counterpart here yet and
//! are answered with nothing.

use futures_util::future::{BoxFuture, join_all};
use litellm_callbacks::{HostCall, HostFault, HostReply};
use litellm_callbacks_budgets::RowSpend;
use litellm_callbacks_spend_logger::host::{
    AccountingContext, AfterCharge, AfterWrite, BatchAccounting, EnrichKeyIdentity,
    FailedTrackingAlert, IncrementSearchFreeTier, LoggedCost,
};
use litellm_redis_store::spend_counters::SpendCounter;
use litellm_repositories::{
    Database, EndUserRecord, EndUserRepository, KeyRepository, Organization,
    OrganizationRepository, Project, ProjectRepository, Side, Team, TeamMembership,
    TeamMembershipRepository, TeamRepository, User, UserRepository,
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::callbacks::hosted::Answers;

/// The counter step's host calls, which `litellm-callbacks-budgets` names but does not export.
const BUDGET_WINDOWS: &str = "budget_windows";
const STORED_SPEND: &str = "stored_spend";
const CACHED_SPEND: &str = "cached_spend";

#[derive(Deserialize)]
struct KeyIdentity {
    key_alias: Option<String>,
    user_id: Option<String>,
    team_id: Option<String>,
    org_id: Option<String>,
    project_id: Option<String>,
}

/// A request that asks one answer of each of its entries.
#[derive(Deserialize)]
struct Entities {
    entities: Vec<serde::de::IgnoredAny>,
}

#[derive(Deserialize)]
struct Sources {
    sources: Vec<serde::de::IgnoredAny>,
}

#[derive(Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
enum StoredCounter {
    Entity { counter_key: String },
    Window {},
}

#[derive(Deserialize)]
struct StoredSpend {
    counters: Vec<StoredCounter>,
}

#[derive(Deserialize)]
struct Alert {
    error_message: String,
    failing_model: String,
}

fn parsed<T: DeserializeOwned>(request: Value) -> Result<T, HostFault> {
    serde_json::from_value(request).map_err(|error| HostFault::Malformed(error.to_string()))
}

/// The rows a lifetime counter's spend is stored in, read as `SpendCounterReseed.from_db` and
/// `end_user_from_db` read them: the key, team, user, team member, organization, project and end
/// user rows. A budget window, a committed twin and every other counter have no row here.
pub struct Rows {
    keys: KeyRepository,
    teams: TeamRepository,
    users: UserRepository,
    members: TeamMembershipRepository,
    orgs: OrganizationRepository,
    projects: ProjectRepository,
    end_users: EndUserRepository,
}

impl Rows {
    pub fn new(database: Database) -> Self {
        Self {
            keys: KeyRepository::new(database.clone()),
            teams: TeamRepository::new(database.clone()),
            users: UserRepository::new(database.clone()),
            members: TeamMembershipRepository::new(database.clone()),
            orgs: OrganizationRepository::new(database.clone()),
            projects: ProjectRepository::new(database.clone()),
            end_users: EndUserRepository::new(database),
        }
    }

    /// The spend `counter`'s row holds, `None` without a row or when the read failed.
    async fn spend(&self, counter: &SpendCounter) -> Option<f64> {
        let read = match counter {
            SpendCounter::Key {
                token,
                window: None,
            } => self
                .keys
                .find(token, Side::Read)
                .await
                .map(|key| key.map(|key| key.view().spend)),
            SpendCounter::Team {
                team_id,
                window: None,
            } => self
                .teams
                .find_with_model_aliases(team_id, Side::Read)
                .await
                .map(|team| team.as_ref().map(Team::spend)),
            SpendCounter::User(user_id) => self
                .users
                .find(user_id, Side::Read)
                .await
                .map(|user| user.as_ref().map(User::spend)),
            SpendCounter::TeamMember { user_id, team_id } => self
                .members
                .find_with_budget(user_id, team_id, Side::Read)
                .await
                .map(|member| member.as_ref().map(TeamMembership::spend)),
            SpendCounter::Org(organization_id) => self
                .orgs
                .find_with_optional_budget(organization_id, false)
                .await
                .map(|org| org.as_ref().map(Organization::spend)),
            SpendCounter::Project(project_id) => self
                .projects
                .find_with_budget(project_id)
                .await
                .map(|project| project.as_ref().map(Project::spend)),
            SpendCounter::EndUser(end_user_id) => self
                .end_users
                .find_with_budget_and_object_permission(end_user_id)
                .await
                .map(|end_user| end_user.as_ref().map(EndUserRecord::spend)),
            _ => return None,
        };
        read.unwrap_or_else(|error| {
            tracing::warn!(error = %litellm_error::Report::new(&error), "a spend counter's stored spend could not be read");
            None
        })
    }
}

/// Answers the spend log callback's host calls from the gateway's database, `None` when no
/// database is configured.
pub struct RustSpendLogHost {
    rows: Option<Rows>,
}

impl RustSpendLogHost {
    pub fn new(rows: Option<Rows>) -> Self {
        Self { rows }
    }

    async fn stored(&self, counter: StoredCounter) -> Option<f64> {
        let StoredCounter::Entity { counter_key } = counter else {
            return None;
        };
        self.stored_spend(&SpendCounter::parse(&counter_key).ok()?)
            .await
    }

    async fn stored_spend(&self, counter: &SpendCounter) -> Option<f64> {
        self.rows.as_ref()?.spend(counter).await
    }

    async fn reply(&self, call: &'static str, request: Value) -> HostReply {
        match call {
            EnrichKeyIdentity::NAME => {
                let known: KeyIdentity = parsed(request)?;
                Ok(json!({
                    "key_alias": known.key_alias,
                    "user_id": known.user_id,
                    "team_id": known.team_id,
                    "org_id": known.org_id,
                    "project_id": known.project_id,
                }))
            }
            AccountingContext::NAME => Ok(json!({})),
            BUDGET_WINDOWS => {
                let asked: Entities = parsed(request)?;
                Ok(json!(vec![json!([]); asked.entities.len()]))
            }
            STORED_SPEND => {
                let asked: StoredSpend = parsed(request)?;
                let stored = join_all(
                    asked
                        .counters
                        .into_iter()
                        .map(|counter| self.stored(counter)),
                );
                Ok(json!(stored.await))
            }
            CACHED_SPEND => {
                let asked: Sources = parsed(request)?;
                Ok(json!(vec![0.0; asked.sources.len()]))
            }
            FailedTrackingAlert::NAME => {
                let alert: Alert = parsed(request)?;
                tracing::error!(
                    failing_model = alert.failing_model,
                    "{}",
                    alert.error_message
                );
                Ok(Value::Null)
            }
            AfterWrite::NAME
            | AfterCharge::NAME
            | LoggedCost::NAME
            | BatchAccounting::NAME
            | IncrementSearchFreeTier::NAME => Ok(Value::Null),
            other => Err(HostFault::Malformed(format!(
                "the gateway answers no {other} host call"
            ))),
        }
    }
}

impl Answers for RustSpendLogHost {
    fn answer(&self, call: &'static str, request: Value) -> BoxFuture<'_, HostReply> {
        Box::pin(self.reply(call, request))
    }
}

impl RowSpend for RustSpendLogHost {
    fn stored<'a>(&'a self, counter: &'a SpendCounter) -> BoxFuture<'a, Option<f64>> {
        Box::pin(self.stored_spend(counter))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use litellm_callbacks::{ErasedCallback, Field, FieldSet, FieldValues, Hook, HookInput};
    use litellm_callbacks_budgets::Budgets;
    use litellm_callbacks_spend_logger::{SpendLogCallback, SpendLogParts};
    use litellm_redaction::TokenRedactor;
    use litellm_redis_store::{Number, spend_counters::SpendCounters};
    use litellm_repositories::SpendLogWrite;
    use litellm_spend_log_writer::{LedgerStore, SpendLogFlusher, SpendLogWriter, WriterSettings};
    use rstest::rstest;

    use litellm_gateway_auth_core::Token;

    use super::*;
    use crate::{callbacks::hosted, host::auth::keys::Held};

    const TOKEN: &str = "sk-stored";
    const KEY: &str = "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb";

    struct Store;

    impl LedgerStore for Store {
        async fn insert(&self, writes: &[SpendLogWrite]) -> Result<usize, litellm_db::Error> {
            Ok(writes.len())
        }
    }

    /// A served call's `spend_log_input` as the gateway builds it, for a key with no owner.
    fn served() -> Value {
        let at = json!({"raw": "2026-11-18T11:06:40", "utc": "2026-11-18T11:06:40+00:00"});
        json!({
            "kind": "success",
            "identity": {"token": KEY, "user_id": null, "key_alias": "ci"},
            "call_type": "anthropic_messages",
            "stream": false,
            "stream_complete": false,
            "row": {"row": {
                "call_type": "anthropic_messages",
                "cache_hit": false,
                "now": 1_795_000_000.0,
                "fallback_session_id": "session-1",
                "start_time": at,
                "end_time": at,
                "completion_start_time": at,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "metadata": {
                    "user_api_key": KEY,
                    "user_api_key_hash": KEY,
                    "user_api_key_alias": "ci",
                    "user_api_key_request_route": "/v1/messages",
                    "model_group": "claude",
                    "model_info": {"id": "claude-a"},
                    "deployment": "anthropic/claude-haiku-5-5",
                    "status": "success",
                },
                "litellm_params": {"litellm_call_id": "call-1"},
                "kwargs": {
                    "model": "anthropic/claude-haiku-5-5",
                    "custom_llm_provider": "anthropic",
                    "litellm_call_id": "call-1",
                    "litellm_trace_id": "trace-1",
                },
                "cache_key": "Cache OFF",
            }},
        })
    }

    fn input(cost: Option<f64>) -> HookInput {
        let reads = FieldSet::of(&[
            Field::Outcome,
            Field::SpendLogInput,
            Field::SpendLogId,
            Field::Cost,
            Field::BudgetReservation,
        ]);
        let fields = FieldValues::new(reads)
            .with(Field::Outcome, json!("success"))
            .with(Field::SpendLogInput, served())
            .with(Field::SpendLogId, json!({"id": "msg_1"}));
        let fields = match cost {
            Some(cost) => fields.with(Field::Cost, json!({"response_cost": cost})),
            None => fields,
        };
        HookInput {
            hook: Hook::AttemptEnded,
            fields,
            call: None,
        }
    }

    struct Charged {
        rows: Vec<Value>,
        key_spend: Option<f64>,
    }

    /// One success through the spend log callback with this host answering it.
    async fn charge(cost: Option<f64>) -> Charged {
        let counters = Arc::new(SpendCounters::new(None));
        let writer = Arc::new(SpendLogWriter::new(
            SpendLogFlusher::new(Store, WriterSettings::default()),
            || {},
        ));
        let callback = Arc::new(SpendLogCallback::new(SpendLogParts {
            writer: writer.clone(),
            budgets: Arc::new(Budgets::new(counters.clone())),
            tokens: Arc::new(TokenRedactor::new(0, Vec::new())),
        }));
        let host = RustSpendLogHost::new(None);
        hosted::turn(
            callback as Arc<dyn ErasedCallback>,
            input(cost),
            None,
            &host,
        )
        .await
        .unwrap();
        let counter = SpendCounter::Key {
            token: KEY.to_owned(),
            window: None,
        };
        Charged {
            rows: writer
                .take(10)
                .iter()
                .map(|row| serde_json::from_str(&row.to_json()).unwrap())
                .collect(),
            key_spend: counters.get_local(&counter).unwrap().map(Number::as_f64),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn a_priced_call_is_written_and_charged_to_its_key() {
        let charged = charge(Some(0.25)).await;
        let row = &charged.rows[0];
        assert_eq!(
            (
                charged.rows.len(),
                row["request_id"].clone(),
                row["api_key"].clone(),
                row["spend"].clone(),
                row["total_tokens"].clone(),
                charged.key_spend,
            ),
            (
                1,
                json!("msg_1"),
                json!(KEY),
                json!(0.25),
                json!(12),
                Some(0.25)
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_unpriced_call_is_neither_written_nor_charged() {
        let charged = charge(None).await;
        assert_eq!((charged.rows.len(), charged.key_spend), (0, None));
    }

    const ROWS: &str = r#"
        INSERT INTO "LiteLLM_TeamTable" (team_id, spend) VALUES ('team', 2.5);
        INSERT INTO "LiteLLM_UserTable" (user_id, spend) VALUES ('user', 3.5);
        INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend) VALUES ('user', 'team', 4.5);
        INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by) VALUES ('budget', 'test', 'test');
        INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, created_by, updated_by, spend)
            VALUES ('org', 'org', 'budget', 'test', 'test', 5.5);
        INSERT INTO "LiteLLM_ProjectTable" (project_id, created_by, updated_by, spend) VALUES ('project', 'test', 'test', 6.5);
        INSERT INTO "LiteLLM_EndUserTable" (user_id, spend) VALUES ('customer', 7.5);
    "#;

    /// Every lifetime counter starts from the spend its row holds, through the spend log's host
    /// call and a reservation's row read alike. A budget window, a counter whose row is gone and
    /// one with no row at all start from nothing.
    #[rstest]
    #[tokio::test]
    async fn a_lifetime_counter_starts_from_the_spend_its_row_holds() {
        let held = Held::holding(TOKEN, json!({"spend": 1.5})).await;
        held.execute(ROWS).await;
        let host = RustSpendLogHost::new(Some(Rows::new(held.repositories())));
        let key = Token::new(TOKEN).hash().as_str().to_owned();
        let counters = [
            format!("spend:key:{key}"),
            "spend:team:team".to_owned(),
            "spend:user:user".to_owned(),
            "spend:team_member:user:team".to_owned(),
            "spend:org:org".to_owned(),
            "spend:project:project".to_owned(),
            "spend:end_user:customer".to_owned(),
            format!("spend:key:{key}:window:1d"),
            "spend:team:gone".to_owned(),
            "spend:tag:prod".to_owned(),
        ];
        let request = json!({"counters": counters
            .iter()
            .map(|counter_key| json!({"source": "entity", "counter_key": counter_key}))
            .chain([json!({"source": "window"})])
            .collect::<Vec<_>>()});
        let member = SpendCounter::parse("spend:team_member:user:team").unwrap();

        assert_eq!(
            (
                host.reply(STORED_SPEND, request).await.unwrap(),
                RowSpend::stored(&host, &member).await,
            ),
            (
                json!([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, null, null, null, null]),
                Some(4.5),
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_host_call_the_gateway_does_not_know_is_refused() {
        let host = RustSpendLogHost::new(None);
        assert!(matches!(
            host.reply("reseed_counter", json!({})).await,
            Err(HostFault::Malformed(_))
        ));
    }
}
