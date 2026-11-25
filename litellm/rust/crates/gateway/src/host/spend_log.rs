//! The gateway's answers to the spend log callback's host calls, the counterpart of the proxy's
//! `SpendLogHost` (`rust_spend_logger.py`) and the counter step's `_CounterHost`.
//!
//! A caller the gateway serves is the master key or a key that names no team, owner, project,
//! organization, budget or access group, since `host/auth` declines any other. So the key's
//! identity is already whole, its charge debits no budget window, customer, access group or
//! committed counter, and the only spend the database holds for it is the key row's own. The
//! work Python does after a row is written or charged, its alerts and the logging payload's cost
//! have no counterpart here yet and are answered with nothing.

use std::sync::Arc;

use futures_util::future::{BoxFuture, join_all};
use litellm_callbacks::{HostCall, HostFault, HostReply};
use litellm_callbacks_spend_logger::host::{
    AccountingContext, AfterCharge, AfterWrite, BatchAccounting, EnrichKeyIdentity,
    FailedTrackingAlert, IncrementSearchFreeTier, LoggedCost,
};
use litellm_redis_store::spend_counters::SpendCounter;
use litellm_repositories::{KeyRepository, Side};
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

/// Answers the spend log callback's host calls from the gateway's key database, `None` when no
/// database is configured.
pub struct RustSpendLogHost {
    keys: Option<Arc<KeyRepository>>,
}

impl RustSpendLogHost {
    pub fn new(keys: Option<Arc<KeyRepository>>) -> Self {
        Self { keys }
    }

    /// The spend the database holds for a key's lifetime counter. Every other counter a caller of
    /// the Rust gateway is charged on has no row, as the master key and its `default_user_id` have none the
    /// gateway reads, and a read that fails is `None`, as `SpendCounterReseed.from_db` gives it.
    async fn stored(&self, counter: StoredCounter) -> Option<f64> {
        let StoredCounter::Entity { counter_key } = counter else {
            return None;
        };
        let Ok(SpendCounter::Key {
            token,
            window: None,
        }) = SpendCounter::parse(&counter_key)
        else {
            return None;
        };
        match self.keys.as_ref()?.find(&token, Side::Read).await {
            Ok(key) => Some(key?.view().spend),
            Err(error) => {
                tracing::warn!(error = %litellm_error::Report::new(&error), "the key's stored spend could not be read");
                None
            }
        }
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

#[cfg(test)]
mod tests {
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

    /// A key's lifetime counter starts from the spend its row holds, and a budget window, which
    /// has no row, starts from nothing.
    #[rstest]
    #[tokio::test]
    async fn a_keys_counter_starts_from_the_spend_its_row_holds() {
        let held = Held::holding(TOKEN, json!({"spend": 1.5})).await;
        let host = RustSpendLogHost::new(Some(Arc::new(KeyRepository::new(held.repositories()))));
        let counter = SpendCounter::Key {
            token: Token::new(TOKEN).hash().as_str().to_owned(),
            window: None,
        };
        let request = json!({"counters": [
            {"source": "entity", "counter_key": counter.key()},
            {"source": "window"},
        ]});
        assert_eq!(
            host.reply(STORED_SPEND, request).await.unwrap(),
            json!([1.5, null])
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
