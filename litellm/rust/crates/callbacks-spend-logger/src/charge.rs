//! The success and failure flows and `update_database`'s ledger half: gate, search free tier,
//! build, claim or enqueue, the post-write host call, then the counters and the after-charge host
//! call, in the cost callback's order.
//!
//! The reservation is settled once, by the counter step (`charge_counters`), which reconciles it
//! against the counters it holds and reseeds any it finds cold; a call that charges nothing
//! releases it. Each step stamps the caller's copy, which [`crate::SpendLogCallback`] writes back.

use std::sync::atomic::{AtomicBool, Ordering};

use litellm_callbacks::{BudgetReservation, Host, Scalar};
use litellm_callbacks_budgets::{CounterStep, CustomerWindow, Failure as Failed, post_call};
use litellm_callbacks_spend_log::{
    LedgerColumns, LedgerInput, RowConfig, RowInput, SpendLogRow, SpendLogRowBuilder,
    SpendLogStatus, hash_token, is_sha256_hex, strip_bearer,
};
use litellm_python_compat::json;
use litellm_redis_store::{RequestRedis, spend_counters::SpendCounter};
use litellm_spend_log_writer::{LedgerStore, QueuedSpendLog, Retention};
use serde_json::{Value as Json, json};

use crate::{
    SpendLogCallback, Typed, customer_margin,
    hook_input::{Failure, Identity, RowRequest, Settings, Success},
    host::{
        AccountingContext, AccountingRow, AfterCharge, AfterWrite, BatchAccounting,
        ChargeFailureCounters, EnrichKeyIdentity, EnrichedKeyIdentity, FailedTrackingAlert,
        IncrementSearchFreeTier, LoggedCost,
    },
    policy::{self, Disposition},
    search_free_tier,
};

const RESPONSES_ID_CALL_TYPES: [&str; 2] = ["aget_responses", "acancel_responses"];

type Error = Box<dyn std::error::Error + Send + Sync>;

/// What `update_database` did with one row: whether the request is charged, and what the counter
/// step charges besides the request's own counters (the customer window counters and the
/// committed twins). Both survive a row that fails to build.
struct Written {
    charged: bool,
    customer_windows: Vec<CustomerWindow>,
    committed_tracked: Vec<SpendCounter>,
}

impl Written {
    /// Charged, with only what the request itself carried.
    fn carried(request: &RowRequest) -> Self {
        Self {
            charged: true,
            customer_windows: Vec::new(),
            committed_tracked: request.committed_tracked.clone().unwrap_or_default(),
        }
    }
}

/// One row to write: its place in the hook's rows (the host reads its kwargs there), its
/// request, and the request id the typed contract fields gave it.
struct RowWrite<'a> {
    position: usize,
    request: &'a RowRequest,
    request_id: Option<&'a str>,
}

/// Who and what one charged row's counter step charges.
struct Charge<'a> {
    identity: &'a Identity,
    row: &'a RowRequest,
    cost: f64,
    customer_hierarchy: bool,
}

/// How a built row was recorded.
enum Recorded {
    /// Queued, or written as a job's cost row that carried nothing to claim.
    Queued,
    /// The job's cost row landed now, so this poll owns the charge.
    Claimed,
    /// An earlier poll already owns the job's charge.
    Owned,
}

/// What a success tells CheckBatchCost (`BatchAccountingOutcome`): whether the spend write began,
/// and whether tracking failed.
#[derive(Default)]
struct BatchOutcome {
    write_started: AtomicBool,
    failed: AtomicBool,
}

impl BatchOutcome {
    async fn report(&self, host: &Host) {
        let write_started = self.write_started.load(Ordering::Relaxed);
        let failed = self.failed.load(Ordering::Relaxed);
        if !(write_started || failed) {
            return;
        }
        if let Err(error) = host
            .call(BatchAccounting {
                write_started,
                failed,
            })
            .await
        {
            tracing::debug!(%error, "the batch accounting host call failed");
        }
    }
}

/// `hash_token` for a raw `sk-` key; anything else is already the logged form.
fn hashed_token(token: Option<&str>) -> Option<String> {
    token.map(|token| {
        if token.starts_with("sk-") {
            hash_token(token)
        } else {
            token.to_owned()
        }
    })
}

/// `_persistable_api_key`: a key the row did not log as is, and that is no hash, is stored
/// hashed with any bearer prefix dropped.
fn persistable_api_key(token: Option<&str>, logged: &str) -> Option<String> {
    let token = token?;
    if token == logged || is_sha256_hex(token) {
        return Some(token.to_owned());
    }
    let stripped = strip_bearer(token);
    (!stripped.is_empty()).then(|| hash_token(stripped))
}

/// A finite numeric cost from a typed `Cost` member.
fn finite(cost: Option<&Scalar>) -> Option<f64> {
    let value = match cost? {
        Scalar::Float(value) => *value,
        Scalar::Int(value) => *value as f64,
        _ => return None,
    };
    value.is_finite().then_some(value)
}

/// `_write_spend_metadata_to_kwargs` on the row input: the enriched `user_api_key_*` values fill
/// the metadata the row reads wherever it holds none.
fn enrich_row(row: &mut RowRequest, enriched: &EnrichedKeyIdentity) {
    let Some(metadata) = row.row.get_mut("metadata").and_then(Json::as_object_mut) else {
        return;
    };
    for (key, value) in [
        ("user_api_key_alias", &enriched.key_alias),
        ("user_api_key_user_id", &enriched.user_id),
        ("user_api_key_team_id", &enriched.team_id),
        ("user_api_key_org_id", &enriched.org_id),
        ("user_api_key_project_id", &enriched.project_id),
        ("user_api_key_team_alias", &enriched.team_alias),
    ] {
        let Some(value) = value else { continue };
        if metadata.get(key).is_none_or(Json::is_null) {
            metadata.insert(key.to_owned(), Json::String(value.clone()));
        }
    }
}

/// `_release_budget_reservation` settles the reservation dict the request's metadata holds before
/// main builds the failure row, so the stored request body shows the settled `finalized` and
/// `applied_adjustment` stamps, as `write_settled_budget_reservation` later writes them back.
fn stamp_stored_reservation(row: &mut RowRequest, settled: &BudgetReservation) {
    for bucket in ["metadata", "litellm_metadata"] {
        let Some(held) = row
            .row
            .pointer_mut(&format!(
                "/litellm_params/proxy_server_request/body/{bucket}/user_api_key_budget_reservation"
            ))
            .and_then(Json::as_object_mut)
        else {
            continue;
        };
        if let Some(finalized) = settled.finalized {
            held.insert("finalized".to_owned(), Json::Bool(finalized));
        }
        let (Some(Json::Array(entries)), Some(settled_entries)) =
            (held.get_mut("entries"), &settled.entries)
        else {
            continue;
        };
        for (entry, settled_entry) in entries.iter_mut().zip(settled_entries) {
            let adjustment = settled_entry
                .as_ref()
                .and_then(|settled_entry| settled_entry.applied_adjustment);
            if let (Some(entry), Some(adjustment)) = (entry.as_object_mut(), adjustment) {
                entry.insert("applied_adjustment".to_owned(), json!(adjustment));
            }
        }
    }
}

fn fill(slot: &mut Option<String>, value: &Option<String>) {
    if slot.is_none() {
        slot.clone_from(value);
    }
}

/// `_charge_request_cost`'s second charge: the row of a repeat poll carries only what the poll
/// still pays for.
fn repriced(row: &RowRequest, cost: f64) -> RowRequest {
    let mut repriced = row.clone();
    for key in ["kwargs", "standard_logging_object"] {
        if let Some(section) = repriced.row.get_mut(key).and_then(Json::as_object_mut)
            && section.contains_key("response_cost")
        {
            section.insert("response_cost".to_owned(), json!(cost));
        }
    }
    repriced
}

/// The model the row input logged, for the failed tracking alert.
fn logged_model(row: &Json) -> String {
    match policy::model(row) {
        None | Some(Json::Null) => String::new(),
        Some(Json::String(model)) => model.clone(),
        Some(other) => other.to_string(),
    }
}

impl<S: LedgerStore + 'static> SpendLogCallback<S> {
    async fn enrich(
        &self,
        host: &Host,
        identity: &Identity,
        resolve_missing_key_identity: bool,
    ) -> Option<EnrichedKeyIdentity> {
        let api_key = identity.token.clone()?;
        match host
            .call(EnrichKeyIdentity {
                api_key,
                resolve_missing_key_identity,
                key_alias: identity.key_alias.clone(),
                user_id: identity.user_id.clone(),
                team_id: identity.team_id.clone(),
                org_id: identity.org_id.clone(),
                project_id: identity.project_id.clone(),
            })
            .await
        {
            Ok(enriched) => Some(enriched),
            Err(error) => {
                tracing::debug!(%error, "Failed to enrich spend log metadata with key info");
                None
            }
        }
    }

    async fn release(
        &self,
        host: &Host,
        reservation: &mut Option<BudgetReservation>,
        window_resets: &[Option<String>],
    ) {
        if let Some(reservation) = reservation.as_mut() {
            post_call::release_or_invalidate(&self.budgets, host, reservation, window_resets).await;
        }
    }

    /// `_PROXY_track_cost_callback`: whatever it could not track raises the failed tracking
    /// alert, and a batch's costing learns whether it may be retried.
    pub(crate) async fn on_success(
        &self,
        success: Success,
        fields: &Typed,
        reservation: &mut Option<BudgetReservation>,
        host: &Host,
        redis: Option<&RequestRedis>,
    ) {
        let mut success = success;
        if let Some(resolve) = policy::enrich(&success)
            && let Some(enriched) = self.enrich(host, &success.identity, resolve).await
        {
            enrich_row(&mut success.row, &enriched);
            let identity = &mut success.identity;
            fill(&mut identity.key_alias, &enriched.key_alias);
            fill(&mut identity.user_id, &enriched.user_id);
            fill(&mut identity.team_id, &enriched.team_id);
            fill(&mut identity.org_id, &enriched.org_id);
            fill(&mut identity.project_id, &enriched.project_id);
        }
        let outcome = BatchOutcome::default();
        if reservation.is_some() || success.settings.apply_search_free_tier {
            outcome.write_started.store(true, Ordering::Relaxed);
        }
        if let Err(error) = self
            .track(&mut success, fields, reservation, &outcome, host, redis)
            .await
        {
            outcome.failed.store(true, Ordering::Relaxed);
            failed_tracking_alert(&success, &error, host).await;
        }
        if success.batch_accounting {
            outcome.report(host).await;
        }
    }

    async fn track(
        &self,
        success: &mut Success,
        fields: &Typed,
        reservation: &mut Option<BudgetReservation>,
        outcome: &BatchOutcome,
        host: &Host,
        redis: Option<&RequestRedis>,
    ) -> Result<(), String> {
        let resets = success.window_resets.clone();
        let cost = match &fields.cost {
            Some(cost) => finite(cost.response_cost.as_ref()),
            None => success.kwargs_response_cost.filter(|cost| cost.is_finite()),
        };
        let cost_finalized = fields
            .cost
            .as_ref()
            .and_then(|cost| cost.cost_finalized)
            .unwrap_or(false);
        match policy::disposition(success, cost) {
            Disposition::InteractionInProgress => Ok(()),
            Disposition::Charge(priced) => {
                let (cost, discount) = if success.settings.apply_search_free_tier
                    && !cost_finalized
                    && policy::payload(&success.row.row).is_some()
                {
                    self.search_free_tier(success, priced, host).await
                } else {
                    (priced, None)
                };
                let cost = customer_margin::on_success(&mut success.row, cost);
                let customer_margin = success.row.customer_margin;
                if (discount.is_some() || customer_margin.is_some())
                    && let Err(error) = host
                        .call(LoggedCost {
                            cost: priced,
                            discount,
                            customer_margin,
                        })
                        .await
                {
                    tracing::warn!(%error, "the charged cost did not reach the logging payload");
                }
                if !tracked(success) {
                    self.release(host, reservation, &resets).await;
                    return Ok(());
                }
                let charged = self
                    .charge_request(success, fields, cost, reservation, outcome, host, redis)
                    .await
                    .map_err(|error| error.to_string())?;
                if let Some(charged) = charged {
                    after_charge(success, charged, host).await;
                }
                Ok(())
            }
            Disposition::Unpriced => {
                self.release(host, reservation, &resets).await;
                Err(unpriced(success, fields))
            }
            Disposition::DeferBatch | Disposition::InteractionUnbilled | Disposition::Released => {
                self.release(host, reservation, &resets).await;
                Ok(())
            }
        }
    }

    /// `apply_search_free_tier`: the cost after the grounding the free allowance covers, and the
    /// discount when the allowance was drawn from. Any counter failure bills the grounding at
    /// full price.
    async fn search_free_tier(
        &self,
        success: &mut Success,
        cost: f64,
        host: &Host,
    ) -> (f64, Option<f64>) {
        let now = chrono::Utc::now();
        let Some(draw) = search_free_tier::draw(&success.row.row, success.cache_hit, now) else {
            return (cost, None);
        };
        let total_after = match host
            .call(IncrementSearchFreeTier {
                key: draw.key.clone(),
                units: draw.units,
                ttl: draw.ttl,
            })
            .await
        {
            Ok(Some(total_after)) => total_after,
            Ok(None) => {
                tracing::warn!(
                    "apply_search_free_tier is on but the proxy has no Redis, grounding is billed at full price"
                );
                return (cost, None);
            }
            Err(error) => {
                tracing::warn!(%error, key = draw.key, "search free tier counter is unavailable, grounding is billed at full price");
                return (cost, None);
            }
        };
        let discount = search_free_tier::discount(&draw, cost, total_after);
        (
            search_free_tier::apply(&mut success.row.row, cost, discount),
            Some(discount),
        )
    }

    /// `_charge_request_cost`: what the request charged, trying the repeat poll's own cost, under
    /// its own guardrail cost id, when an earlier poll already charged the job.
    #[expect(
        clippy::too_many_arguments,
        reason = "the Python function's values plus the host and the request's Redis batches"
    )]
    async fn charge_request(
        &self,
        success: &Success,
        fields: &Typed,
        cost: f64,
        reservation: &mut Option<BudgetReservation>,
        outcome: &BatchOutcome,
        host: &Host,
        redis: Option<&RequestRedis>,
    ) -> Result<Option<f64>, Failed> {
        let request_id = fields.spend_log_id.as_ref().and_then(|id| id.id.as_deref());
        let write = RowWrite {
            position: 0,
            request: &success.row,
            request_id,
        };
        if self
            .update_and_count(
                &success.settings,
                &success.identity,
                &write,
                cost,
                reservation,
                &success.window_resets,
                outcome,
                host,
                redis,
            )
            .await?
        {
            return Ok(Some(cost));
        }
        let repeat = policy::repeat_poll_cost(success.call_type.as_deref(), cost, &success.row.row);
        if repeat <= 0.0 {
            return Ok(None);
        }
        let repriced = repriced(&success.row, repeat);
        let guardrail_request_id = policy::guardrail_cost_request_id(&repriced.row);
        let charged = self
            .update_and_count(
                &success.settings,
                &success.identity,
                &RowWrite {
                    position: write.position,
                    request: &repriced,
                    request_id: guardrail_request_id.as_deref(),
                },
                repeat,
                &mut None,
                &[],
                outcome,
                host,
                redis,
            )
            .await?;
        Ok(charged.then_some(repeat))
    }

    /// `_update_database_and_spend_counters`: write the row, release the reservation when nothing
    /// was charged, then the counters, which settle the reservation and whose failure
    /// invalidates the reserved counters and is raised.
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per value the Python function takes"
    )]
    async fn update_and_count(
        &self,
        settings: &Settings,
        identity: &Identity,
        write: &RowWrite<'_>,
        cost: f64,
        reservation: &mut Option<BudgetReservation>,
        window_resets: &[Option<String>],
        outcome: &BatchOutcome,
        host: &Host,
        redis: Option<&RequestRedis>,
    ) -> Result<bool, Failed> {
        outcome.write_started.store(true, Ordering::Relaxed);
        let written = self
            .update_database(settings, identity, write, Some(cost), host)
            .await;
        if !written.charged {
            self.release(host, reservation, window_resets).await;
            return Ok(false);
        }
        let charge = Charge {
            identity,
            row: write.request,
            cost,
            customer_hierarchy: true,
        };
        if let Err(error) = self
            .count(
                host,
                redis,
                &charge,
                written,
                reservation.as_mut(),
                window_resets,
            )
            .await
        {
            if let Some(held) = reservation.as_mut() {
                post_call::invalidate(self.budgets.store(), held).await;
                held.finalized = Some(true);
            }
            return Err(error);
        }
        Ok(true)
    }

    /// `increment_spend_counters` on the ledger path: the whole post-call counter step in
    /// `charge_counters`, which reconciles the held reservation first; its stamps and
    /// finalization are copied onto the reservation. The increments ride `redis`, the request's
    /// post-call batch, so a later reader in the request can tell whether its own charge landed.
    async fn count(
        &self,
        host: &Host,
        redis: Option<&RequestRedis>,
        charge: &Charge<'_>,
        written: Written,
        reservation: Option<&mut BudgetReservation>,
        window_resets: &[Option<String>],
    ) -> Result<(), Failed> {
        let step = counter_step(charge, written, window_resets);
        let outcome = self
            .budgets
            .charge_counters(host, redis, &step, reservation.as_deref())
            .await;
        if let Some(held) = reservation {
            if let Some(entries) = held.entries.as_mut() {
                for (entry, stamp) in entries.iter_mut().zip(&outcome.stamps) {
                    if let (Some(entry), Some(stamp)) = (entry.as_mut(), stamp) {
                        entry.applied_adjustment = Some(*stamp);
                    }
                }
            }
            if outcome.finalized {
                held.finalized = Some(true);
            }
        }
        outcome.error.map_or(Ok(()), Err)
    }

    /// `DBSpendUpdateWriter.update_database` on the ledger path. A failure is logged and still
    /// counts as charged, as the Python writer does, and keeps what the counter step needs.
    async fn update_database(
        &self,
        settings: &Settings,
        identity: &Identity,
        write: &RowWrite<'_>,
        response_cost: Option<f64>,
        host: &Host,
    ) -> Written {
        if settings.disable_spend_updates {
            return Written::carried(write.request);
        }
        let billable = response_cost.filter(|cost| cost.is_finite());
        if response_cost.is_some() && billable.is_none() {
            tracing::warn!(
                ?response_cost,
                "Spend tracking - recording a non-finite response_cost as 0"
            );
        }
        let built = self.build(settings, write);
        let skipped = built.as_ref().is_ok_and(|row| {
            settings.disable_responses_id_security
                && RESPONSES_ID_CALL_TYPES.contains(&row.call_type.as_str())
                && row.status == SpendLogStatus::Success
                && billable.is_none_or(|cost| cost == 0.0)
        });
        let token = hashed_token(identity.token.as_deref());
        let accounting = host
            .call(AccountingContext {
                position: write.position,
                built: match &built {
                    Ok(row) if !skipped => Some(AccountingRow::from(row)),
                    _ => None,
                },
                token: token.clone(),
                team_id: identity.team_id.clone(),
                user_id: identity.user_id.clone(),
                org_id: identity.org_id.clone(),
                end_user_id: identity.end_user_id.clone(),
                charges: billable.is_some_and(|cost| cost != 0.0),
                warm_committed: write.request.committed_tracked.is_none()
                    && billable.is_some_and(|cost| cost > 0.0),
            })
            .await;
        let result = match (built, accounting) {
            (Err(error), accounting) => {
                let written = match accounting {
                    Ok(accounting) => Written {
                        charged: true,
                        customer_windows: accounting.customer_windows,
                        committed_tracked: write
                            .request
                            .committed_tracked
                            .clone()
                            .unwrap_or(accounting.committed_tracked),
                    },
                    Err(_) => Written::carried(write.request),
                };
                Err((error, written))
            }
            (Ok(_), Err(error)) => Err((error.into(), Written::carried(write.request))),
            (Ok(row), Ok(accounting)) => {
                let written = Written {
                    charged: true,
                    customer_windows: accounting.customer_windows.clone(),
                    committed_tracked: write
                        .request
                        .committed_tracked
                        .clone()
                        .unwrap_or_else(|| accounting.committed_tracked.clone()),
                };
                if skipped {
                    return written;
                }
                match self
                    .write_row(
                        settings, identity, write, row, billable, token, accounting, host,
                    )
                    .await
                {
                    Ok(charged) => Ok(Written { charged, ..written }),
                    Err(error) => Err((error, written)),
                }
            }
        };
        result.unwrap_or_else(|(error, written)| {
            tracing::error!(
                %error,
                response_cost,
                user_id = identity.user_id,
                team_id = identity.team_id,
                "Spend tracking - update_database failed. Spend log insertion may not have completed for this request."
            );
            written
        })
    }

    /// The row the builder makes from the row's input, under the typed request id. A call with
    /// no id is written as the Python row writes it, under the builder's `None` id.
    fn build(&self, settings: &Settings, write: &RowWrite<'_>) -> Result<SpendLogRow, Error> {
        let request_id = write.request_id;
        let mut input = write.request.row.clone();
        input
            .as_object_mut()
            .ok_or("the row input is not a JSON object")?
            .insert("request_id".to_owned(), json!(request_id));
        let builder = SpendLogRowBuilder::new(
            RowConfig::from(&settings.row_config),
            std::sync::Arc::clone(&self.tokens),
        );
        Ok(builder.build_row(&RowInput::from_value(&json::from_json(input))?)?)
    }

    /// The ledger columns, the record, then the post-write host call. Returns whether the
    /// request is charged: false only when an earlier poll owns a job's charge.
    #[expect(
        clippy::too_many_arguments,
        reason = "the row, its resolved accounting and who it is charged to"
    )]
    async fn write_row(
        &self,
        settings: &Settings,
        identity: &Identity,
        write: &RowWrite<'_>,
        mut row: SpendLogRow,
        billable: Option<f64>,
        token: Option<String>,
        accounting: crate::host::Accounting,
        host: &Host,
    ) -> Result<bool, Error> {
        if let Some(org_id) = identity.org_id.as_ref().filter(|id| !id.is_empty()) {
            row.organization_id.clone_from(org_id);
        }
        if let Some(team_id) = identity.team_id.as_ref().filter(|id| !id.is_empty()) {
            row.team_id.clone_from(team_id);
        }
        let mut entities = vec![
            (
                "api_key".to_owned(),
                persistable_api_key(token.as_deref(), &row.api_key),
            ),
            ("user".to_owned(), identity.user_id.clone()),
            ("end_user".to_owned(), identity.end_user_id.clone()),
            ("team_id".to_owned(), identity.team_id.clone()),
            ("organization_id".to_owned(), identity.org_id.clone()),
            ("project_id".to_owned(), identity.project_id.clone()),
        ];
        entities.extend(accounting.customer_entities);
        entities.push(("agent_id".to_owned(), row.agent_id.clone()));
        entities.push((
            "proxy_budget_name".to_owned(),
            settings.proxy_budget_name.clone(),
        ));
        let charges = billable.is_some_and(|cost| cost != 0.0);
        let columns = row.ledger_columns(&LedgerInput {
            billable_cost: billable,
            org_id: None,
            team_id: None,
            delivery_id: uuid::Uuid::new_v4().simple().to_string(),
            entities,
            model_access_groups: accounting.model_access_groups,
            windows: if charges {
                accounting.windows
            } else {
                Vec::new()
            },
            savings: accounting.savings.unwrap_or_default(),
        })?;
        let keep = !settings.disable_spend_logs
            && !(row.status == SpendLogStatus::Failure && settings.disable_error_logs);
        let mut written = row.clone();
        written.spend = Some(columns.spend);
        let accounting_payload = row
            .settles_once()
            .then(|| columns.accounting_payload.clone());
        let recorded = self
            .record(
                row,
                columns,
                keep,
                accounting.chains_through_db,
                settings.retention().as_ref(),
            )
            .await?;
        if matches!(recorded, Recorded::Owned) {
            return Ok(false);
        }
        let claimed = matches!(recorded, Recorded::Claimed);
        if let Err(error) = host
            .call(AfterWrite {
                position: write.position,
                row: written,
                token,
                team_id: identity.team_id.clone(),
                billable_cost: billable.unwrap_or(0.0),
                spend_logs_kept: keep,
                claimed,
                accounting_payload: accounting_payload.filter(|_| claimed),
            })
            .await
        {
            tracing::debug!(%error, "the post-write host call failed");
        }
        Ok(true)
    }

    /// `_record_spend_log`: a settle-once job row claims its charge now, every other row is
    /// queued, as an accounting-only row when request logs are off. A Responses turn the next
    /// turn replays from the database asks the monitor to flush at once.
    async fn record(
        &self,
        row: SpendLogRow,
        columns: LedgerColumns,
        keep: bool,
        chains_through_db: bool,
        retention: Option<&Retention>,
    ) -> Result<Recorded, Error> {
        if row.settles_once() {
            return self.claim(row, columns, keep, retention).await;
        }
        let entry = QueuedSpendLog::new(row, Some(columns), keep)?;
        self.writer.enqueue(vec![entry], false);
        if chains_through_db {
            self.writer.request_flush();
        }
        Ok(Recorded::Queued)
    }

    /// `_claim_settle_once_cost_row` for a ledger row: the job's delivery row is the claim, so
    /// the poll whose insert lands owns the charge. A $0 poll records nothing.
    async fn claim(
        &self,
        row: SpendLogRow,
        columns: LedgerColumns,
        keep: bool,
        retention: Option<&Retention>,
    ) -> Result<Recorded, Error> {
        if columns.spend <= 0.0 {
            return Ok(Recorded::Queued);
        }
        let entry = QueuedSpendLog::new(row, Some(columns), keep)?;
        match self.writer.claim(&entry, retention).await {
            Ok(true) => Ok(Recorded::Claimed),
            Ok(false) => Ok(Recorded::Owned),
            Err(error) => {
                tracing::warn!(%error, request_id = entry.request_id(), "Could not claim a job's cost row, queueing it");
                self.writer.enqueue(vec![entry], false);
                Ok(Recorded::Queued)
            }
        }
    }

    /// `async_post_call_failure_hook`: release the reservation unless a completed cost settles
    /// it, then one row for the final attempt and one per billed retry attempt. The failure
    /// logging already sent the request's post-call batch, so a charge declared here sends it
    /// again instead of waiting out its deadline with the client already holding the error.
    pub(crate) async fn on_failure(
        &self,
        failure: Failure,
        fields: &Typed,
        reservation: &mut Option<BudgetReservation>,
        host: &Host,
        redis: Option<&RequestRedis>,
    ) {
        let mut failure = failure;
        let completed_cost = fields
            .cost
            .as_ref()
            .and_then(|cost| cost.cost_finalized)
            .unwrap_or(false);
        let disabled = failure.settings.disable_spend_updates;
        let resets = failure.window_resets.clone();
        if !completed_cost || disabled {
            self.release(host, reservation, &resets).await;
            if let Some(settled) = reservation.as_ref() {
                for row in &mut failure.rows {
                    stamp_stored_reservation(row, settled);
                }
            }
        }
        if (completed_cost && disabled) || !failure.track {
            return;
        }
        if failure.enrich
            && let Some(enriched) = self.enrich(host, &failure.identity, true).await
        {
            for row in &mut failure.rows {
                enrich_row(row, &enriched);
            }
        }
        let costs: Vec<Option<f64>> = failure_costs(fields)
            .into_iter()
            .chain(std::iter::repeat(None))
            .zip(failure.rows.iter_mut())
            .map(|(cost, row)| cost.map(|cost| customer_margin::on_failure_row(row, cost)))
            .collect();
        let request_ids = failure_request_ids(fields);
        let writes: Vec<(RowWrite<'_>, Option<f64>)> = failure
            .rows
            .iter()
            .enumerate()
            .map(|(position, request)| {
                (
                    RowWrite {
                        position,
                        request,
                        request_id: request_ids.get(position).copied().flatten(),
                    },
                    costs.get(position).copied().flatten(),
                )
            })
            .collect();
        if completed_cost {
            let outcome = BatchOutcome::default();
            for (write, cost) in &writes {
                if let Err(error) = self
                    .update_and_count(
                        &failure.settings,
                        &failure.identity,
                        write,
                        cost.unwrap_or(0.0),
                        reservation,
                        &resets,
                        &outcome,
                        host,
                        redis,
                    )
                    .await
                {
                    tracing::error!(%error, "Failed to write the spend log for a failure with a completed cost");
                }
            }
            if let Some(redis) = redis {
                redis.flush_post().await;
            }
            return;
        }
        let mut counted = Vec::new();
        for (write, cost) in &writes {
            let written = self
                .update_database(&failure.settings, &failure.identity, write, *cost, host)
                .await;
            let cost = cost.unwrap_or(0.0);
            if written.charged && cost > 0.0 {
                counted.push((write.request, cost, written));
            }
        }
        if disabled {
            return;
        }
        let steps: Vec<CounterStep> = counted
            .into_iter()
            .map(|(row, cost, written)| {
                let charge = Charge {
                    identity: &failure.identity,
                    row,
                    cost,
                    customer_hierarchy: false,
                };
                counter_step(&charge, written, &[])
            })
            .collect();
        self.count_failure_rows(host, steps).await;
    }

    /// `_increment_failure_spend_counters`: the failure rows' counters, sent before the client sees
    /// the error. The proxy runs them on the request's post-call batch, so a counter store slower
    /// than its wait finishes them in the background with the host still answering. The
    /// reservation was already released, so every counter takes the full row cost.
    async fn count_failure_rows(&self, host: &Host, steps: Vec<CounterStep>) {
        if steps.is_empty() {
            return;
        }
        if let Err(error) = host.call(ChargeFailureCounters { steps }).await {
            tracing::error!(%error, "Failed to raise spend counters for the failure spend rows");
        }
    }
}

/// The counter step for one charged row: its owners, tags and access groups, the cost, and what
/// `update_database` left for the counters to charge.
fn counter_step(
    charge: &Charge<'_>,
    written: Written,
    window_resets: &[Option<String>],
) -> CounterStep {
    let identity = charge.identity;
    CounterStep {
        token: identity.token.clone(),
        team_id: identity.team_id.clone(),
        user_id: identity.user_id.clone(),
        org_id: identity.org_id.clone(),
        end_user_id: identity.end_user_id.clone(),
        project_id: identity.project_id.clone(),
        tags: policy::tags(&charge.row.row).map(|tags| tags.into_iter().map(Some).collect()),
        model_access_groups: Some(
            charge
                .row
                .model_access_groups
                .iter()
                .cloned()
                .map(Some)
                .collect(),
        ),
        response_cost: Some(charge.cost),
        committed_tracked: written.committed_tracked,
        customer_windows: written.customer_windows,
        customer_scopes: if charge.customer_hierarchy {
            charge.row.customer_hierarchy.clone()
        } else {
            Vec::new()
        },
        window_resets: window_resets.to_vec(),
    }
}

/// Each failure row's cost: the final attempt's `failure_row_response_cost`, then each billed
/// retry attempt's own.
fn failure_costs(fields: &Typed) -> Vec<Option<f64>> {
    std::iter::once(
        fields
            .cost
            .as_ref()
            .and_then(|cost| cost.failure_row_response_cost),
    )
    .chain(fields.attempts.iter().map(|attempt| attempt.response_cost))
    .collect()
}

/// Each failure row's request id, in the same order as [`failure_costs`].
fn failure_request_ids(fields: &Typed) -> Vec<Option<&str>> {
    std::iter::once(fields.spend_log_id.as_ref().and_then(|id| id.id.as_deref()))
        .chain(fields.attempts.iter().map(|attempt| {
            attempt
                .spend_log_id
                .as_ref()
                .and_then(|id| id.id.as_deref())
        }))
        .collect()
}

/// `_should_track_cost_callback`.
fn tracked(success: &Success) -> bool {
    if success.settings.disable_spend_updates {
        return false;
    }
    let identity = &success.identity;
    identity.agent_id.is_some()
        || identity.app_id.is_some()
        || identity.token.is_some()
        || identity.user_id.is_some()
        || identity.team_id.is_some()
        || identity.end_user_id.is_some()
        || policy::tracks_unattributed(success.call_type.as_deref())
}

/// Why a priced call could not be charged: it lost its cost.
fn unpriced(success: &Success, fields: &Typed) -> String {
    let model = logged_model(&success.row.row);
    let reason = if policy::payload(&success.row.row).is_none() {
        "the call logged no standard logging payload"
    } else {
        fields
            .cost
            .as_ref()
            .and_then(|cost| cost.response_cost_failure_debug_info.as_deref())
            .unwrap_or("the logging payload carries no cost failure details")
    };
    format!("cost tracking failed for model {model:?}: {reason}")
}

/// The cost callback's `except`: the failed tracking alert, fired and forgotten.
async fn failed_tracking_alert(success: &Success, error: &str, host: &Host) {
    tracing::error!(error, "Error in tracking cost callback");
    let failing_model = logged_model(&success.row.row);
    let call_type = success.call_type.as_deref().unwrap_or_default();
    let error_message = format!(
        "Error in tracking cost callback - {error}\nmodel: {failing_model}\ncall_type: {call_type}\nAdd custom pricing - https://docs.litellm.ai/docs/proxy/custom_pricing"
    );
    if let Err(error) = host
        .call(FailedTrackingAlert {
            error_message,
            failing_model,
        })
        .await
    {
        tracing::debug!(%error, "the failed tracking alert host call failed");
    }
}

/// The work `update_cache` and the cost callback leave to Python once a request is charged.
async fn after_charge(success: &Success, charged: f64, host: &Host) {
    let identity = &success.identity;
    if let Err(error) = host
        .call(AfterCharge {
            token: identity.token.clone(),
            key_alias: identity.key_alias.clone(),
            user_id: identity.user_id.clone(),
            end_user_id: identity.end_user_id.clone(),
            team_id: identity.team_id.clone(),
            tags: policy::tags(&success.row.row),
            response_cost: charged,
            end_user_max_budget: success.end_user_max_budget,
        })
        .await
    {
        tracing::debug!(%error, "the after-charge host call failed");
    }
}
