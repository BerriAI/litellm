//! The standalone budget reset sweep. Fronted deployments leave scheduling to Python.

mod schedule;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use jiff::{Timestamp, civil::Time, tz::TimeZone};
use litellm_config::{Config, Value as Setting};
use litellm_env::Lookup;
use litellm_error::Report;
use litellm_redis_client::RedisClient;
use litellm_redis_store::{
    Number,
    auth_cache::AuthObject,
    budget_reset_lease::Lease,
    spend_counters::{SpendCounter, SpendCounters},
};
use litellm_repositories::{
    BudgetResetRepository, BudgetWindows, LifetimeBudget, ResetEntity, WindowAdvance, WindowOwner,
};
use rand::Rng;
use serde::Deserialize;
use serde_json::Value;
use strum::VariantArray;

use crate::{PeriodicJob, callbacks::windows::Entity, host::auth::AuthStore};

use self::schedule::{Schedule, stored_timestamp};

const PAGE_SIZE: i64 = 500;

#[derive(Deserialize)]
struct StoredWindow {
    budget_duration: String,
    reset_at: String,
    max_budget: Option<f64>,
}

struct WindowPlan {
    limits: Value,
    windows: Vec<WindowAdvance>,
    counters: BTreeMap<String, (SpendCounter, f64)>,
}

/// The inputs and shared stores a standalone sweep holds for the life of the scheduler.
struct Reset {
    repository: BudgetResetRepository,
    auth: Arc<AuthStore>,
    counters: Arc<SpendCounters>,
    redis: RedisClient,
    schedule: Schedule,
    rollover: bool,
    default_end_user_tier: Option<String>,
}

pub(crate) fn start(
    config: &Config,
    environment: &dyn Lookup,
    repository: BudgetResetRepository,
    auth: Arc<AuthStore>,
    counters: Arc<SpendCounters>,
    redis: RedisClient,
) -> Option<PeriodicJob> {
    let general = &config.general_settings.additional_fields;
    if general
        .get("disable_reset_budget")
        .and_then(Setting::as_bool)
        == Some(true)
    {
        return None;
    }
    let lite = &config.litellm_settings.additional_fields;
    let timezone = lite
        .get("timezone")
        .and_then(Setting::as_str)
        .and_then(|timezone| TimeZone::get(timezone).ok())
        .unwrap_or(TimeZone::UTC);
    let wall_time = lite
        .get("budget_reset_time")
        .and_then(Setting::as_str)
        .filter(|time| !time.is_empty())
        .map(str::parse)
        .transpose();
    let wall_time = match wall_time {
        Ok(time) => time.unwrap_or(Time::MIN),
        Err(error) => {
            tracing::warn!(%error, "Budget reset scheduler disabled because its wall time is invalid");
            return None;
        }
    };
    let interval = |key, env, default| {
        general
            .get(key)
            .and_then(Setting::as_u64)
            .or_else(|| environment.get(env)?.parse().ok())
            .unwrap_or(default)
    };
    let min = interval(
        "proxy_budget_rescheduler_min_time",
        "PROXY_BUDGET_RESCHEDULER_MIN_TIME",
        597,
    );
    let max = interval(
        "proxy_budget_rescheduler_max_time",
        "PROXY_BUDGET_RESCHEDULER_MAX_TIME",
        605,
    );
    let jitter = rand::thread_rng().gen_range(0..=max.saturating_sub(min).min(30));
    let reset = Arc::new(Reset {
        repository,
        auth,
        counters,
        redis,
        schedule: Schedule {
            timezone,
            wall_time,
        },
        rollover: lite.get("budget_rollover").and_then(Setting::as_bool) == Some(true),
        default_end_user_tier: lite
            .get("max_end_user_budget_id")
            .and_then(Setting::as_str)
            .map(str::to_owned),
    });
    Some(PeriodicJob::start(
        Duration::from_secs(min.saturating_add(jitter).max(1)),
        move || {
            let reset = reset.clone();
            async move { reset.run().await }
        },
    ))
}

impl Reset {
    async fn run(&self) {
        let lease = match Lease::acquire(&self.redis).await {
            Ok(Some(lease)) => lease,
            Ok(None) => return,
            Err(error) => {
                tracing::warn!(error = %Report::new(&error), "Skipping budget reset without the Redis lease");
                return;
            }
        };
        if tokio::time::timeout_at(lease.deadline(), self.sweep(&lease))
            .await
            .is_err()
        {
            tracing::warn!("Budget reset exceeded its lease deadline");
            return;
        }
        if let Err(error) = lease.release().await {
            tracing::warn!(error = %Report::new(&error), "Budget reset lease could not be released");
        }
    }

    async fn sweep(&self, lease: &Lease<'_>) {
        let now = Timestamp::now();
        if !self.reset_lifetimes(lease, now).await {
            return;
        }
        for &owner in WindowOwner::VARIANTS {
            let mut cursor = String::new();
            loop {
                let page = match self.repository.window_page(owner, &cursor, PAGE_SIZE).await {
                    Ok(page) if page.is_empty() => break,
                    Ok(page) => page,
                    Err(error) => {
                        tracing::warn!(?owner, error = %Report::new(&error), "Reading budget windows for reset failed");
                        break;
                    }
                };
                for row in &page {
                    let Some(plan) = self.plan(owner, row, now).await else {
                        continue;
                    };
                    if !matches!(lease.is_current().await, Ok(true)) {
                        tracing::warn!(
                            "Stopping budget reset because the Redis lease could not be verified"
                        );
                        return;
                    }
                    match self
                        .repository
                        .advance_windows(owner, &row.id, &row.limits, &plan.limits, &plan.windows)
                        .await
                    {
                        Ok(true) => self.finish(owner, &row.id, plan).await,
                        Ok(false) => (),
                        Err(error) => {
                            tracing::warn!(?owner, error = %Report::new(&error), "Advancing budget windows failed")
                        }
                    }
                }
                cursor.clone_from(&page.last().expect("page is not empty").id);
                if page.len() < PAGE_SIZE as usize {
                    break;
                }
            }
        }
    }

    async fn reset_lifetimes(&self, lease: &Lease<'_>, now: Timestamp) -> bool {
        let Some(now_utc) =
            chrono::DateTime::from_timestamp(now.as_second(), now.subsec_nanosecond() as u32)
        else {
            return false;
        };
        for &owner in WindowOwner::VARIANTS {
            let mut cursor = String::new();
            loop {
                let page = match self
                    .repository
                    .lifetime_page(owner, &cursor, now_utc, PAGE_SIZE)
                    .await
                {
                    Ok(page) => page,
                    Err(error) => {
                        tracing::warn!(?owner, error = %Report::new(&error), "Reading lifetime budgets for reset failed");
                        break;
                    }
                };
                let Some(last) = page.last() else { break };
                cursor.clone_from(&last.id);
                for budget in page {
                    let previous = budget.reset_at.and_then(|time| {
                        let time = time.and_utc();
                        Timestamp::new(time.timestamp(), time.timestamp_subsec_nanos() as i32).ok()
                    });
                    let Some((end, _)) =
                        self.schedule
                            .advance_lifetime(&budget.duration, previous, now)
                    else {
                        continue;
                    };
                    let Some(end) = chrono::DateTime::from_timestamp(
                        end.as_second(),
                        end.subsec_nanosecond() as u32,
                    ) else {
                        continue;
                    };
                    if !matches!(lease.is_current().await, Ok(true)) {
                        tracing::warn!(
                            "Stopping budget reset because the Redis lease could not be verified"
                        );
                        return false;
                    }
                    if let Err(error) = self.reset_lifetime(owner, &budget, end).await {
                        tracing::warn!(?owner, error = %Report::new(&error), "Advancing lifetime budget failed");
                    }
                }
            }
        }
        true
    }

    async fn reset_lifetime(
        &self,
        owner: WindowOwner,
        budget: &LifetimeBudget,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), litellm_db::Error> {
        let id = &budget.id;
        let (counter, auth) = match owner {
            WindowOwner::Key => (
                SpendCounter::Key {
                    token: id.clone(),
                    window: None,
                },
                AuthObject::Key(id.clone()),
            ),
            WindowOwner::User => (SpendCounter::User(id.clone()), AuthObject::User(id.clone())),
            WindowOwner::Team => (
                SpendCounter::Team {
                    team_id: id.clone(),
                    window: None,
                },
                AuthObject::Team(id.clone()),
            ),
            WindowOwner::Budget => {
                let entities = self
                    .repository
                    .advance_tier(
                        budget,
                        end,
                        self.rollover,
                        self.default_end_user_tier.as_deref(),
                    )
                    .await?;
                for entity in entities {
                    self.evict_tier_entity(entity).await;
                }
                self.auth
                    .invalidate_reset_entry(AuthObject::TeamMemberDefaultBudget(id.clone()))
                    .await;
                if self.default_end_user_tier.as_ref() == Some(id) {
                    self.auth
                        .invalidate_reset_entry(AuthObject::DefaultEndUserBudget(id.clone()))
                        .await;
                }
                return Ok(());
            }
        };
        let cap = budget.max_budget.filter(|cap| cap.is_finite());
        let room = cap
            .zip(budget.rollover_max_budget)
            .and_then(|(cap, ceiling)| {
                (cap > 0.0 && ceiling > cap && ceiling.is_finite()).then_some(ceiling - cap)
            });
        let active = room.map_or(0.0, |room| {
            budget.rollover_credit.unwrap_or(0.0).clamp(0.0, room)
        });
        let effective = cap.unwrap_or(0.0) + active;
        let credit = room.map(|room| (effective - budget.spend).clamp(0.0, room));
        let decrement = if self.rollover && cap.is_some() {
            budget.spend.min(effective)
        } else {
            budget.spend
        };
        if self
            .repository
            .advance_lifetime(owner, budget, end, decrement, credit)
            .await?
        {
            self.evict_lifetime_counter(counter).await;
            self.auth.invalidate_reset_entry(auth).await;
        }
        Ok(())
    }

    async fn evict_tier_entity(&self, entity: ResetEntity) {
        let ResetEntity { kind, id, team_id } = entity;
        let (counter, auth, extra) = match (kind.as_str(), team_id) {
            ("org", _) => (
                SpendCounter::Org(id.clone()),
                AuthObject::Org(id.clone()),
                Some(AuthObject::OrgWithBudget(id)),
            ),
            ("member", Some(team_id)) => (
                SpendCounter::TeamMember {
                    team_id: team_id.clone(),
                    user_id: id.clone(),
                },
                AuthObject::TeamMembership {
                    team_id: team_id.clone(),
                    user_id: id.clone(),
                },
                Some(AuthObject::TeamMembershipReservation {
                    team_id,
                    user_id: id,
                }),
            ),
            ("key", _) => (
                SpendCounter::Key {
                    token: id.clone(),
                    window: None,
                },
                AuthObject::Key(id),
                None,
            ),
            ("project", _) => (
                SpendCounter::Project(id.clone()),
                AuthObject::Project(id),
                None,
            ),
            ("end_user", _) => (
                SpendCounter::EndUser(id.clone()),
                AuthObject::EndUser(id),
                None,
            ),
            _ => return,
        };
        self.evict_lifetime_counter(counter).await;
        self.auth.invalidate_reset_entry(auth).await;
        if let Some(extra) = extra {
            self.auth.invalidate_reset_entry(extra).await;
        }
    }

    async fn evict_lifetime_counter(&self, counter: SpendCounter) {
        let _ = self.counters.delete_floor(&counter);
        let twins: Vec<_> = counter
            .clone()
            .committed()
            .into_iter()
            .chain([counter])
            .collect();
        if let Err(error) = self.counters.delete(&twins).await {
            tracing::warn!(error = %Report::new(&error), "Committed lifetime budget counter could not be evicted");
        }
    }

    async fn plan(
        &self,
        owner: WindowOwner,
        row: &BudgetWindows,
        now: Timestamp,
    ) -> Option<WindowPlan> {
        let mut limits: Vec<Value> = serde_json::from_str(row.limits.get()).ok()?;
        let mut windows = Vec::new();
        let mut counters = BTreeMap::new();
        for value in &mut limits {
            let Ok(stored) = StoredWindow::deserialize(&*value) else {
                continue;
            };
            let Some((end, start)) = stored_timestamp(&stored.reset_at).and_then(|previous| {
                self.schedule
                    .advance(&stored.budget_duration, previous, now)
            }) else {
                continue;
            };
            let starts_at = chrono::DateTime::from_timestamp(
                start.as_second(),
                u32::try_from(start.subsec_nanosecond()).ok()?,
            )?;
            let fields = value.as_object_mut()?;
            fields.insert("reset_at".to_owned(), Value::String(end.to_string()));
            fields.insert("window_start".to_owned(), Value::String(start.to_string()));
            if let Some(entity) = window_entity(owner) {
                let counter = entity.counter(&row.id, &stored.budget_duration);
                let spend = if self.rollover
                    && let Some(cap) = stored.max_budget.filter(|cap| cap.is_finite())
                {
                    let current = if let Some((_, staged)) = counters.get(&stored.budget_duration) {
                        *staged
                    } else {
                        self.counters
                            .read(&counter)
                            .await
                            .ok()
                            .and_then(|reading| reading.value())
                            .map_or(0.0, Number::as_f64)
                    };
                    (current - cap).max(0.0)
                } else {
                    0.0
                };
                counters.insert(stored.budget_duration.clone(), (counter, spend));
            }
            windows.push(WindowAdvance {
                duration: stored.budget_duration,
                starts_at,
            });
        }
        (!windows.is_empty()).then_some(WindowPlan {
            limits: Value::Array(limits),
            windows,
            counters,
        })
    }

    async fn finish(&self, owner: WindowOwner, id: &str, plan: WindowPlan) {
        for (counter, spend) in plan.counters.into_values() {
            if let Err(error) = self
                .counters
                .set(&counter, Number::Float(spend), None)
                .await
            {
                tracing::warn!(error = %Report::new(&error), "Committed budget window counter could not be reset");
            }
        }
        let entry = match owner {
            WindowOwner::Key => AuthObject::Key(id.to_owned()),
            WindowOwner::Team => AuthObject::Team(id.to_owned()),
            WindowOwner::User => AuthObject::User(id.to_owned()),
            WindowOwner::Budget => return,
        };
        self.auth.invalidate_reset_entry(entry).await;
    }
}

fn window_entity(owner: WindowOwner) -> Option<Entity> {
    match owner {
        WindowOwner::Key => Some(Entity::Key),
        WindowOwner::Team => Some(Entity::Team),
        WindowOwner::User => Some(Entity::User),
        WindowOwner::Budget => None,
    }
}
