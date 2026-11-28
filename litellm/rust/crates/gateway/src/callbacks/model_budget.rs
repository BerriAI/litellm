//! Per-model spend checks before routing and charging after a successful attempt.

use std::{borrow::Cow, sync::Arc};

use arc_swap::ArcSwap;
use chrono::{DateTime, Utc};
use futures_util::future::BoxFuture;
use jiff::Timestamp;
use litellm_callbacks::{
    Abort, Callback, CallbackError, Field, FieldSet, Hook, HookInput, HookOutput, Manifest, Scalar,
    Selection, Subscription, Verdict, typed,
};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{Credential, Holder, Named, Team};
use litellm_gateway_inference::record;
use litellm_model_catalog::CostMap;
use litellm_redis_store::model_budgets::{Counter, ModelBudgets, Period, Scope};
use litellm_repositories::SpendLogRepository;
use litellm_router_core::Router;
use serde_json::json;

use super::{
    Registered,
    budget_check::over_budget,
    fields,
    model_budget_resolution::{Entries, latest, resolve},
};
use crate::host::{FreeModels, NativeRows};

/// Reads the same windows as the Python limiter, without reserving spend.
pub(crate) struct ModelBudget {
    manifest: Manifest,
    counters: Arc<ModelBudgets>,
    shared_windows: Option<SpendLogRepository>,
    router: Router,
    prices: Arc<ArcSwap<CostMap>>,
    free: Arc<FreeModels>,
}

struct Entity<'a> {
    scope: Scope,
    id: Cow<'a, str>,
    entries: &'a Entries,
}

impl<'a> Entity<'a> {
    fn new(scope: Scope, id: impl Into<Cow<'a, str>>, entries: &'a Entries) -> Option<Self> {
        (!entries.is_empty()).then(|| Self {
            scope,
            id: id.into(),
            entries,
        })
    }
}

pub(crate) fn member_entries(holder: &Holder<NativeRows>) -> Option<&Entries> {
    holder.user().and_then(Named::stored)?;
    let Team::Row(team) = holder.team()? else {
        return None;
    };
    let entries = holder
        .membership()
        .map(|member| &member.budget_extras.model_max_budget)
        .filter(|entries| !entries.is_empty())
        .unwrap_or(&team.member_budget_extras.model_max_budget);
    (!entries.is_empty()).then_some(entries)
}

fn entities(identity: &Authenticated<NativeRows>) -> Vec<Entity<'_>> {
    let holder = &identity.affiliation.holder;
    let team = holder.team().and_then(|team| match team {
        Team::Row(row) => Some(row),
        _ => None,
    });
    let key = match &identity.credential {
        Credential::Key { key } => identity
            .hash()
            .and_then(|hash| Entity::new(Scope::Key, hash.as_str(), key.model_max_budget())),
        Credential::MasterKey => None,
    };
    let user = holder.user().and_then(Named::stored);
    let member = user.zip(team).and_then(|(user, team)| {
        let entries = member_entries(holder)?;
        Entity::new(
            Scope::TeamMember,
            format!(
                "{}:{}:{}",
                user.user_id.chars().count(),
                user.user_id,
                team.team_id
            ),
            entries,
        )
    });
    [
        key,
        user.and_then(|user| {
            Entity::new(Scope::User, user.user_id.as_str(), &user.model_max_budget)
        }),
        team.and_then(|team| {
            Entity::new(Scope::Team, team.team_id.as_str(), &team.model_max_budget)
        }),
        member,
        identity.end_user.end_user_id.as_deref().and_then(|id| {
            Entity::new(
                Scope::EndUser,
                id,
                &identity.end_user.budget_extras.model_max_budget,
            )
        }),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn failure(error: impl std::fmt::Display) -> CallbackError {
    CallbackError::Failed(error.to_string())
}

fn open(start: DateTime<Utc>, period: Period, now: Timestamp) -> bool {
    let started = Timestamp::new(start.timestamp(), start.timestamp_subsec_nanos() as i32);
    started
        .ok()
        .and_then(|started| started.checked_add(period.duration(started)?).ok())
        .is_some_and(|end| now < end)
}

impl ModelBudget {
    pub(crate) fn new(
        counters: Arc<ModelBudgets>,
        shared_windows: Option<SpendLogRepository>,
        router: Router,
        prices: Arc<ArcSwap<CostMap>>,
        free: Arc<FreeModels>,
    ) -> Arc<Self> {
        Arc::new(Self {
            manifest: Manifest {
                selection: Selection::named("model_budget"),
                subscriptions: vec![
                    Subscription {
                        reads: FieldSet::of(&[Field::Model]),
                        may_abort: true,
                        ..Subscription::on(Hook::PreCall)
                    },
                    Subscription::on(Hook::AttemptEnded),
                ],
                has_request_state: true,
            },
            counters,
            shared_windows,
            router,
            prices,
            free,
        })
    }

    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        Registered::for_caller(self.clone(), Authenticated::clone)
    }

    fn shared(&self, scope: Scope) -> Option<&SpendLogRepository> {
        (scope != Scope::EndUser)
            .then_some(self.shared_windows.as_ref())
            .flatten()
    }

    async fn current_spend(
        &self,
        counter: &Counter<'_>,
        requested: &str,
        period: Period,
        now: Timestamp,
        max: f64,
    ) -> Result<f64, CallbackError> {
        let local = self
            .counters
            .spend_with_legacy(counter, requested, now)
            .await
            .map_err(failure)?;
        if local >= max {
            return Ok(local);
        }
        let Some(database) = self.shared(counter.scope) else {
            return Ok(local);
        };
        match database
            .read_model_budget_window(&counter.spend(), counter.duration)
            .await
        {
            Ok(Some((start, spend))) if open(start, period, now) => Ok(local.max(spend)),
            Ok(_) => Ok(local),
            Err(error) => {
                tracing::warn!(%error, "Could not read the shared model budget window");
                Ok(local)
            }
        }
    }

    pub(crate) async fn exceeded(
        &self,
        model: &str,
        identity: &Authenticated<NativeRows>,
        only: Option<Scope>,
    ) -> Result<Option<(Scope, bool, Abort)>, CallbackError> {
        if self.free.is_free(model) {
            return Ok(None);
        }
        let prices = self.prices.load();
        let catalog = prices.catalog();
        let expanded = latest(catalog, model);
        let registry = self.router.registry();
        let team = identity.affiliation.holder.team().map(Team::id);
        let entities = entities(identity);
        let key = entities
            .iter()
            .find(|entity| entity.scope == Scope::Key)
            .map(|entity| entity.entries);
        let now = Timestamp::now();
        for entity in entities
            .into_iter()
            .filter(|entity| only.is_none_or(|scope| scope == entity.scope))
        {
            for requested in std::iter::once(model).chain(expanded.as_deref()) {
                let key = if entity.scope == Scope::Team {
                    key
                } else {
                    None
                };
                let Some(budget) =
                    resolve(entity.entries, key, catalog, &registry, requested, team)
                else {
                    continue;
                };
                let Some(max) = budget.max else {
                    continue;
                };
                let counter = Counter {
                    scope: entity.scope,
                    id: &entity.id,
                    model: budget.model,
                    duration: budget.duration,
                };
                let spend = self
                    .current_spend(&counter, requested, budget.period, now, max)
                    .await?;
                if spend >= max {
                    let message = format!(
                        "Budget has been exceeded for model={}: current cost {spend}, max budget {max}",
                        budget.model
                    );
                    return Ok(Some((
                        entity.scope,
                        requested == model,
                        over_budget(serde_json::Map::from_iter([(
                            "message".to_owned(),
                            json!(message),
                        )])),
                    )));
                }
            }
        }
        Ok(None)
    }

    async fn charge(
        &self,
        counter: &Counter<'_>,
        period: Period,
        now: Timestamp,
        cost: f64,
    ) -> Result<(), CallbackError> {
        self.counters
            .charge(counter, period, now, cost)
            .await
            .map_err(failure)?;
        let Some(database) = self.shared(counter.scope) else {
            return Ok(());
        };
        let key = counter.spend();
        let seen = database
            .read_model_budget_window(&key, counter.duration)
            .await
            .map_err(failure)?;
        let start = match seen {
            Some((start, _)) if open(start, period, now) => start,
            _ => {
                let now_utc =
                    DateTime::from_timestamp(now.as_second(), now.subsec_nanosecond() as u32)
                        .ok_or_else(|| failure("Invalid model budget window start"))?;
                database
                    .open_model_budget_window(
                        &key,
                        counter.duration,
                        now_utc,
                        seen.map(|(start, _)| start),
                    )
                    .await
                    .map_err(failure)?;
                let Some((start, _)) = database
                    .read_model_budget_window(&key, counter.duration)
                    .await
                    .map_err(failure)?
                else {
                    return Err(failure("The model budget window did not open"));
                };
                start
            }
        };
        database
            .charge_model_budget_window(&key, counter.duration, start, cost)
            .await
            .map_err(failure)
    }

    async fn succeeded(&self, identity: &Authenticated<NativeRows>) -> HookOutput {
        let Some((model, cost)) = record::read(|entries| {
            if entries.ended() != Some(record::Ended::Served) {
                return None;
            }
            let served = entries.served()?;
            let model = served
                .deployment
                .as_ref()
                .and_then(|deployment| deployment.model_info.team_public_model_name.as_deref())
                .filter(|name| !name.is_empty() && !name.contains('*'))
                .unwrap_or(&served.model_group);
            let Scalar::Float(cost) = fields::cost(entries)?.response_cost? else {
                return None;
            };
            Some((model.to_owned(), cost))
        })
        .flatten() else {
            return Ok(None);
        };
        let prices = self.prices.load();
        let catalog = prices.catalog();
        let registry = self.router.registry();
        let team = identity.affiliation.holder.team().map(Team::id);
        let entities = entities(identity);
        let key = entities
            .iter()
            .find(|entity| entity.scope == Scope::Key)
            .map(|entity| entity.entries);
        let now = Timestamp::now();
        for entity in entities {
            let key = if entity.scope == Scope::Team {
                key
            } else {
                None
            };
            if let Some(budget) = resolve(entity.entries, key, catalog, &registry, &model, team) {
                let counter = Counter {
                    scope: entity.scope,
                    id: &entity.id,
                    model: budget.model,
                    duration: budget.duration,
                };
                if let Err(error) = self.charge(&counter, budget.period, now, cost).await {
                    tracing::error!(%error, "Could not charge a model budget");
                }
            }
        }
        Ok(None)
    }
}

impl Callback for ModelBudget {
    type State = Authenticated<NativeRows>;

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        state: Option<&'a Self::State>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            let Some(identity) = state else {
                return Ok(None);
            };
            match input.hook {
                Hook::PreCall => match input.fields.get::<typed::Model>()? {
                    Some(model) => Ok(self
                        .exceeded(&model, identity, None)
                        .await?
                        .map(|(_, _, abort)| Verdict::Abort(abort))),
                    None => Ok(None),
                },
                Hook::AttemptEnded => self.succeeded(identity).await,
                _ => Ok(None),
            }
        })
    }
}
