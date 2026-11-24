use std::sync::Arc;

use litellm_callbacks_parallel_request_limiter::Ctx;
use litellm_python_compat::{json::loaded_str, number, truthy::json_truthy, unicode::is_latin1};
use litellm_redis_store::RequestRedis;
use serde_json::Value;

use crate::{
    error::CapacityError,
    host::{CapacityHost, CounterStore, Estimate, Facts, Published, Throttle, Touched},
    model::{
        BORROWER_KINDS, BorrowerKind, BorrowingLimits, Charge, Counter, CounterKind, Descriptor,
        GroupInfo, Increment, KeyFacts, LimiterResponse, LoggedGroup, MODEL_SATURATION_CHECK,
        PRIORITY_CLASS_SHARE_AT_SATURATION, PRIORITY_MODEL, Priority, RateLimit, ReservationOp,
        Settings, Status, TeamShare, TokenRateLimitType, Window,
    },
    priority::{
        allocation, class_share_holder, class_share_kind, counter_value,
        is_team_class_share_holder, normalized_weights, pool_member, priority_class, team_member,
        team_share,
    },
    refusal::{Context, Refusal},
    state::{Hold, ShareState, SharedStash, Stash},
};

pub struct Engine {
    pub limiter: Arc<dyn CounterStore>,
    pub host: Arc<dyn CapacityHost>,
}

pub enum AdmitError {
    Refused(Refusal),
    Failed(CapacityError),
}

impl From<CapacityError> for AdmitError {
    fn from(error: CapacityError) -> Self {
        Self::Failed(error)
    }
}

pub enum Frame {
    Allow,
    Refuse(Refusal),
    Unverifiable(CapacityError),
}

const UNCOUNTABLE_GROUP: &str = "failed to extract field PyIncrement.counter";

const SOCKET_TURN_UNCOUNTED: [&str; 3] = [
    "model_per_project_itpm",
    "model_per_project_otpm",
    "model_per_team_ptu",
];

fn socket_turn_tpm_windows(descriptors: &[Descriptor]) -> Vec<Descriptor> {
    descriptors
        .iter()
        .filter(|descriptor| !SOCKET_TURN_UNCOUNTED.contains(&descriptor.key.as_str()))
        .filter_map(|descriptor| {
            let tokens = descriptor.tokens_per_unit()?;
            Some(Descriptor {
                key: descriptor.key.clone(),
                value: descriptor.value.clone(),
                rate_limit: Some(RateLimit {
                    requests_per_unit: None,
                    tokens_per_unit: Some(tokens),
                    window_size: descriptor
                        .rate_limit
                        .as_ref()
                        .and_then(|limit| limit.window_size),
                }),
            })
        })
        .collect()
}

pub enum PreCall {
    Allow(Option<Published>),
    Refuse(Refusal),
    Raise(CapacityError),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SaturationFallback {
    pub enabled: bool,
    pub invalid: Option<String>,
    pub armed: bool,
}

/// A borrower's share and, when it caps how far it borrows, its ceiling, both on the share's counter.
pub struct BorrowingShare {
    pub borrower: BorrowingLimits,
    pub share: Descriptor,
    pub ceiling: Option<Descriptor>,
}

impl BorrowingShare {
    pub fn enforced(&self, saturated: bool) -> Option<&Descriptor> {
        saturated.then_some(&self.share)
    }

    fn kind(&self) -> BorrowerKind {
        self.borrower.kind
    }
}

fn borrower_of(borrowing: &[BorrowingShare], kind: BorrowerKind) -> Option<&BorrowingShare> {
    borrowing.iter().find(|share| share.kind() == kind)
}

/// The borrowers that hold tokens before the call: all of them once saturated, only the capped ones before.
fn holding(borrowing: &[BorrowingShare], saturated: bool) -> Vec<&BorrowingShare> {
    borrowing
        .iter()
        .filter(|share| saturated || share.ceiling.is_some())
        .collect()
}

/// The first of `items` with the least key, the way Python's `min` picks among ties.
fn first_least<T>(items: impl IntoIterator<Item = (i64, T)>) -> Option<(i64, T)> {
    items
        .into_iter()
        .fold(None, |best, (key, item)| match best {
            Some((least, _)) if least <= key => best,
            _ => Some((key, item)),
        })
}

/// What one borrower holds on the model being admitted: its share once saturated, its ceiling, and the tokens.
struct HeldShare {
    kind: BorrowerKind,
    share: Descriptor,
    enforced: Option<Descriptor>,
    cap: Option<Descriptor>,
    reservation: i64,
}

impl HeldShare {
    fn descriptors(&self) -> impl Iterator<Item = &Descriptor> {
        self.enforced.iter().chain(self.cap.iter())
    }
}

fn held_shares(borrowing: &[BorrowingShare], saturated: bool, reservation: i64) -> Vec<HeldShare> {
    borrowing
        .iter()
        .map(|share| HeldShare {
            kind: share.kind(),
            share: share.share.clone(),
            enforced: share.enforced(saturated).cloned(),
            cap: share.ceiling.clone(),
            reservation: if (saturated || share.ceiling.is_some())
                && share.share.tokens_per_unit().is_some()
            {
                reservation
            } else {
                0
            },
        })
        .collect()
}

/// The descriptor with the counter a full answer reported dropped, so the rest are counted without it.
fn without_counter(descriptor: &Descriptor, counter: &Status) -> Descriptor {
    if descriptor.key != counter.descriptor_key
        || Some(&descriptor.value) != counter.descriptor_value.as_ref()
    {
        return descriptor.clone();
    }
    let mut limits = descriptor.rate_limit.clone().unwrap_or_default();
    if counter.rate_limit_type == "tokens" {
        limits.tokens_per_unit = None;
    } else {
        limits.requests_per_unit = None;
    }
    Descriptor {
        key: descriptor.key.clone(),
        value: descriptor.value.clone(),
        rate_limit: Some(limits),
    }
}

/// The team, key or customer organization a share belongs to, as the dynamic limiter metrics label it.
struct ShareHolder {
    kind: BorrowerKind,
    entity_id: String,
    alias: Option<String>,
}

/// Whose share refused the request: the borrower the reason names, or the class share's team, key or organization.
fn throttled_holder(reason: &str, settings: &Settings, key: &KeyFacts) -> Option<ShareHolder> {
    let kind = if reason == PRIORITY_CLASS_SHARE_AT_SATURATION {
        class_share_kind(class_share_holder(settings, key).as_deref())?
    } else {
        BorrowerKind::by_reason(reason).unwrap_or(BorrowerKind::Team)
    };
    let (entity_id, alias) = match kind {
        BorrowerKind::Team => (key.team_text(), key.team_alias.clone()),
        BorrowerKind::Key => (key.api_key.clone(), key.key_alias.clone()),
        BorrowerKind::CustomerOrg => (key.customer_org_id.clone(), key.customer_org_alias.clone()),
    };
    Some(ShareHolder {
        kind,
        entity_id: entity_id?,
        alias,
    })
}

/// The label and id a class share refusal names. A customer member's slice has no borrower label of its own.
fn class_share_refusal_holder(settings: &Settings, key: &KeyFacts) -> (&'static str, String) {
    match throttled_holder(PRIORITY_CLASS_SHARE_AT_SATURATION, settings, key) {
        Some(holder) => (holder.kind.label(), holder.entity_id),
        None => (
            "Customer member",
            key.customer_member_id
                .clone()
                .unwrap_or_else(|| "None".into()),
        ),
    }
}

type Holds = [Option<Arc<Hold>>; 3];

pub struct Request<'a> {
    pub key: &'a KeyFacts,
    pub call_type: Option<&'a str>,
    pub settings: &'a Settings,
    pub stash: &'a SharedStash,
    pub fallback: SaturationFallback,
    pub redis: Option<&'a RequestRedis>,
    pub facts: &'a Facts,
}

impl Request<'_> {
    pub fn ctx(&self) -> Ctx<'_> {
        Ctx {
            policy: self.settings.policy(),
            request: self.redis,
        }
    }

    pub fn priority(&self) -> Option<&Priority> {
        self.key.priority.as_ref()
    }

    pub fn priority_value(&self) -> Option<Value> {
        self.key.priority_value()
    }
}

fn python_optional(value: Option<&str>) -> &str {
    value.unwrap_or("None")
}

pub fn team_increment(
    descriptor: Option<&Descriptor>,
    amount: i64,
    ttl: i64,
) -> Vec<ReservationOp> {
    let Some(descriptor) = descriptor.filter(|_| amount != 0) else {
        return Vec::new();
    };
    if descriptor.tokens_per_unit().is_none() {
        return Vec::new();
    }
    vec![ReservationOp::Increment {
        counter: descriptor.counter(CounterKind::Tokens),
        amount,
        ttl: Some(ttl),
    }]
}

pub fn window_guarded(
    operations: Vec<ReservationOp>,
    window: Option<&Window>,
) -> Vec<ReservationOp> {
    let Some(window) = window else {
        return operations;
    };
    operations
        .into_iter()
        .map(|operation| match operation {
            ReservationOp::Increment {
                counter,
                amount,
                ttl,
            } if counter == window.counter => ReservationOp::Guarded {
                counter,
                amount,
                ttl,
                expected_window_start: window.start.clone(),
                backend: window.backend.clone(),
            },
            other => other,
        })
        .collect()
}

fn tightest(descriptors: &[Option<&Descriptor>]) -> Option<(Descriptor, i64)> {
    let mut best: Option<(Descriptor, i64)> = None;
    for descriptor in descriptors.iter().flatten() {
        let Some(tokens) = descriptor.tokens_per_unit() else {
            continue;
        };
        if best.as_ref().is_none_or(|(_, least)| tokens < *least) {
            best = Some(((*descriptor).clone(), tokens));
        }
    }
    best
}

fn tightest_of(
    first: Option<(Descriptor, i64)>,
    second: Option<(Descriptor, i64)>,
) -> Option<(Descriptor, i64)> {
    match (first, second) {
        (Some(first), Some(second)) if second.1 < first.1 => Some(second),
        (Some(first), _) => Some(first),
        (None, second) => second,
    }
}

fn capped_idle_share(
    share: Option<&Descriptor>,
    cap: Option<&Descriptor>,
) -> Option<(Descriptor, i64)> {
    let tokens = share?.tokens_per_unit()?;
    Some((cap?.clone(), tokens))
}

fn team_settlement(
    state: &ShareState,
    served: Option<&Hold>,
    charge: Charge,
    ttl: i64,
) -> Vec<ReservationOp> {
    let own_usage = |reserved: i64| {
        if served.is_none() {
            charge.against(reserved)
        } else {
            0
        }
    };
    let mut operations = window_guarded(
        team_increment(
            state.limit.as_ref(),
            own_usage(state.reserved_tokens) - state.reserved_tokens,
            ttl,
        ),
        state.reserved_window.as_ref(),
    );
    operations.extend(window_guarded(
        team_increment(
            state.cap.as_ref(),
            own_usage(state.cap_reserved_tokens) - state.cap_reserved_tokens,
            ttl,
        ),
        state.cap_reserved_window.as_ref(),
    ));
    if let Some(hold) = served {
        operations.extend(hold_settlement(hold, charge, ttl));
    }
    operations
}

fn hold_settlement(hold: &Hold, charge: Charge, ttl: i64) -> Vec<ReservationOp> {
    let mut operations = window_guarded(
        team_increment(
            hold.descriptor.as_ref(),
            charge.against(hold.reserved_tokens) - hold.reserved_tokens,
            ttl,
        ),
        hold.window.as_ref(),
    );
    operations.extend(window_guarded(
        team_increment(
            hold.cap.as_ref(),
            charge.against(hold.cap_reserved_tokens) - hold.cap_reserved_tokens,
            ttl,
        ),
        hold.cap_window.as_ref(),
    ));
    operations
}

/// Every borrower's settlement, against the hold its served fallback target took when one served the call.
fn borrowing_settlement(
    stash: &Stash,
    served: &Holds,
    charge: Charge,
    ttl: i64,
) -> Vec<ReservationOp> {
    BORROWER_KINDS
        .into_iter()
        .flat_map(|kind| {
            team_settlement(
                stash.share(kind),
                served[kind.index()].as_deref(),
                charge,
                ttl,
            )
        })
        .collect()
}

fn served_holds(stash: &Stash, model_group: Option<&str>) -> Holds {
    BORROWER_KINDS.map(|kind| model_group.and_then(|group| stash.share(kind).hold(group)))
}

fn release_holds(stash: &mut Stash, served: &Holds) {
    for kind in BORROWER_KINDS {
        let state = stash.share_mut(kind);
        state.clear_reservation();
        state.drop_hold(served[kind.index()].as_ref());
    }
}

pub struct Settlement<'a> {
    pub model_group: Option<&'a LoggedGroup>,
    pub team_id: Option<&'a str>,
    pub team_alias: Option<&'a str>,
    pub key_hash: Option<&'a str>,
    pub key_alias: Option<&'a str>,
    pub customer_org_id: Option<&'a str>,
    pub customer_org_alias: Option<&'a str>,
}

impl Settlement<'_> {
    fn holder(&self, kind: BorrowerKind) -> Option<ShareHolder> {
        let (entity_id, alias) = match kind {
            BorrowerKind::Team => (self.team_id, self.team_alias),
            BorrowerKind::Key => (self.key_hash, self.key_alias),
            BorrowerKind::CustomerOrg => (self.customer_org_id, self.customer_org_alias),
        };
        Some(ShareHolder {
            kind,
            entity_id: entity_id?.to_owned(),
            alias: alias.map(str::to_owned),
        })
    }
}

impl Engine {
    pub async fn saturation(&self, settings: &Settings, model: &str, group: &GroupInfo) -> f64 {
        match self.read_saturation(settings, model, group).await {
            Ok(saturation) => saturation,
            Err(error) => {
                let message = format!(
                    "Error checking saturation for {model}: {}",
                    error.python_message()
                );
                let _ = self.host.log("error", &message);
                0.0
            }
        }
    }

    async fn read_saturation(
        &self,
        settings: &Settings,
        model: &str,
        group: &GroupInfo,
    ) -> Result<f64, CapacityError> {
        let mut saturation: f64 = 0.0;
        let ttl = &settings.saturation_check_cache_ttl;
        if let Some(rpm) = group.rpm.filter(|rpm| *rpm > 0) {
            let counter = scope(MODEL_SATURATION_CHECK, model).with_kind(CounterKind::Requests);
            if let Some(value) = self.counter_value(&counter, ttl).await? {
                saturation = saturation.max(number::int_of(&value)? as f64 / rpm as f64);
            }
        }
        if let Some(tpm) = group.tpm.filter(|tpm| *tpm > 0) {
            let counter = scope(MODEL_SATURATION_CHECK, model).with_kind(CounterKind::Tokens);
            if let Some(value) = self.counter_value(&counter, ttl).await? {
                saturation = saturation.max(number::float_of(&value)? / tpm as f64);
            }
        }
        Ok(saturation)
    }

    async fn counter_value(
        &self,
        counter: &Counter,
        ttl: &Value,
    ) -> Result<Option<Value>, CapacityError> {
        let value = match self.limiter.read_counter(counter, ttl).await {
            Ok(value) => value,
            Err(error) => {
                let message = format!(
                    "Saturation read failed in Redis, using this worker's copy: {}",
                    error.python_message()
                );
                let _ = self.host.log("warning", &message);
                self.limiter.read_local_counter(counter)?
            }
        };
        Ok(value.filter(|value| !value.is_null() && !value.is_object()))
    }

    fn model_wide(settings: &Settings, model: &str, group: &GroupInfo) -> Descriptor {
        Descriptor {
            key: MODEL_SATURATION_CHECK.into(),
            value: model.into(),
            rate_limit: Some(RateLimit {
                requests_per_unit: group.rpm.filter(|rpm| *rpm != 0),
                tokens_per_unit: group.tpm.filter(|tpm| *tpm != 0),
                window_size: Some(settings.window_size),
            }),
        }
    }

    async fn class_share(
        &self,
        request: &Request<'_>,
        model: &str,
        weights: &[(Value, f64)],
    ) -> Result<Option<TeamShare>, CapacityError> {
        let settings = request.settings;
        if settings.priority_reservation.is_none() || settings.class_split.is_empty() {
            return Ok(None);
        }
        let class = priority_class(settings, request.priority())?;
        let holder = class_share_holder(settings, request.key);
        let member = match &holder {
            None => pool_member(&class),
            Some(holder) if is_team_class_share_holder(holder) => team_member(holder),
            Some(holder) => holder.clone(),
        };
        let touched = match self
            .host
            .touch(model, &member, &class, settings.window_size)
        {
            Ok(Touched::Members(members)) => Ok(members),
            Ok(Touched::Pending(pending)) => self.host.touched(pending).await,
            Err(error) => Err(error),
        };
        let members = match touched {
            Ok(members) => members,
            Err(error) if error.is_registry_unavailable() => {
                let message = format!(
                    "Active team registry unavailable for {model}, holding team {} to the class pool: {}",
                    python_optional(request.key.team_text().as_deref()),
                    error.python_message()
                );
                let _ = self.host.log("warning", &message);
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if holder.is_none() {
            return Ok(None);
        }
        Ok(Some(team_share(
            &class,
            &members,
            weights,
            settings.default_priority,
        )))
    }

    async fn priority_descriptors(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
        share: Option<&TeamShare>,
    ) -> Result<Vec<Descriptor>, CapacityError> {
        let settings = request.settings;
        if settings.priority_reservation.is_none() {
            return Ok(Vec::new());
        }
        let weights = normalized_weights(settings, group)?;
        let allocated = allocation(settings, model, request.priority(), &weights, group)?;
        if allocated.unlicensed {
            self.host.unlicensed_priority()?;
        }
        let weight = share.map_or(allocated.weight, |share| share.fraction);
        let floor = if share.is_some() { 1 } else { 0 };
        let tokens = group.tpm.map(|tpm| floor.max((tpm as f64 * weight) as i64));
        let requests = group.rpm.map(|rpm| floor.max((rpm as f64 * weight) as i64));
        if tokens.is_none() && requests.is_none() {
            return Ok(Vec::new());
        }
        let rate_limit = RateLimit {
            requests_per_unit: requests,
            tokens_per_unit: tokens,
            window_size: Some(settings.window_size),
        };
        Ok(vec![match share {
            None => Descriptor {
                key: PRIORITY_MODEL.into(),
                value: allocated.key,
                rate_limit: Some(rate_limit),
            },
            Some(_) => Descriptor {
                key: PRIORITY_CLASS_SHARE_AT_SATURATION.into(),
                value: counter_value(
                    model,
                    class_share_holder(settings, request.key)
                        .as_deref()
                        .unwrap_or("None"),
                ),
                rate_limit: Some(rate_limit),
            },
        }])
    }

    /// One counter per borrower and resolved model group, so an alias and its target share the borrower's share.
    pub fn borrowing_shares(
        &self,
        request: &Request<'_>,
    ) -> Result<Vec<BorrowingShare>, CapacityError> {
        let window_size = request.settings.window_size;
        Ok(request
            .facts
            .borrowing()?
            .into_iter()
            .map(|borrower| {
                let value = format!("{}:{}", borrower.entity_id, borrower.group);
                let descriptor = |key: String, tpm: Option<i64>, rpm: Option<i64>| Descriptor {
                    key,
                    value: value.clone(),
                    rate_limit: Some(RateLimit {
                        requests_per_unit: rpm,
                        tokens_per_unit: tpm,
                        window_size: Some(window_size),
                    }),
                };
                let share = descriptor(borrower.kind.at_saturation(), borrower.tpm, borrower.rpm);
                let ceiling = borrower
                    .ceiling()
                    .map(|(tpm, rpm)| descriptor(borrower.kind.max_borrow_multiple(), tpm, rpm));
                BorrowingShare {
                    borrower,
                    share,
                    ceiling,
                }
            })
            .collect())
    }

    pub async fn token_reservation(
        &self,
        descriptor: &Descriptor,
        model: &str,
        call_type: Option<&str>,
    ) -> Result<i64, CapacityError> {
        let Some(tpm_share) = descriptor.tokens_per_unit() else {
            return Ok(0);
        };
        Ok(
            match self
                .host
                .estimate_tokens(tpm_share, Some(model), call_type)
                .await?
            {
                Estimate::NoBody | Estimate::Unpriced => 0,
                Estimate::Failed => 1,
                Estimate::Tokens(tokens) => tokens.max(1),
            },
        )
    }

    pub async fn settled_estimate(
        &self,
        model: Option<&str>,
        call_type: Option<&str>,
    ) -> Result<Option<i64>, CapacityError> {
        Ok(
            match self.host.estimate_tokens(0, model, call_type).await? {
                Estimate::Tokens(tokens) => Some(tokens.max(0)),
                Estimate::NoBody | Estimate::Failed => Some(0),
                Estimate::Unpriced => None,
            },
        )
    }

    /// One estimate for every borrower that holds tokens, sized with the output cap of the tightest TPM share,
    /// since that is the cap the admitted request is sent with.
    async fn tightest_token_reservation(
        &self,
        holding: &[&BorrowingShare],
        model: &str,
        call_type: Option<&str>,
    ) -> Result<i64, CapacityError> {
        let tightest = first_least(holding.iter().filter_map(|share| {
            let tpm = share.share.tokens_per_unit().filter(|tpm| *tpm != 0)?;
            Some((tpm, &share.share))
        }));
        match tightest {
            None => Ok(0),
            Some((_, descriptor)) => self.token_reservation(descriptor, model, call_type).await,
        }
    }

    pub async fn admit(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
    ) -> Result<Published, AdmitError> {
        let saturation = self.saturation(request.settings, model, group).await;
        let threshold = request.settings.saturation_threshold;
        if saturation >= threshold && self.fallback_requested(&request.fallback).await {
            let priority = request.priority_value();
            let context = Context {
                model,
                priority: priority.as_ref(),
                saturation,
                window_size: request.settings.window_size,
            };
            return Err(AdmitError::Refused(context.saturation_fallback(threshold)));
        }
        self.check_rate_limits(request, model, group, saturation)
            .await
    }

    async fn fallback_requested(&self, fallback: &SaturationFallback) -> bool {
        if let Some(invalid) = &fallback.invalid {
            let message = format!(
                "router_settings.fallback_on_saturation={invalid} is not a boolean, treating it as disabled"
            );
            let _ = self.host.log("warning", &message);
        }
        fallback.enabled && fallback.armed
    }

    async fn check_rate_limits(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
        saturation: f64,
    ) -> Result<Published, AdmitError> {
        let settings = request.settings;
        let enforce_priority = saturation >= settings.saturation_threshold;
        let model_wide = Self::model_wide(settings, model, group);
        let weights = normalized_weights(settings, group)?;
        let share = self.class_share(request, model, &weights).await?;
        let priority_descriptors = self
            .priority_descriptors(request, model, group, share.as_ref())
            .await?;
        {
            let mut stash = request.stash.lock();
            stash.priority_class_share = priority_descriptors
                .iter()
                .find(|descriptor| descriptor.key == PRIORITY_CLASS_SHARE_AT_SATURATION)
                .cloned();
            stash.priority_class_share_holder = class_share_holder(settings, request.key);
            stash.priority = request.priority_value();
        }
        let borrowing = self.borrowing_shares(request)?;
        Self::track_shares(
            &mut request.stash.lock(),
            &borrowing,
            enforce_priority,
            true,
        );
        let reservation = self
            .tightest_token_reservation(
                &holding(&borrowing, enforce_priority),
                model,
                request.call_type,
            )
            .await?;
        let held = held_shares(&borrowing, enforce_priority, reservation);

        let per_request = Increment {
            requests: 1,
            tokens: 0,
        };
        let mut enforced = vec![model_wide];
        let mut increments = vec![per_request];
        if enforce_priority {
            for descriptor in &priority_descriptors {
                enforced.push(descriptor.clone());
                increments.push(per_request);
            }
        }
        for share in &held {
            for descriptor in share.descriptors() {
                enforced.push(descriptor.clone());
                increments.push(Increment {
                    requests: 1,
                    tokens: share.reservation,
                });
            }
        }
        let atomic = self
            .limiter
            .check_and_increment(&enforced, &increments, request.ctx())
            .await?;
        if atomic.over_limit() {
            let priority = request.priority_value();
            let context = Context {
                model,
                priority: priority.as_ref(),
                saturation,
                window_size: settings.window_size,
            };
            let (label, holder_id) = class_share_refusal_holder(settings, request.key);
            return Err(AdmitError::Refused(Self::refusal(
                &context,
                &atomic,
                group,
                &borrowing,
                share.as_ref(),
                (label, &holder_id),
            )));
        }
        let reservation_of = |kind: BorrowerKind| {
            held.iter()
                .find(|share| share.kind == kind)
                .map_or(0, |share| share.reservation)
        };
        {
            let mut stash = request.stash.lock();
            for kind in BORROWER_KINDS {
                let reservation = reservation_of(kind);
                let state = stash.share_mut(kind);
                let reserved_window = |descriptor: &Option<Descriptor>| {
                    descriptor
                        .as_ref()
                        .filter(|_| reservation != 0)
                        .and_then(|descriptor| {
                            atomic.window_of(&descriptor.counter(CounterKind::Tokens))
                        })
                };
                state.reserved_tokens = if state.limit.is_some() {
                    reservation
                } else {
                    0
                };
                state.cap_reserved_tokens = if state.cap.is_some() { reservation } else { 0 };
                state.reserved_window = reserved_window(&state.limit);
                state.cap_reserved_window = reserved_window(&state.cap);
            }
        }
        let tightest_held = first_least(borrowing.iter().filter_map(|share| {
            let reservation = reservation_of(share.kind());
            let tpm = share.share.tokens_per_unit().filter(|_| reservation != 0)?;
            Some((tpm, reservation))
        }));
        if let Some((tpm_share, _)) = tightest_held
            && settings.rate_limit_type != TokenRateLimitType::Input
        {
            request.stash.lock().output_cap =
                Some((tpm_share, request.call_type.map(str::to_owned)));
        }
        let mut published = Published {
            overall_code: atomic.overall_code,
            statuses: atomic.statuses,
        };
        if !priority_descriptors.is_empty() && !enforce_priority {
            published.statuses.extend(
                self.track_priority_usage(priority_descriptors, request)
                    .await?,
            );
        }
        Ok(published)
    }

    /// Record what each borrower enforces on the model. `admitting` replaces the enforced share outright, a socket
    /// turn keeps the one its admission still holds tokens on.
    fn track_shares(
        stash: &mut Stash,
        borrowing: &[BorrowingShare],
        saturated: bool,
        admitting: bool,
    ) {
        for kind in BORROWER_KINDS {
            let share = borrower_of(borrowing, kind);
            let state = stash.share_mut(kind);
            let enforced = share.and_then(|share| share.enforced(saturated)).cloned();
            if admitting || enforced.is_some() || state.reserved_tokens == 0 {
                state.limit = enforced;
            }
            state.share = share.map(|share| share.share.clone());
            state.cap = share.and_then(|share| share.ceiling.clone());
        }
    }

    /// Count one request against each tracked priority counter. The atomic call writes nothing once any counter is
    /// full, so a full counter is reported as it stands and the rest are counted without it.
    async fn track_priority_usage(
        &self,
        mut descriptors: Vec<Descriptor>,
        request: &Request<'_>,
    ) -> Result<Vec<Status>, CapacityError> {
        let mut statuses = Vec::new();
        loop {
            let increments = vec![
                Increment {
                    requests: 1,
                    tokens: 0,
                };
                descriptors.len()
            ];
            let tracked = self
                .limiter
                .check_and_increment(&descriptors, &increments, request.ctx())
                .await?;
            let Some(full) = tracked
                .statuses
                .iter()
                .find(|status| status.over_limit())
                .cloned()
            else {
                statuses.extend(tracked.statuses);
                return Ok(statuses);
            };
            descriptors = descriptors
                .iter()
                .map(|descriptor| without_counter(descriptor, &full))
                .collect();
            statuses.push(full);
        }
    }

    fn refusal(
        context: &Context<'_>,
        atomic: &LimiterResponse,
        group: &GroupInfo,
        borrowing: &[BorrowingShare],
        class_share: Option<&TeamShare>,
        class_holder: (&str, &str),
    ) -> Refusal {
        for status in atomic.statuses.iter().filter(|status| status.over_limit()) {
            match status.descriptor_key.as_str() {
                MODEL_SATURATION_CHECK => return context.model_capacity(status, group),
                PRIORITY_MODEL => return context.priority_pool(status, group),
                key => {
                    if let Some(share) =
                        BorrowerKind::by_reason(key).and_then(|kind| borrower_of(borrowing, kind))
                    {
                        return context.team_share(status, &share.borrower);
                    }
                    if key == PRIORITY_CLASS_SHARE_AT_SATURATION
                        && let Some(share) = class_share
                    {
                        return context.class_share(status, share, class_holder);
                    }
                }
            }
        }
        context.unknown(atomic.statuses.iter().find(|status| status.over_limit()))
    }

    pub async fn pre_call(&self, request: &Request<'_>, model: &str, group: &GroupInfo) -> PreCall {
        match self.admit(request, model, group).await {
            Ok(published) => PreCall::Allow(Some(published)),
            Err(AdmitError::Refused(refusal)) => {
                self.throttled(&refusal, model, request).await;
                PreCall::Refuse(refusal)
            }
            Err(AdmitError::Failed(error)) if error.is_http() => PreCall::Raise(error),
            Err(AdmitError::Failed(error)) => {
                let message = format!(
                    "Error in dynamic rate limiter: {}, allowing request",
                    error.python_message()
                );
                let _ = self.host.log("error", &message);
                PreCall::Allow(None)
            }
        }
    }

    /// Count a refusal that reaches the caller under the reason its `rate_limited_by` header carries.
    async fn throttled(&self, refusal: &Refusal, model: &str, request: &Request<'_>) {
        let Some(reason) = refusal.header("rate_limited_by").map(loaded_str) else {
            return;
        };
        let Some(holder) = throttled_holder(&reason, request.settings, request.key) else {
            return;
        };
        let or_default = |value: Option<&Value>, default: &str| match value {
            Some(value) if json_truthy(value) => value.clone(),
            _ => Value::from(default),
        };
        let event = Throttle {
            model_group: model.to_owned(),
            holder_kind: holder.kind,
            holder_id: holder.entity_id,
            holder_alias: holder.alias,
            priority: or_default(refusal.header("x-litellm-priority"), "default"),
            rate_limit_type: loaded_str(&or_default(refusal.header("rate_limit_type"), "tokens")),
            reason,
        };
        let _ = self.host.throttled(&event).await;
    }

    pub async fn enforce_frame(
        &self,
        request: &Request<'_>,
        model: &str,
        connection: Option<&SharedStash>,
    ) -> Result<Frame, CapacityError> {
        let Some(group) = request.facts.group()? else {
            return Ok(Frame::Allow);
        };
        let settings = request.settings;
        let weights = normalized_weights(settings, &group)?;
        let share = self.class_share(request, model, &weights).await?;
        let share_descriptors = match &share {
            None => Vec::new(),
            Some(share) => {
                self.priority_descriptors(request, model, &group, Some(share))
                    .await?
            }
        };
        let frame_share = share_descriptors
            .iter()
            .find(|descriptor| descriptor.key == PRIORITY_CLASS_SHARE_AT_SATURATION)
            .cloned();
        if let (Some(stash), Some(frame_share)) = (connection, frame_share) {
            let mut stash = stash.lock();
            stash.priority_class_share = Some(frame_share);
            stash.priority_class_share_holder = class_share_holder(settings, request.key);
        }
        let borrowing = self.borrowing_shares(request)?;
        if share_descriptors.is_empty() && borrowing.is_empty() {
            if let Some(stash) = connection {
                Self::track_shares(&mut stash.lock(), &borrowing, false, false);
            }
            return Ok(Frame::Allow);
        }
        let saturation = self.saturation(settings, model, &group).await;
        let saturated = saturation >= settings.saturation_threshold;
        if let Some(stash) = connection {
            Self::track_shares(&mut stash.lock(), &borrowing, saturated, false);
        }
        let mut descriptors = if saturated {
            share_descriptors
        } else {
            Vec::new()
        };
        for share in &borrowing {
            descriptors.extend(share.enforced(saturated).cloned());
            descriptors.extend(share.ceiling.clone());
        }
        if descriptors.is_empty() {
            return Ok(Frame::Allow);
        }
        let windows = socket_turn_tpm_windows(&descriptors);
        let response = match self
            .limiter
            .should_rate_limit(&windows, true, true, request.ctx())
            .await
        {
            Ok(response) => response,
            Err(error) => return Ok(Frame::Unverifiable(error)),
        };
        let Some(over_limit) = response.statuses.iter().find(|status| status.over_limit()) else {
            return Ok(Frame::Allow);
        };
        let priority = request.priority_value();
        let context = Context {
            model,
            priority: priority.as_ref(),
            saturation,
            window_size: settings.window_size,
        };
        let over_class_share = over_limit.descriptor_key == PRIORITY_CLASS_SHARE_AT_SATURATION;
        let over_borrower = BorrowerKind::by_reason(&over_limit.descriptor_key)
            .and_then(|kind| borrower_of(&borrowing, kind));
        let refusal = match (&share, over_borrower) {
            (Some(share), _) if over_class_share || borrowing.is_empty() => {
                let (label, holder_id) = class_share_refusal_holder(settings, request.key);
                context.class_share(over_limit, share, (label, &holder_id))
            }
            (_, Some(borrower)) => context.team_share(over_limit, &borrower.borrower),
            _ => return Ok(Frame::Refuse(context.unknown(Some(over_limit)))),
        };
        self.throttled(&refusal, model, request).await;
        Ok(Frame::Refuse(refusal))
    }

    /// Hold a router fallback target to each borrower's share on it once it saturates, and to its borrow ceiling
    /// before. With `commit` the attempt is counted toward the target's saturation, and toward a borrower's counter
    /// only while its share is enforced, holding the prompt's estimated tokens there.
    pub async fn fallback_target_allowed(
        &self,
        request: &Request<'_>,
        model: &str,
        commit: bool,
    ) -> Result<bool, CapacityError> {
        let borrowing = self.borrowing_shares(request)?;
        let group = request.facts.group()?;
        if commit {
            self.hold_for_borrowers_without_a_share(request.stash, model, &borrowing)
                .await?;
        }
        let Some(group) = group.filter(|_| !borrowing.is_empty()) else {
            return Ok(true);
        };
        let settings = request.settings;
        let saturation = self.saturation(settings, model, &group).await;
        let saturated = saturation >= settings.saturation_threshold;
        let call_type = request.facts.call_type()?;
        let reservation = self
            .tightest_token_reservation(
                &holding(&borrowing, saturated),
                model,
                call_type.as_deref(),
            )
            .await?;
        let targets = held_shares(&borrowing, saturated, reservation);
        if !commit {
            for target in &targets {
                for descriptor in target.descriptors() {
                    if !self
                        .share_has_room(descriptor, target.reservation, request.ctx())
                        .await?
                    {
                        return Ok(false);
                    }
                }
            }
            return Ok(true);
        }
        let mut descriptors = vec![Self::model_wide(settings, model, &group)];
        let mut increments = vec![Increment {
            requests: 1,
            tokens: 0,
        }];
        for target in &targets {
            for descriptor in target.descriptors() {
                descriptors.push(descriptor.clone());
                increments.push(Increment {
                    requests: 1,
                    tokens: target.reservation,
                });
            }
        }
        let response = self
            .limiter
            .check_and_increment(&descriptors, &increments, request.ctx())
            .await?;
        if response.over_limit() {
            return Ok(false);
        }
        let window_of = |descriptor: &Option<Descriptor>| {
            descriptor
                .as_ref()
                .and_then(|descriptor| response.window_of(&descriptor.counter(CounterKind::Tokens)))
        };
        for target in targets {
            let hold = Hold {
                reserved_tokens: if target.enforced.is_some() {
                    target.reservation
                } else {
                    0
                },
                window: window_of(&target.enforced),
                cap_reserved_tokens: if target.cap.is_some() {
                    target.reservation
                } else {
                    0
                },
                cap_window: window_of(&target.cap),
                descriptor: target.enforced,
                cap: target.cap,
                share: Some(target.share.clone()),
            };
            Self::hold_fallback_share(request.stash, target.kind, model, &target.share, hold);
        }
        Ok(true)
    }

    /// A borrower with no limit on the target takes an empty hold, so the target's usage never settles to its share
    /// on the admitting model.
    async fn hold_for_borrowers_without_a_share(
        &self,
        stash: &SharedStash,
        model: &str,
        borrowing: &[BorrowingShare],
    ) -> Result<(), CapacityError> {
        let missing: Vec<BorrowerKind> = {
            let stash = stash.lock();
            BORROWER_KINDS
                .into_iter()
                .filter(|kind| {
                    borrower_of(borrowing, *kind).is_none()
                        && stash.share(*kind).hold(model).is_none()
                })
                .collect()
        };
        if missing.is_empty() {
            return Ok(());
        }
        let group = self.host.model_group(model).await?;
        let mut stash = stash.lock();
        for kind in missing {
            let hold = Arc::new(Hold::empty());
            let state = stash.share_mut(kind);
            state.put_hold(model, &hold);
            state.put_hold(&group, &hold);
        }
        Ok(())
    }

    async fn share_has_room(
        &self,
        descriptor: &Descriptor,
        tokens: i64,
        ctx: Ctx<'_>,
    ) -> Result<bool, CapacityError> {
        let checked = self
            .limiter
            .should_rate_limit(std::slice::from_ref(descriptor), true, false, ctx)
            .await?;
        Ok(checked.statuses.iter().all(|status| {
            !status.over_limit()
                && (status.rate_limit_type != "tokens"
                    || status
                        .limit_remaining
                        .as_ref()
                        .and_then(serde_json::Number::as_f64)
                        .unwrap_or(0.0)
                        >= tokens as f64)
        }))
    }

    fn hold_fallback_share(
        stash: &SharedStash,
        kind: BorrowerKind,
        model: &str,
        share: &Descriptor,
        hold: Hold,
    ) {
        let mut stash = stash.lock();
        let state = stash.share_mut(kind);
        let earlier = state.hold(model);
        let carried = earlier
            .as_ref()
            .filter(|earlier| hold.window.is_some() && earlier.window == hold.window)
            .map_or(0, |earlier| earlier.reserved_tokens);
        let carried_cap = earlier
            .as_ref()
            .filter(|earlier| hold.cap_window.is_some() && earlier.cap_window == hold.cap_window)
            .map_or(0, |earlier| earlier.cap_reserved_tokens);
        let hold = Arc::new(Hold {
            reserved_tokens: hold.reserved_tokens + carried,
            cap_reserved_tokens: hold.cap_reserved_tokens + carried_cap,
            ..hold
        });
        let group = share
            .value
            .rsplit_once(':')
            .map_or(share.value.as_str(), |(_, tail)| tail);
        state.put_hold(model, &hold);
        state.put_hold(group, &hold);
    }

    pub async fn settle_success(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        settlement: &Settlement<'_>,
        priority: Option<&str>,
        charge: Charge,
        redis: Option<&RequestRedis>,
    ) -> Result<(), CapacityError> {
        let ttl = settings.window_size;
        let total_tokens = charge.tokens();
        let group = settlement.model_group;
        let (team_tokens, served) = match stash {
            None => (Vec::new(), Holds::default()),
            Some(stash) => {
                let mut stash = stash.lock();
                let served = match group {
                    Some(group) => served_holds(&stash, group.key()?),
                    None => Holds::default(),
                };
                let team_tokens = borrowing_settlement(&stash, &served, charge, ttl);
                release_holds(&mut stash, &served);
                (team_tokens, served)
            }
        };
        let Some(model_group) = group.filter(|group| group.truthy) else {
            if !team_tokens.is_empty() {
                self.limiter.apply(&team_tokens, redis).await?;
            }
            return Ok(());
        };
        if total_tokens == 0 && team_tokens.is_empty() {
            return Ok(());
        }
        if total_tokens != 0 && !model_group.keyed {
            return Err(CapacityError::Type(UNCOUNTABLE_GROUP.into()));
        }
        let model_group = model_group.text.as_str();
        let priority_key = match settings.reserves_name(priority) {
            Some(explicit) => format!("{model_group}:{explicit}"),
            None => format!("{model_group}:default_pool"),
        };
        let (class_share, class_holder) = stash.map_or((None, None), |stash| {
            let stash = stash.lock();
            (
                stash.priority_class_share.clone(),
                stash.priority_class_share_holder.clone(),
            )
        });
        let settled_holder = class_holder.or_else(|| settlement.team_id.map(str::to_owned));
        let priority_counter = match &class_share {
            None => scope(PRIORITY_MODEL, &priority_key),
            Some(share) => scope(
                &share.key,
                &settled_holder.map_or_else(
                    || share.value.clone(),
                    |holder| counter_value(model_group, &holder),
                ),
            ),
        }
        .with_kind(CounterKind::Tokens);
        let mut operations = Vec::new();
        if total_tokens != 0 {
            operations.push(ReservationOp::Increment {
                counter: scope(MODEL_SATURATION_CHECK, model_group).with_kind(CounterKind::Tokens),
                amount: total_tokens,
                ttl: Some(ttl),
            });
            operations.push(ReservationOp::Increment {
                counter: priority_counter,
                amount: total_tokens,
                ttl: Some(ttl),
            });
        }
        operations.extend(team_tokens);
        if !operations.is_empty() {
            self.limiter.apply(&operations, redis).await?;
            self.record_team_share(settings, stash, Some(model_group), settlement, &served)
                .await;
        }
        Ok(())
    }

    /// Settle a failed attempt: a borrower's admitting reservation is refunded or charged the usage the attempt
    /// recorded, and a fallback target that failed settles its own hold. Returns the holds that attempt released.
    pub async fn settle_failure(
        &self,
        settings: &Settings,
        stash: &SharedStash,
        failed_group: Option<&LoggedGroup>,
        used: i64,
        redis: Option<&RequestRedis>,
    ) -> Result<[Option<Arc<Hold>>; 3], CapacityError> {
        let (refund, failed) = {
            let mut stash = stash.lock();
            let failed = match failed_group {
                Some(group) => served_holds(&stash, group.key()?),
                None => Holds::default(),
            };
            let refund = borrowing_settlement(
                &stash,
                &failed,
                Charge::Reported(used),
                settings.window_size,
            );
            release_holds(&mut stash, &failed);
            (refund, failed)
        };
        if !refund.is_empty() {
            self.limiter.apply(&refund, redis).await?;
        }
        Ok(failed)
    }

    pub async fn release_admission(
        &self,
        settings: &Settings,
        stash: &SharedStash,
        redis: Option<&RequestRedis>,
    ) -> Result<(), CapacityError> {
        let refund = {
            let mut stash = stash.lock();
            let none = Holds::default();
            let refund =
                borrowing_settlement(&stash, &none, Charge::Reported(0), settings.window_size);
            for kind in BORROWER_KINDS {
                stash.share_mut(kind).clear_reservation();
            }
            refund
        };
        if refund.is_empty() {
            return Ok(());
        }
        self.limiter.apply(&refund, redis).await
    }

    pub async fn record_team_share(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        model_group: Option<&str>,
        settlement: &Settlement<'_>,
        holds: &[Option<Arc<Hold>>; 3],
    ) {
        if let Err(error) = self
            .try_record_team_share(settings, stash, model_group, settlement, holds)
            .await
        {
            let message = format!(
                "Dynamic rate limiter share metrics failed: {}",
                error.python_message()
            );
            let _ = self.host.log("warning", &message);
        }
    }

    /// Publish each borrower's share on the model that served the request, the fallback target's own hold when a
    /// router fallback served it, and the tokens settled above that share so far this window.
    async fn try_record_team_share(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        model_group: Option<&str>,
        settlement: &Settlement<'_>,
        holds: &[Option<Arc<Hold>>; 3],
    ) -> Result<(), CapacityError> {
        if !self.host.metrics_enabled().await? {
            return Ok(());
        }
        let (Some(stash), Some(model_group)) = (stash, model_group) else {
            return Ok(());
        };
        let group = self.host.model_group(model_group).await?;
        let (shares, priority) = {
            let stash = stash.lock();
            let class_kind = class_share_kind(stash.priority_class_share_holder.as_deref());
            let mut shares = Vec::new();
            for kind in BORROWER_KINDS {
                let state = stash.share(kind);
                let holder = settlement.holder(kind);
                let hold = holds[kind.index()].as_deref();
                let class_share = stash
                    .priority_class_share
                    .as_ref()
                    .filter(|_| class_kind == Some(kind));
                let share = match hold {
                    Some(hold) => tightest_of(
                        tightest(&[hold.descriptor.as_ref()]),
                        capped_idle_share(hold.share.as_ref(), hold.cap.as_ref()),
                    ),
                    None => tightest_of(
                        tightest(&[state.limit.as_ref(), class_share]),
                        capped_idle_share(state.share.as_ref(), state.cap.as_ref()),
                    ),
                }
                .or_else(|| match hold {
                    Some(hold) => tightest(&[hold.share.as_ref()]),
                    None => {
                        let holder_id = holder
                            .as_ref()
                            .map_or("None", |holder| holder.entity_id.as_str());
                        let on_model = format!("{holder_id}:{group}");
                        tightest(&[state.share.as_ref().filter(|share| share.value == on_model)])
                    }
                });
                if let (Some(holder), Some(share)) = (holder, share) {
                    shares.push((holder, share));
                }
            }
            (shares, stash.priority.clone())
        };
        let priority = if priority.as_ref().is_some_and(json_truthy) {
            priority.unwrap_or(Value::Null)
        } else {
            Value::from("default")
        };
        for (holder, (descriptor, share_tokens)) in shares {
            let used = self
                .counter_value(
                    &descriptor.counter(CounterKind::Tokens),
                    &settings.saturation_check_cache_ttl,
                )
                .await?;
            let used_tokens = match used {
                None => 0,
                Some(value) => number::float_of(&value)? as i64,
            };
            let gauge = crate::host::ShareGauge {
                model_group: model_group.to_owned(),
                holder_kind: holder.kind,
                holder_id: holder.entity_id,
                holder_alias: holder.alias,
                priority: priority.clone(),
                share_tokens,
                borrowed_tokens: (used_tokens - share_tokens).max(0),
            };
            self.host.set_share(&gauge).await?;
        }
        Ok(())
    }

    pub fn post_call_headers(key: &KeyFacts) -> Vec<(String, Value)> {
        let priority = key.priority_value();
        let header = if priority.as_ref().is_some_and(json_truthy) {
            priority.unwrap_or(Value::Null)
        } else {
            Value::from("default")
        };
        let mut headers = Vec::new();
        if is_latin1(&loaded_str(&header)) {
            headers.push(("x-litellm-priority".to_owned(), header));
        }
        headers.push((
            "x-litellm-rate-limiter-version".to_owned(),
            Value::from("v3"),
        ));
        headers
    }
}

pub struct Scope {
    key: String,
    value: String,
}

impl Scope {
    pub fn with_kind(self, kind: CounterKind) -> Counter {
        Counter {
            key: self.key,
            value: self.value,
            kind,
        }
    }
}

pub fn scope(key: &str, value: &str) -> Scope {
    Scope {
        key: key.to_owned(),
        value: value.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use parking_lot::Mutex;
    use serde_json::{Number, Value, json};

    use litellm_callbacks_parallel_request_limiter::{Ctx, PolicyFacts};

    use crate::{
        host::{Fact, SeamFuture, ShareGauge},
        model::{ClassShareEntity, Status, TokenRateLimitType},
        state::Stash,
    };

    use super::*;

    const TEAM_MODEL_LIMIT_AT_SATURATION: &str = "model_per_team_saturated";
    const TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE: &str = "model_per_team_max_borrow_multiple";

    struct FakeLimiter {
        redis: Result<Option<Value>, String>,
        local: Option<Value>,
        atomic: LimiterResponse,
        ttls: Mutex<Vec<(Counter, Value)>>,
        script: Mutex<Vec<LimiterResponse>>,
        checked: Mutex<Vec<(Vec<Descriptor>, Vec<Increment>)>>,
    }

    impl FakeLimiter {
        fn new(
            redis: Result<Option<Value>, String>,
            local: Option<Value>,
            atomic: LimiterResponse,
        ) -> Self {
            Self {
                redis,
                local,
                atomic,
                ttls: Mutex::new(Vec::new()),
                script: Mutex::new(Vec::new()),
                checked: Mutex::new(Vec::new()),
            }
        }
    }

    impl CounterStore for FakeLimiter {
        fn read_counter<'a>(
            &'a self,
            counter: &'a Counter,
            backfill_ttl: &'a Value,
        ) -> SeamFuture<'a, Option<Value>> {
            self.ttls
                .lock()
                .push((counter.clone(), backfill_ttl.clone()));
            let reply = self.redis.clone().map_err(CapacityError::Store);
            Box::pin(async move { reply })
        }

        fn read_local_counter(&self, _counter: &Counter) -> Result<Option<Value>, CapacityError> {
            Ok(self.local.clone())
        }

        fn check_and_increment<'a>(
            &'a self,
            descriptors: &'a [Descriptor],
            increments: &'a [Increment],
            _ctx: Ctx<'a>,
        ) -> SeamFuture<'a, LimiterResponse> {
            self.checked
                .lock()
                .push((descriptors.to_vec(), increments.to_vec()));
            let mut script = self.script.lock();
            let reply = if script.is_empty() {
                self.atomic.clone()
            } else {
                script.remove(0)
            };
            Box::pin(async move { Ok(reply) })
        }

        fn should_rate_limit<'a>(
            &'a self,
            _descriptors: &'a [Descriptor],
            _read_only: bool,
            _skip_parallel_check: bool,
            _ctx: Ctx<'a>,
        ) -> SeamFuture<'a, LimiterResponse> {
            Box::pin(async move { Ok(answer("OK", Vec::new())) })
        }

        fn apply<'a>(
            &'a self,
            _operations: &'a [ReservationOp],
            _redis: Option<&'a RequestRedis>,
        ) -> SeamFuture<'a, ()> {
            Box::pin(async move { Ok(()) })
        }
    }

    #[derive(Default)]
    struct FakeHost {
        logged: Mutex<Vec<(String, String)>>,
    }

    fn done<'a, T: Send + 'a>(value: T) -> SeamFuture<'a, T> {
        Box::pin(async move { Ok(value) })
    }

    impl CapacityHost for FakeHost {
        fn touch(
            &self,
            _model: &str,
            _member: &str,
            _priority_class: &Value,
            _window: i64,
        ) -> Result<Touched, CapacityError> {
            Ok(Touched::Members(Vec::new()))
        }

        fn touched(&self, _pending: u64) -> SeamFuture<'_, Vec<(String, Value)>> {
            done(Vec::new())
        }

        fn metrics_enabled(&self) -> SeamFuture<'_, bool> {
            done(false)
        }

        fn throttled<'a>(&'a self, _event: &'a Throttle) -> SeamFuture<'a, ()> {
            done(())
        }

        fn set_share<'a>(&'a self, _gauge: &'a ShareGauge) -> SeamFuture<'a, ()> {
            done(())
        }

        fn unlicensed_priority(&self) -> Result<(), CapacityError> {
            Ok(())
        }

        fn log(&self, level: &'static str, message: &str) -> Result<(), CapacityError> {
            self.logged
                .lock()
                .push((level.to_owned(), message.to_owned()));
            Ok(())
        }

        fn model_group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, String> {
            done(model.to_owned())
        }

        fn estimate_tokens<'a>(
            &'a self,
            _tpm_share: i64,
            _model: Option<&'a str>,
            _call_type: Option<&'a str>,
        ) -> SeamFuture<'a, Estimate> {
            done(Estimate::NoBody)
        }

        fn post_call_success<'a>(&'a self, _headers: &'a [(String, Value)]) -> SeamFuture<'a, ()> {
            done(())
        }
    }

    fn settings(ttl: Value) -> Settings {
        Settings {
            priority_reservation: Some(vec![
                (json!("high"), json!(0.9)),
                (json!("low"), json!(0.1)),
            ]),
            default_priority: 0.5,
            saturation_threshold: 0.5,
            class_split: Vec::new(),
            saturation_check_cache_ttl: ttl,
            window_size: 60,
            rate_limit_type: TokenRateLimitType::default(),
            licensed: true,
            policy: PolicyFacts::default(),
        }
    }

    fn group(tpm: Option<i64>, rpm: Option<i64>) -> GroupInfo {
        GroupInfo {
            tpm,
            rpm,
            input_cost_per_token: None,
            output_cost_per_token: None,
        }
    }

    fn answer(code: &str, statuses: Vec<Status>) -> LimiterResponse {
        LimiterResponse {
            overall_code: code.into(),
            statuses,
            reservation_windows: Vec::new(),
        }
    }

    fn engine(limiter: &Arc<FakeLimiter>, host: &Arc<FakeHost>) -> Engine {
        Engine {
            limiter: Arc::clone(limiter) as Arc<dyn CounterStore>,
            host: Arc::clone(host) as Arc<dyn CapacityHost>,
        }
    }

    async fn pre_call(
        engine: &Engine,
        settings: &Settings,
        model: &str,
        group: &GroupInfo,
    ) -> PreCall {
        let key = KeyFacts {
            priority: Some(Priority {
                value: json!("high"),
                hash_error: None,
            }),
            ..KeyFacts::default()
        };
        let stash: SharedStash = Arc::new(Mutex::new(Stash::default()));
        let facts = Facts {
            borrowing: Some(Fact::Ok(Vec::new())),
            ..Facts::default()
        };
        let request = Request {
            key: &key,
            call_type: None,
            settings,
            stash: &stash,
            fallback: SaturationFallback {
                enabled: false,
                invalid: None,
                armed: false,
            },
            redis: None,
            facts: &facts,
        };
        engine.pre_call(&request, model, group).await
    }

    fn tokens(key: &str, tpm: Option<i64>) -> Descriptor {
        Descriptor {
            key: key.into(),
            value: "team:model".into(),
            rate_limit: Some(RateLimit {
                requests_per_unit: None,
                tokens_per_unit: tpm,
                window_size: Some(60),
            }),
        }
    }

    #[rstest]
    #[case::tighter_cap_wins(Some(100), Some(40), Some((TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE, 40)))]
    #[case::a_tie_keeps_the_enforced_share(Some(40), Some(40), Some((TEAM_MODEL_LIMIT_AT_SATURATION, 40)))]
    #[case::idle_cap_alone(None, Some(40), Some((TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE, 40)))]
    #[case::enforced_alone(Some(40), None, Some((TEAM_MODEL_LIMIT_AT_SATURATION, 40)))]
    #[case::neither(None, None, None)]
    fn the_tightest_share_reads_an_idle_cap_against_the_share_like_python(
        #[case] enforced: Option<i64>,
        #[case] share: Option<i64>,
        #[case] expected: Option<(&str, i64)>,
    ) {
        let enforced = enforced.map(|tpm| tokens(TEAM_MODEL_LIMIT_AT_SATURATION, Some(tpm)));
        let share = share.map(|tpm| tokens(TEAM_MODEL_LIMIT_AT_SATURATION, Some(tpm)));
        let cap = tokens(TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE, Some(1000));
        let tightest = tightest_of(
            tightest(&[enforced.as_ref()]),
            capped_idle_share(share.as_ref(), Some(&cap)),
        );
        assert_eq!(
            tightest
                .as_ref()
                .map(|(descriptor, tokens)| (descriptor.key.as_str(), *tokens)),
            expected
        );
    }

    #[tokio::test]
    async fn saturation_check_cache_ttl_configuration_backfills_each_read_for_its_ttl_like_python()
    {
        let limiter = Arc::new(FakeLimiter::new(Ok(None), None, answer("OK", Vec::new())));
        let host = Arc::new(FakeHost::default());
        let engine = engine(&limiter, &host);
        let model_group = group(Some(1000), Some(100));
        engine
            .saturation(&settings(json!(5)), "test-saturation-ttl", &model_group)
            .await;
        engine
            .saturation(&settings(json!(30)), "test-saturation-ttl", &model_group)
            .await;
        let requests =
            scope(MODEL_SATURATION_CHECK, "test-saturation-ttl").with_kind(CounterKind::Requests);
        let tokens =
            scope(MODEL_SATURATION_CHECK, "test-saturation-ttl").with_kind(CounterKind::Tokens);
        assert_eq!(
            *limiter.ttls.lock(),
            vec![
                (requests.clone(), json!(5)),
                (tokens.clone(), json!(5)),
                (requests, json!(30)),
                (tokens, json!(30)),
            ]
        );
    }

    #[tokio::test]
    async fn a_failed_saturation_read_falls_back_to_this_workers_copy_like_python() {
        let limiter = Arc::new(FakeLimiter::new(
            Err("redis down".into()),
            Some(json!(3)),
            answer("OK", Vec::new()),
        ));
        let host = Arc::new(FakeHost::default());
        let saturation = engine(&limiter, &host)
            .saturation(&settings(json!(60)), "down", &group(None, Some(4)))
            .await;
        assert!((saturation - 0.75).abs() < f64::EPSILON);
        assert_eq!(
            *host.logged.lock(),
            vec![(
                "warning".to_owned(),
                "Saturation read failed in Redis, using this worker's copy: redis down".to_owned()
            )]
        );
    }

    #[rstest]
    #[case::each_raise_branch_unknown_descriptor_for_fail_closed_fallback(
        "gpt-4",
        group(Some(1000), Some(100)),
        "unknown_descriptor_for_fail_closed_fallback",
        "requests",
        100
    )]
    #[case::fails_closed_on_unknown_descriptor(
        "fail-closed-model",
        group(Some(1000), None),
        "future_unrecognized_descriptor",
        "tokens",
        100
    )]
    #[case::unknown_descriptor_path_populates_provider(
        "gpt-4o-mini",
        group(None, Some(1)),
        "something_we_dont_handle",
        "requests",
        1
    )]
    #[tokio::test]
    async fn dynamic_rate_limiter_v3_refuses_an_over_limit_answer_for_an_unknown_descriptor(
        #[case] model: &str,
        #[case] model_group: GroupInfo,
        #[case] descriptor_key: &str,
        #[case] rate_limit_type: &str,
        #[case] current_limit: i64,
    ) {
        let unknown = Status {
            code: "OVER_LIMIT".into(),
            current_limit: Number::from(current_limit),
            limit_remaining: Some(Number::from(0)),
            rate_limit_type: rate_limit_type.into(),
            descriptor_key: descriptor_key.into(),
            descriptor_value: None,
        };
        let limiter = Arc::new(FakeLimiter::new(
            Ok(None),
            None,
            answer("OVER_LIMIT", vec![unknown]),
        ));
        let host = Arc::new(FakeHost::default());
        let PreCall::Refuse(refusal) = pre_call(
            &engine(&limiter, &host),
            &settings(json!(60)),
            model,
            &model_group,
        )
        .await
        else {
            panic!("an unknown over-limit descriptor must be refused");
        };
        assert_eq!(refusal.model, model);
        assert_eq!(refusal.rate_limit_type.as_deref(), Some(rate_limit_type));
        assert_eq!(
            refusal.detail,
            vec![
                ("error".to_owned(), json!("Rate limit exceeded")),
                ("descriptor_key".to_owned(), json!(descriptor_key)),
                ("rate_limit_type".to_owned(), json!(rate_limit_type)),
            ]
        );
    }

    fn status(code: &str, key: &str, value: &str, rate_limit_type: &str) -> Status {
        Status {
            code: code.into(),
            current_limit: Number::from(10),
            limit_remaining: Some(Number::from(0)),
            rate_limit_type: rate_limit_type.into(),
            descriptor_key: key.into(),
            descriptor_value: Some(value.into()),
        }
    }

    fn pool(value: &str) -> Descriptor {
        Descriptor {
            key: PRIORITY_MODEL.into(),
            value: value.into(),
            rate_limit: Some(RateLimit {
                requests_per_unit: Some(10),
                tokens_per_unit: Some(100),
                window_size: Some(60),
            }),
        }
    }

    #[tokio::test]
    async fn a_full_priority_counter_is_reported_and_the_rest_are_still_counted_without_it() {
        let limiter = Arc::new(FakeLimiter::new(Ok(None), None, answer("OK", Vec::new())));
        let full = status("OVER_LIMIT", PRIORITY_MODEL, "m:high", "tokens");
        let counted = status("OK", PRIORITY_MODEL, "m:high", "requests");
        limiter.script.lock().extend([
            answer("OVER_LIMIT", vec![counted.clone(), full.clone()]),
            answer("OK", vec![counted.clone()]),
        ]);
        let host = Arc::new(FakeHost::default());
        let engine = engine(&limiter, &host);
        let settings = settings(json!(60));
        let key = KeyFacts::default();
        let stash: SharedStash = Arc::new(Mutex::new(Stash::default()));
        let facts = Facts::default();
        let request = Request {
            key: &key,
            call_type: None,
            settings: &settings,
            stash: &stash,
            fallback: SaturationFallback::default(),
            redis: None,
            facts: &facts,
        };
        let statuses = engine
            .track_priority_usage(vec![pool("m:high")], &request)
            .await
            .unwrap_or_default();
        assert_eq!(statuses, vec![full, counted]);
        let checked = limiter.checked.lock();
        assert_eq!(checked.len(), 2);
        assert_eq!(
            checked[1].0[0]
                .rate_limit
                .as_ref()
                .map(|limit| (limit.requests_per_unit, limit.tokens_per_unit)),
            Some((Some(10), None))
        );
        assert!(checked.iter().all(|(_, increments)| increments
            == &vec![Increment {
                requests: 1,
                tokens: 0
            }]));
    }

    #[rstest]
    #[case::other_counter_is_kept(pool("m:low"), Some((Some(10), Some(100))))]
    #[case::the_full_tokens_counter_is_dropped(pool("m:high"), Some((Some(10), None)))]
    fn without_counter_drops_only_the_full_counter(
        #[case] descriptor: Descriptor,
        #[case] expected: Option<(Option<i64>, Option<i64>)>,
    ) {
        let full = status("OVER_LIMIT", PRIORITY_MODEL, "m:high", "tokens");
        assert_eq!(
            without_counter(&descriptor, &full)
                .rate_limit
                .map(|limit| (limit.requests_per_unit, limit.tokens_per_unit)),
            expected
        );
    }

    fn borrower(kind: BorrowerKind, tpm: Option<i64>, multiple: Option<f64>) -> BorrowingShare {
        let limits = BorrowingLimits {
            kind,
            entity_id: kind.name().to_owned(),
            alias: None,
            group: "m".into(),
            tpm,
            rpm: Some(5),
            max_borrow_multiple: multiple,
        };
        let descriptor = |key: String, tpm| Descriptor {
            key,
            value: format!("{}:m", kind.name()),
            rate_limit: Some(RateLimit {
                requests_per_unit: Some(5),
                tokens_per_unit: tpm,
                window_size: Some(60),
            }),
        };
        BorrowingShare {
            share: descriptor(kind.at_saturation(), tpm),
            ceiling: limits
                .ceiling()
                .map(|(tpm, _)| descriptor(kind.max_borrow_multiple(), tpm)),
            borrower: limits,
        }
    }

    #[rstest]
    fn below_saturation_only_a_capped_borrower_holds_tokens_and_a_share_without_tpm_holds_none() {
        let borrowing = vec![
            borrower(BorrowerKind::Team, Some(100), None),
            borrower(BorrowerKind::Key, Some(50), Some(2.0)),
            borrower(BorrowerKind::CustomerOrg, None, Some(2.0)),
        ];
        let idle: Vec<(BorrowerKind, i64, usize)> = held_shares(&borrowing, false, 7)
            .iter()
            .map(|held| (held.kind, held.reservation, held.descriptors().count()))
            .collect();
        assert_eq!(
            idle,
            vec![
                (BorrowerKind::Team, 0, 0),
                (BorrowerKind::Key, 7, 1),
                (BorrowerKind::CustomerOrg, 0, 1),
            ]
        );
        let saturated: Vec<(i64, usize)> = held_shares(&borrowing, true, 7)
            .iter()
            .map(|held| (held.reservation, held.descriptors().count()))
            .collect();
        assert_eq!(saturated, vec![(7, 1), (7, 2), (0, 2)]);
    }

    #[rstest]
    fn the_tightest_share_is_the_first_of_the_least_like_python_min() {
        assert_eq!(
            first_least([(5, "team"), (3, "key"), (3, "customer_org")]),
            Some((3, "key"))
        );
        assert_eq!(first_least(Vec::<(i64, &str)>::new()), None);
    }

    fn customer(api_key: &str, org: Option<&str>, member: Option<&str>) -> KeyFacts {
        KeyFacts {
            api_key: Some(api_key.into()),
            key_alias: Some(format!("{api_key}-alias")),
            customer_org_id: org.map(str::to_owned),
            customer_org_alias: org.map(|org| format!("{org}-alias")),
            customer_member_id: member.map(str::to_owned),
            ..KeyFacts::default()
        }
    }

    #[rstest]
    #[case::key_share("model_per_key_saturated", vec![], Some((BorrowerKind::Key, "sk-a", Some("sk-a-alias"))))]
    #[case::org_ceiling("model_per_customer_org_max_borrow_multiple", vec![], Some((BorrowerKind::CustomerOrg, "org-a", Some("org-a-alias"))))]
    #[case::team_share_without_a_team("model_per_team_saturated", vec![], None)]
    #[case::class_share_of_a_key(PRIORITY_CLASS_SHARE_AT_SATURATION, vec![ClassShareEntity::Key], Some((BorrowerKind::Key, "sk-a", Some("sk-a-alias"))))]
    #[case::class_share_of_an_org(PRIORITY_CLASS_SHARE_AT_SATURATION, vec![ClassShareEntity::CustomerOrganization], Some((BorrowerKind::CustomerOrg, "org-a", Some("org-a-alias"))))]
    #[case::class_share_of_a_member_has_no_label(PRIORITY_CLASS_SHARE_AT_SATURATION, vec![ClassShareEntity::CustomerMember], None)]
    fn a_throttle_is_counted_under_the_holder_whose_share_refused(
        #[case] reason: &str,
        #[case] split: Vec<ClassShareEntity>,
        #[case] expected: Option<(BorrowerKind, &str, Option<&str>)>,
    ) {
        let settings = Settings {
            class_split: split,
            ..settings(json!(60))
        };
        let key = customer("sk-a", Some("org-a"), Some("member-a"));
        let holder = throttled_holder(reason, &settings, &key);
        assert_eq!(
            holder.as_ref().map(|holder| (
                holder.kind,
                holder.entity_id.as_str(),
                holder.alias.as_deref()
            )),
            expected
        );
    }

    #[rstest]
    fn a_customer_member_class_share_refusal_names_the_member() {
        let settings = Settings {
            class_split: vec![ClassShareEntity::CustomerMember],
            ..settings(json!(60))
        };
        let key = customer("sk-a", Some("org-a"), Some("member-a"));
        assert_eq!(
            class_share_refusal_holder(&settings, &key),
            ("Customer member", "member-a".to_owned())
        );
    }

    fn refund(descriptor: Option<&Descriptor>, amount: i64) -> ReservationOp {
        ReservationOp::Increment {
            counter: descriptor.unwrap().counter(CounterKind::Tokens),
            amount,
            ttl: Some(60),
        }
    }

    #[rstest]
    fn an_estimated_charge_keeps_each_borrowers_reservation_where_a_reported_zero_refunds_it(
        #[values(BorrowerKind::Team, BorrowerKind::Key, BorrowerKind::CustomerOrg)]
        kind: BorrowerKind,
    ) {
        let share = borrower(kind, Some(100), Some(2.0));
        let mut stash = Stash::default();
        *stash.share_mut(kind) = ShareState {
            limit: Some(share.share.clone()),
            reserved_tokens: 30,
            cap: share.ceiling.clone(),
            cap_reserved_tokens: 30,
            ..ShareState::default()
        };
        let none = Holds::default();
        assert_eq!(
            borrowing_settlement(&stash, &none, Charge::Estimated(500), 60),
            vec![]
        );
        assert_eq!(
            borrowing_settlement(&stash, &none, Charge::Reported(0), 60),
            vec![
                refund(Some(&share.share), -30),
                refund(share.ceiling.as_ref(), -30)
            ]
        );
    }

    #[rstest]
    fn an_estimated_charge_keeps_the_fallback_targets_hold_where_a_reported_zero_refunds_it() {
        let share = borrower(BorrowerKind::Team, Some(100), Some(2.0));
        let hold = Hold {
            descriptor: Some(share.share.clone()),
            reserved_tokens: 30,
            cap: share.ceiling.clone(),
            cap_reserved_tokens: 30,
            ..Hold::empty()
        };
        let state = ShareState {
            fallback_holds: vec![("fallback".into(), Arc::new(hold))],
            ..ShareState::default()
        };
        let served = state.hold("fallback");
        assert_eq!(
            team_settlement(&state, served.as_deref(), Charge::Estimated(500), 60),
            vec![]
        );
        assert_eq!(
            team_settlement(&state, served.as_deref(), Charge::Reported(0), 60),
            vec![
                refund(Some(&share.share), -30),
                refund(share.ceiling.as_ref(), -30)
            ]
        );
    }

    #[rstest]
    fn a_borrower_without_a_share_on_the_served_target_settles_nothing_to_its_admitting_share() {
        let mut state = ShareState {
            limit: Some(borrower(BorrowerKind::Team, Some(100), None).share),
            reserved_tokens: 30,
            ..ShareState::default()
        };
        state.put_hold("other", &Arc::new(Hold::empty()));
        let served = state.hold("other");
        let operations = team_settlement(&state, served.as_deref(), Charge::Reported(90), 60);
        assert_eq!(
            operations,
            vec![ReservationOp::Increment {
                counter: borrower(BorrowerKind::Team, Some(100), None)
                    .share
                    .counter(CounterKind::Tokens),
                amount: -30,
                ttl: Some(60),
            }]
        );
    }
}
