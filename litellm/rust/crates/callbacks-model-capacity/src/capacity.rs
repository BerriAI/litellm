use std::sync::Arc;

use serde_json::Value;

use crate::{
    error::CapacityError,
    model::{
        BorrowingLimits, Counter, CounterKind, Descriptor, GroupInfo, Increment, KeyFacts,
        LimiterResponse, LoggedGroup, MODEL_SATURATION_CHECK, PRIORITY_CLASS_SHARE_AT_SATURATION,
        PRIORITY_MODEL, Priority, RateLimit, ReservationOp, Settings,
        TEAM_MODEL_LIMIT_AT_SATURATION, TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE, TeamShare, Window,
    },
    priority::{
        allocation, counter_value, normalized_weights, pool_member, priority_class, team_member,
        team_share,
    },
    pyfmt,
    refusal::{Context, Refusal},
    seam::{CapacityHost, Estimate, Throttle, Trace, V3Limiter},
    state::{Hold, SharedStash},
};

pub struct Engine {
    pub limiter: Arc<dyn V3Limiter>,
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
    Allow,
    Refuse(Refusal),
    Raise(CapacityError),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SaturationFallback {
    pub enabled: bool,
    pub invalid: Option<String>,
    pub armed: bool,
}

pub struct BorrowingShare {
    pub limits: BorrowingLimits,
    pub share: Descriptor,
    pub ceiling: Option<Descriptor>,
}

impl BorrowingShare {
    pub fn enforced(&self, saturated: bool) -> Option<&Descriptor> {
        saturated.then_some(&self.share)
    }
}

pub struct Request<'a> {
    pub key: &'a KeyFacts,
    pub call_type: Option<&'a str>,
    pub settings: &'a Settings,
    pub stash: &'a SharedStash,
    pub fallback: SaturationFallback,
}

impl Request<'_> {
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

fn python_int(value: &Value) -> Result<i64, CapacityError> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|float| float.trunc() as i64))
            .ok_or_else(|| CapacityError::Config(format!("{number} is not an int"))),
        Value::Bool(flag) => Ok(i64::from(*flag)),
        Value::String(text) => pyfmt::numeric_text(text)
            .and_then(|text| text.parse::<i64>().ok())
            .ok_or_else(|| {
                CapacityError::Config(format!(
                    "invalid literal for int() with base 10: {}",
                    pyfmt::str_repr(text)
                ))
            }),
        other => Err(CapacityError::Config(format!("{other} is not an int"))),
    }
}

fn python_float(value: &Value) -> Result<f64, CapacityError> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| CapacityError::Config(format!("{number} is not a float"))),
        Value::Bool(flag) => Ok(f64::from(u8::from(*flag))),
        Value::String(text) => pyfmt::numeric_text(text)
            .and_then(|text| text.parse::<f64>().ok())
            .ok_or_else(|| {
                CapacityError::Config(format!(
                    "could not convert string to float: {}",
                    pyfmt::str_repr(text)
                ))
            }),
        other => Err(CapacityError::Config(format!("{other} is not a float"))),
    }
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
    stash: &crate::state::Stash,
    served: Option<&Hold>,
    total_tokens: i64,
    ttl: i64,
) -> Vec<ReservationOp> {
    let mut operations = Vec::new();
    match served {
        None => {
            operations.extend(window_guarded(
                team_increment(
                    stash.borrowing_limit.as_ref(),
                    total_tokens - stash.reserved_tokens,
                    ttl,
                ),
                stash.reserved_window.as_ref(),
            ));
            operations.extend(window_guarded(
                team_increment(
                    stash.borrowing_cap.as_ref(),
                    total_tokens - stash.cap_reserved_tokens,
                    ttl,
                ),
                stash.cap_reserved_window.as_ref(),
            ));
        }
        Some(hold) => {
            operations.extend(window_guarded(
                team_increment(stash.borrowing_limit.as_ref(), -stash.reserved_tokens, ttl),
                stash.reserved_window.as_ref(),
            ));
            operations.extend(window_guarded(
                team_increment(
                    stash.borrowing_cap.as_ref(),
                    -stash.cap_reserved_tokens,
                    ttl,
                ),
                stash.cap_reserved_window.as_ref(),
            ));
            operations.extend(hold_settlement(hold, total_tokens, ttl));
        }
    }
    operations
}

fn hold_settlement(hold: &Hold, used: i64, ttl: i64) -> Vec<ReservationOp> {
    let mut operations = window_guarded(
        team_increment(hold.descriptor.as_ref(), used - hold.reserved_tokens, ttl),
        hold.window.as_ref(),
    );
    operations.extend(window_guarded(
        team_increment(hold.cap.as_ref(), used - hold.cap_reserved_tokens, ttl),
        hold.cap_window.as_ref(),
    ));
    operations
}

pub struct Settlement<'a> {
    pub model_group: Option<&'a LoggedGroup>,
    pub team_id: Option<&'a str>,
    pub team_alias: Option<&'a str>,
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
                let _ = self.host.log("error", &message).await;
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
                saturation = saturation.max(python_int(&value)? as f64 / rpm as f64);
            }
        }
        if let Some(tpm) = group.tpm.filter(|tpm| *tpm > 0) {
            let counter = scope(MODEL_SATURATION_CHECK, model).with_kind(CounterKind::Tokens);
            if let Some(value) = self.counter_value(&counter, ttl).await? {
                saturation = saturation.max(python_float(&value)? / tpm as f64);
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
                let _ = self.host.log("warning", &message).await;
                self.limiter.read_local_counter(counter).await?
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
        if settings.priority_reservation.is_none() || !settings.class_split {
            return Ok(None);
        }
        let class = priority_class(settings, request.priority())?;
        let member = request
            .key
            .team_text()
            .map_or_else(|| pool_member(&class), |team| team_member(&team));
        let members = match self
            .host
            .touch(model, &member, &class, settings.window_size)
            .await
        {
            Ok(members) => members,
            Err(error) if error.is_registry_unavailable() => {
                let message = format!(
                    "Active team registry unavailable for {model}, holding team {} to the class pool: {}",
                    python_optional(request.key.team_text().as_deref()),
                    error.python_message()
                );
                let _ = self.host.log("warning", &message).await;
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if request.key.team_id.is_none() {
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
            self.host.unlicensed_priority().await?;
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
                value: counter_value(model, request.key.team_text().as_deref().unwrap_or("None")),
                rate_limit: Some(rate_limit),
            },
        }])
    }

    pub async fn borrowing_share(
        &self,
        request: &Request<'_>,
        model: &str,
    ) -> Result<Option<BorrowingShare>, CapacityError> {
        let Some(limits) = self.limiter.borrowing_limits(model).await? else {
            return Ok(None);
        };
        if limits.tpm.is_none() && limits.rpm.is_none() {
            return Ok(None);
        }
        let value = format!(
            "{}:{}",
            request.key.team_text().as_deref().unwrap_or("None"),
            limits.group
        );
        let descriptor = |key: &str, tpm: Option<i64>, rpm: Option<i64>| Descriptor {
            key: key.into(),
            value: value.clone(),
            rate_limit: Some(RateLimit {
                requests_per_unit: rpm,
                tokens_per_unit: tpm,
                window_size: Some(request.settings.window_size),
            }),
        };
        let share = descriptor(TEAM_MODEL_LIMIT_AT_SATURATION, limits.tpm, limits.rpm);
        let ceiling = limits
            .ceiling()
            .map(|(tpm, rpm)| descriptor(TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE, tpm, rpm));
        Ok(Some(BorrowingShare {
            limits,
            share,
            ceiling,
        }))
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
                .limiter
                .estimate_tokens(tpm_share, model, call_type)
                .await?
            {
                Estimate::NoBody => 0,
                Estimate::Failed => 1,
                Estimate::Tokens(tokens) => tokens.max(1),
            },
        )
    }

    pub async fn admit(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
    ) -> Result<(), AdmitError> {
        let saturation = self.saturation(request.settings, model, group).await;
        let threshold = request.settings.saturation_threshold;
        if saturation >= threshold && self.fallback_requested(&request.fallback).await {
            let priority = request.priority_value();
            let context = Context {
                model,
                priority: priority.as_ref(),
                team_id: None,
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
            let _ = self.host.log("warning", &message).await;
        }
        fallback.enabled && fallback.armed
    }

    async fn check_rate_limits(
        &self,
        request: &Request<'_>,
        model: &str,
        group: &GroupInfo,
        saturation: f64,
    ) -> Result<(), AdmitError> {
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
            stash.priority = request.priority_value();
        }
        let team_share = self.borrowing_share(request, model).await?;
        let enforced_team = team_share
            .as_ref()
            .and_then(|share| share.enforced(enforce_priority))
            .cloned();
        let team_cap = team_share.as_ref().and_then(|share| share.ceiling.clone());
        {
            let mut stash = request.stash.lock();
            stash.borrowing_limit = enforced_team.clone();
            stash.borrowing_share = team_share.as_ref().map(|share| share.share.clone());
            stash.borrowing_cap = team_cap.clone();
        }

        let mut enforced = vec![model_wide];
        if enforce_priority {
            enforced.extend(priority_descriptors.iter().cloned());
        }
        let team_reservation = match &team_share {
            Some(team_share) if enforced_team.is_some() || team_cap.is_some() => {
                self.token_reservation(&team_share.share, model, request.call_type)
                    .await?
            }
            _ => 0,
        };
        let mut increments: Vec<Increment> = enforced
            .iter()
            .map(|_| Increment {
                requests: 1,
                tokens: 0,
            })
            .collect();
        for descriptor in enforced_team.iter().chain(team_cap.iter()) {
            enforced.push(descriptor.clone());
            increments.push(Increment {
                requests: 1,
                tokens: team_reservation,
            });
        }
        let atomic = self
            .limiter
            .check_and_increment(&enforced, &increments)
            .await?;
        if atomic.over_limit() {
            let priority = request.priority_value();
            let team = request.key.team_text();
            let context = Context {
                model,
                priority: priority.as_ref(),
                team_id: team.as_deref(),
                saturation,
                window_size: settings.window_size,
            };
            return Err(AdmitError::Refused(Self::refusal(
                &context,
                &atomic,
                group,
                team_share.as_ref(),
                share.as_ref(),
            )));
        }
        {
            let mut stash = request.stash.lock();
            let reserved_window = |descriptor: &Option<Descriptor>| {
                descriptor
                    .as_ref()
                    .filter(|_| team_reservation != 0)
                    .and_then(|descriptor| {
                        atomic.window_of(&descriptor.counter(CounterKind::Tokens))
                    })
            };
            stash.reserved_tokens = if enforced_team.is_some() {
                team_reservation
            } else {
                0
            };
            stash.cap_reserved_tokens = if team_cap.is_some() {
                team_reservation
            } else {
                0
            };
            stash.reserved_window = reserved_window(&enforced_team);
            stash.cap_reserved_window = reserved_window(&team_cap);
        }
        if let Some(tpm_share) = team_share
            .as_ref()
            .and_then(|share| share.share.tokens_per_unit())
            && team_reservation != 0
            && settings.rate_limit_type != crate::model::TokenRateLimitType::Input
        {
            self.limiter
                .cap_output(tpm_share, request.call_type)
                .await?;
        }
        if !priority_descriptors.is_empty() && share.is_some() && !enforce_priority {
            self.limiter.publish(&[atomic.id]).await?;
        } else if !priority_descriptors.is_empty() && !enforce_priority {
            let tracking = self
                .limiter
                .should_rate_limit(&priority_descriptors, false, false)
                .await?;
            self.limiter.publish(&[atomic.id, tracking.id]).await?;
        } else {
            self.limiter.publish(&[atomic.id]).await?;
        }
        Ok(())
    }

    fn refusal(
        context: &Context<'_>,
        atomic: &LimiterResponse,
        group: &GroupInfo,
        team_share: Option<&BorrowingShare>,
        class_share: Option<&TeamShare>,
    ) -> Refusal {
        for status in atomic.statuses.iter().filter(|status| status.over_limit()) {
            match status.descriptor_key.as_str() {
                MODEL_SATURATION_CHECK => return context.model_capacity(status, group),
                PRIORITY_MODEL => return context.priority_pool(status, group),
                TEAM_MODEL_LIMIT_AT_SATURATION | TEAM_MODEL_LIMIT_MAX_BORROW_MULTIPLE => {
                    if let Some(team_share) = team_share {
                        return context.team_share(status, &team_share.limits);
                    }
                }
                PRIORITY_CLASS_SHARE_AT_SATURATION => {
                    if let Some(share) = class_share {
                        return context.class_share(status, share);
                    }
                }
                _ => {}
            }
        }
        context.unknown(atomic.statuses.iter().find(|status| status.over_limit()))
    }

    pub async fn pre_call(&self, request: &Request<'_>, model: &str, group: &GroupInfo) -> PreCall {
        match self.admit(request, model, group).await {
            Ok(()) => PreCall::Allow,
            Err(AdmitError::Refused(refusal)) => {
                self.throttled(&refusal, model, request.key).await;
                PreCall::Refuse(refusal)
            }
            Err(AdmitError::Failed(error)) if error.is_http() => PreCall::Raise(error),
            Err(AdmitError::Failed(error)) => {
                let message = format!(
                    "Error in dynamic rate limiter: {}, allowing request",
                    error.python_message()
                );
                let _ = self.host.log("error", &message).await;
                PreCall::Allow
            }
        }
    }

    async fn throttled(&self, refusal: &Refusal, model: &str, key: &KeyFacts) {
        let (Some(team), Some(reason)) = (key.team_text(), refusal.header("rate_limited_by"))
        else {
            return;
        };
        let or_default = |value: Option<&Value>, default: &str| match value {
            Some(value) if pyfmt::truthy(Some(value)) => value.clone(),
            _ => Value::from(default),
        };
        let event = Throttle {
            model_group: model.to_owned(),
            team,
            team_alias: key.team_alias.clone(),
            priority: or_default(refusal.header("x-litellm-priority"), "default"),
            rate_limit_type: pyfmt::value(Some(&or_default(
                refusal.header("rate_limit_type"),
                "tokens",
            ))),
            reason: pyfmt::value(Some(reason)),
        };
        let _ = self.host.throttled(&event).await;
    }

    pub async fn enforce_frame(
        &self,
        request: &Request<'_>,
        model: &str,
        connection: Option<&SharedStash>,
    ) -> Result<Frame, CapacityError> {
        let Some(group) = self.host.group(model).await? else {
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
            stash.lock().priority_class_share = Some(frame_share);
        }
        let borrowing = self.borrowing_share(request, model).await?;
        if share_descriptors.is_empty() && borrowing.is_none() {
            return Ok(Frame::Allow);
        }
        let saturation = self.saturation(settings, model, &group).await;
        let saturated = saturation >= settings.saturation_threshold;
        let enforced_team = borrowing
            .as_ref()
            .and_then(|borrowing| borrowing.enforced(saturated))
            .cloned();
        let team_cap = borrowing
            .as_ref()
            .and_then(|borrowing| borrowing.ceiling.clone());
        if let (Some(stash), Some(borrowing)) = (connection, &borrowing) {
            let mut stash = stash.lock();
            if enforced_team.is_some() || stash.reserved_tokens == 0 {
                stash.borrowing_limit = enforced_team.clone();
            }
            stash.borrowing_share = Some(borrowing.share.clone());
            stash.borrowing_cap = team_cap.clone();
        }
        let mut descriptors = if saturated {
            share_descriptors
        } else {
            Vec::new()
        };
        descriptors.extend(enforced_team);
        descriptors.extend(team_cap);
        if descriptors.is_empty() {
            return Ok(Frame::Allow);
        }
        let windows = socket_turn_tpm_windows(&descriptors);
        let response = match self.limiter.should_rate_limit(&windows, true, true).await {
            Ok(response) => response,
            Err(error) => return Ok(Frame::Unverifiable(error)),
        };
        let Some(over_limit) = response.statuses.iter().find(|status| status.over_limit()) else {
            return Ok(Frame::Allow);
        };
        let priority = request.priority_value();
        let team = request.key.team_text();
        let context = Context {
            model,
            priority: priority.as_ref(),
            team_id: team.as_deref(),
            saturation,
            window_size: settings.window_size,
        };
        let over_class_share = over_limit.descriptor_key == PRIORITY_CLASS_SHARE_AT_SATURATION;
        let refusal = match (&share, &borrowing) {
            (Some(share), _) if over_class_share || borrowing.is_none() => {
                context.class_share(over_limit, share)
            }
            (_, Some(borrowing)) => context.team_share(over_limit, &borrowing.limits),
            _ => return Ok(Frame::Allow),
        };
        self.throttled(&refusal, model, request.key).await;
        Ok(Frame::Refuse(refusal))
    }

    pub async fn fallback_target_allowed(
        &self,
        request: &Request<'_>,
        model: &str,
        commit: bool,
    ) -> Result<bool, CapacityError> {
        let team_share = self.borrowing_share(request, model).await?;
        let group = self.host.group(model).await?;
        let (Some(team_share), Some(group)) = (team_share, group) else {
            return Ok(true);
        };
        let settings = request.settings;
        let saturation = self.saturation(settings, model, &group).await;
        let enforced = team_share
            .enforced(saturation >= settings.saturation_threshold)
            .cloned();
        let cap = team_share.ceiling.clone();
        let reservation = if enforced.is_some() || cap.is_some() {
            let call_type = self.limiter.stash_call_type().await?;
            self.token_reservation(&team_share.share, model, call_type.as_deref())
                .await?
        } else {
            0
        };
        let team_descriptors: Vec<Descriptor> =
            enforced.iter().chain(cap.iter()).cloned().collect();
        if !commit {
            for descriptor in &team_descriptors {
                if !self.share_has_room(descriptor, reservation).await? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let mut increments = vec![Increment {
            requests: 1,
            tokens: 0,
        }];
        increments.extend(team_descriptors.iter().map(|_| Increment {
            requests: 1,
            tokens: reservation,
        }));
        let mut descriptors = vec![Self::model_wide(settings, model, &group)];
        descriptors.extend(team_descriptors);
        let response = self
            .limiter
            .check_and_increment(&descriptors, &increments)
            .await?;
        if response.over_limit() {
            return Ok(false);
        }
        let window_of = |descriptor: &Option<Descriptor>| {
            descriptor
                .as_ref()
                .and_then(|descriptor| response.window_of(&descriptor.counter(CounterKind::Tokens)))
        };
        let hold = Hold {
            reserved_tokens: if enforced.is_some() { reservation } else { 0 },
            window: window_of(&enforced),
            cap_reserved_tokens: if cap.is_some() { reservation } else { 0 },
            cap_window: window_of(&cap),
            descriptor: enforced,
            cap,
            share: Some(team_share.share.clone()),
        };
        Self::hold_fallback_share(request.stash, model, &team_share.share, hold);
        Ok(true)
    }

    async fn share_has_room(
        &self,
        descriptor: &Descriptor,
        tokens: i64,
    ) -> Result<bool, CapacityError> {
        let checked = self
            .limiter
            .should_rate_limit(std::slice::from_ref(descriptor), true, false)
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

    fn hold_fallback_share(stash: &SharedStash, model: &str, share: &Descriptor, hold: Hold) {
        let mut stash = stash.lock();
        let earlier = stash.hold(model);
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
        stash.put_hold(model, &hold);
        stash.put_hold(group, &hold);
    }

    pub async fn settle_success(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        settlement: &Settlement<'_>,
        priority: Option<&str>,
        total_tokens: i64,
    ) -> Result<(), CapacityError> {
        let ttl = settings.window_size;
        let group = settlement.model_group;
        let (team_tokens, served) = match stash {
            None => (Vec::new(), None),
            Some(stash) => {
                let mut stash = stash.lock();
                let served = match group {
                    Some(group) => group.key()?.and_then(|key| stash.hold(key)),
                    None => None,
                };
                let team_tokens = team_settlement(&stash, served.as_deref(), total_tokens, ttl);
                stash.release_reservations();
                if let Some(hold) = &served {
                    stash.drop_hold(hold);
                }
                (team_tokens, served)
            }
        };
        let Some(model_group) = group.filter(|group| group.truthy) else {
            if !team_tokens.is_empty() {
                self.limiter.apply(&team_tokens, Trace::Attempt).await?;
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
        let class_share = stash.and_then(|stash| stash.lock().priority_class_share.clone());
        let priority_counter = match &class_share {
            None => scope(PRIORITY_MODEL, &priority_key),
            Some(share) => scope(
                &share.key,
                &settlement.team_id.map_or_else(
                    || share.value.clone(),
                    |team| counter_value(model_group, team),
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
            self.limiter.apply(&operations, Trace::Attempt).await?;
            self.record_team_share(
                settings,
                stash,
                Some(model_group),
                settlement,
                served.as_deref(),
            )
            .await;
        }
        Ok(())
    }

    pub async fn settle_failure(
        &self,
        settings: &Settings,
        stash: &SharedStash,
        failed_group: Option<&LoggedGroup>,
        used: i64,
    ) -> Result<Option<Arc<Hold>>, CapacityError> {
        let ttl = settings.window_size;
        let (refund, failed_hold) = {
            let mut stash = stash.lock();
            let failed_hold = match failed_group {
                Some(group) => group.key()?.and_then(|key| stash.hold(key)),
                None => None,
            };
            let own_usage = if failed_hold.is_none() { used } else { 0 };
            let mut refund = window_guarded(
                team_increment(
                    stash.borrowing_limit.as_ref(),
                    own_usage - stash.reserved_tokens,
                    ttl,
                ),
                stash.reserved_window.as_ref(),
            );
            refund.extend(window_guarded(
                team_increment(
                    stash.borrowing_cap.as_ref(),
                    own_usage - stash.cap_reserved_tokens,
                    ttl,
                ),
                stash.cap_reserved_window.as_ref(),
            ));
            if let Some(hold) = &failed_hold {
                refund.extend(hold_settlement(hold, used, ttl));
            }
            stash.release_reservations();
            if let Some(hold) = &failed_hold {
                stash.drop_hold(hold);
            }
            (refund, failed_hold)
        };
        if !refund.is_empty() {
            self.limiter.apply(&refund, Trace::Untraced).await?;
        }
        Ok(failed_hold)
    }

    pub async fn release_admission(
        &self,
        settings: &Settings,
        stash: &SharedStash,
    ) -> Result<(), CapacityError> {
        let refund = {
            let mut stash = stash.lock();
            let refund = team_settlement(&stash, None, 0, settings.window_size);
            stash.release_reservations();
            refund
        };
        if refund.is_empty() {
            return Ok(());
        }
        self.limiter.apply(&refund, Trace::Key).await
    }

    pub async fn record_team_share(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        model_group: Option<&str>,
        settlement: &Settlement<'_>,
        hold: Option<&Hold>,
    ) {
        if let Err(error) = self
            .try_record_team_share(settings, stash, model_group, settlement, hold)
            .await
        {
            let message = format!(
                "Dynamic rate limiter share metrics failed: {}",
                error.python_message()
            );
            let _ = self.host.log("warning", &message).await;
        }
    }

    async fn try_record_team_share(
        &self,
        settings: &Settings,
        stash: Option<&SharedStash>,
        model_group: Option<&str>,
        settlement: &Settlement<'_>,
        hold: Option<&Hold>,
    ) -> Result<(), CapacityError> {
        if !self.host.metrics_enabled().await? {
            return Ok(());
        }
        let (Some(stash), Some(model_group), Some(team)) = (stash, model_group, settlement.team_id)
        else {
            return Ok(());
        };
        let (share, idle_share, priority) = {
            let stash = stash.lock();
            let share = match hold {
                Some(hold) => tightest_of(
                    tightest(&[hold.descriptor.as_ref()]),
                    capped_idle_share(hold.share.as_ref(), hold.cap.as_ref()),
                ),
                None => tightest_of(
                    tightest(&[
                        stash.borrowing_limit.as_ref(),
                        stash.priority_class_share.as_ref(),
                    ]),
                    capped_idle_share(stash.borrowing_share.as_ref(), stash.borrowing_cap.as_ref()),
                ),
            };
            (share, stash.borrowing_share.clone(), stash.priority.clone())
        };
        let share = match (share, hold) {
            (Some(share), _) => Some(share),
            (None, Some(hold)) => tightest(&[hold.share.as_ref()]),
            (None, None) => {
                let group = self.limiter.model_group(model_group).await?;
                let on_model = format!("{team}:{group}");
                tightest(&[idle_share.as_ref().filter(|share| share.value == on_model)])
            }
        };
        let Some((descriptor, share_tokens)) = share else {
            return Ok(());
        };
        let used = self
            .counter_value(
                &descriptor.counter(CounterKind::Tokens),
                &settings.saturation_check_cache_ttl,
            )
            .await?;
        let used_tokens = match used {
            None => 0,
            Some(value) => python_float(&value)? as i64,
        };
        let gauge = crate::seam::ShareGauge {
            model_group: model_group.to_owned(),
            team: team.to_owned(),
            team_alias: settlement.team_alias.map(str::to_owned),
            priority: if pyfmt::truthy(priority.as_ref()) {
                priority.unwrap_or(Value::Null)
            } else {
                Value::from("default")
            },
            share_tokens,
            borrowed_tokens: (used_tokens - share_tokens).max(0),
        };
        self.host.set_share(&gauge).await
    }

    pub fn post_call_headers(key: &KeyFacts) -> Vec<(String, Value)> {
        let priority = key.priority_value();
        let header = if pyfmt::truthy(priority.as_ref()) {
            priority.unwrap_or(Value::Null)
        } else {
            Value::from("default")
        };
        let mut headers = Vec::new();
        if pyfmt::is_latin1(&pyfmt::value(Some(&header))) {
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

    use super::*;

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
}
