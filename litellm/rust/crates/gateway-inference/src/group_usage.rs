//! `set_response_headers`' remaining-usage headers of the model group that served a call:
//! `_remaining_usage` over the group's configured `tpm` and `rpm` and this minute's counts.

use std::{sync::Arc, time::SystemTime};

use axum::{
    extract::{Request, State},
    http::{HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use litellm_redis_store::{Number, router_usage::Kind};
use litellm_router_core::{Deployment, UsageWindow, health::current_minute};

use crate::{Gateway, headers::Routing, routing::Host, routing_cache::counter};

const TOKENS: [&str; 2] = ["x-ratelimit-remaining-tokens", "x-ratelimit-limit-tokens"];
const REQUESTS: [&str; 2] = [
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-requests",
];

/// A deployment's configured limit, which `_set_model_group_info` adds up over the group.
fn limit(deployment: &Deployment, kind: Kind) -> Option<Number> {
    let value = deployment.raw_params.get(<&str>::from(kind))?;
    value
        .as_i64()
        .map(Number::Int)
        .or_else(|| value.as_f64().map(Number::Float))
}

fn total(deployments: &[Arc<Deployment>], kind: Kind) -> Option<Number> {
    deployments
        .iter()
        .filter_map(|deployment| limit(deployment, kind))
        .reduce(|sum, limit| sum + limit)
}

/// What a model group's deployments are configured to serve a minute, added up over the group.
/// A limit no deployment sets is `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupLimits {
    pub tpm: Option<i64>,
    pub rpm: Option<i64>,
}

/// The limits of the model group `name` in `router`, by the name a request asks for it under.
/// `None` when `name` is no model group.
pub fn group_limits(router: &litellm_router_core::Router, name: &str) -> Option<GroupLimits> {
    let deployments = router.registry().group(name)?;
    Some(GroupLimits {
        tpm: whole(&deployments, Kind::Tpm),
        rpm: whole(&deployments, Kind::Rpm),
    })
}

/// The group's limit as the whole number the dynamic rate limiter divides among priorities. A
/// fractional total is no limit to it.
fn whole(deployments: &[Arc<Deployment>], kind: Kind) -> Option<i64> {
    match total(deployments, kind)? {
        Number::Int(limit) => Some(limit),
        Number::Float(_) => None,
    }
}

/// `_usage_total`: the counts that are `int`s, summed, `None` when there is none.
fn used(counts: &[Option<Number>]) -> Option<i64> {
    counts
        .iter()
        .filter_map(|count| match count {
            Some(Number::Int(count)) => Some(*count),
            _ => None,
        })
        .reduce(|sum, count| sum.saturating_add(count))
}

fn remaining(limit: Number, used: Option<i64>) -> Number {
    let used = used.unwrap_or(0);
    match limit {
        Number::Int(limit) => Number::Int(limit.saturating_sub(used)),
        Number::Float(limit) => Number::Float(limit - used as f64),
    }
}

/// The headers `_remaining_usage` builds for `group`, tokens before requests.
async fn group_headers<H: Host>(gateway: &Gateway<H>, group: &str) -> Vec<(&'static str, Number)> {
    let Some(deployments) = gateway.router.registry().group(group) else {
        return Vec::new();
    };
    let limits = [
        (Kind::Tpm, TOKENS, total(&deployments, Kind::Tpm)),
        (Kind::Rpm, REQUESTS, total(&deployments, Kind::Rpm)),
    ];
    if limits.iter().all(|(_, _, limit)| limit.is_none()) {
        return Vec::new();
    }
    let minute = current_minute(SystemTime::now());
    let counters: Vec<_> = limits
        .iter()
        .flat_map(|(kind, _, _)| {
            deployments.iter().map(|deployment| {
                let window = UsageWindow {
                    deployment_id: deployment.id.clone(),
                    model: deployment.params.model.clone(),
                    minute: minute.clone(),
                };
                counter(&window, *kind)
            })
        })
        .collect();
    let counts = match gateway.routing().counted(&counters).await {
        Ok(counts) => counts,
        Err(error) => {
            tracing::warn!(%error, model_group = group, "model group usage was not read");
            return Vec::new();
        }
    };
    limits
        .into_iter()
        .zip(counts.chunks(deployments.len()))
        .filter_map(|((_, names, limit), counts)| Some((names, limit?, used(counts))))
        .flat_map(|([remaining_name, limit_name], limit, used)| {
            [
                (remaining_name, remaining(limit, used)),
                (limit_name, limit),
            ]
        })
        .collect()
}

/// Adds the served group's remaining-usage headers a response does not carry yet, as
/// `apply_remaining_usage_headers` does.
pub(crate) async fn told<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let Some(group) = response
        .extensions()
        .get::<Routing>()
        .and_then(Routing::counted_group)
        .map(str::to_owned)
    else {
        return response;
    };
    for (name, value) in group_headers(&gateway, &group).await {
        if let Ok(value) = HeaderValue::from_str(&value.to_json()) {
            response
                .headers_mut()
                .entry(HeaderName::from_static(name))
                .or_insert(value);
        }
    }
    response
}
