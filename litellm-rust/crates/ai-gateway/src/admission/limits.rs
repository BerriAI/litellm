//! Per-key budget and per-minute request/token windows, checked and reserved under one lock.
//!
//! Process-local: in a multi-replica deployment these counters would live in Redis, as the
//! proxy's do. The check itself is a couple of integer compares per request.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::constants::DEFAULT_INPUT_COST_PER_TOKEN;

use super::identity::{Identity, KeyLimits};

const WINDOW: Duration = Duration::from_secs(60);

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LimitExceeded {
    Budget,
    TokensPerMinute,
    RequestsPerMinute,
}

#[derive(Debug)]
struct KeyWindow {
    started: Instant,
    tokens: u64,
    requests: u64,
    reserved_spend: f64,
}

#[derive(Default)]
pub struct Limits {
    windows: Mutex<HashMap<String, KeyWindow>>,
}

impl Limits {
    /// Admit `input_tokens` for the identity, or say which limit it would cross.
    pub fn reserve(&self, identity: &Identity, input_tokens: usize) -> Result<(), LimitExceeded> {
        let Identity::VirtualKey { key_hash, limits } = identity else {
            return Ok(());
        };
        let Ok(mut windows) = self.windows.lock() else {
            return Ok(());
        };
        let now = Instant::now();
        let window = windows.entry(key_hash.clone()).or_insert(KeyWindow {
            started: now,
            tokens: 0,
            requests: 0,
            reserved_spend: 0.0,
        });
        if now.duration_since(window.started) >= WINDOW {
            window.started = now;
            window.tokens = 0;
            window.requests = 0;
        }
        let tokens = input_tokens as u64;
        let cost = input_tokens as f64 * DEFAULT_INPUT_COST_PER_TOKEN;
        check(limits, window, tokens, cost)?;
        window.tokens += tokens;
        window.requests += 1;
        window.reserved_spend += cost;
        Ok(())
    }
}

fn check(
    limits: &KeyLimits,
    window: &KeyWindow,
    tokens: u64,
    cost: f64,
) -> Result<(), LimitExceeded> {
    if let Some(max_budget) = limits.max_budget
        && limits.spend + window.reserved_spend + cost > max_budget
    {
        return Err(LimitExceeded::Budget);
    }
    if let Some(tpm) = limits.tpm_limit
        && window.tokens + tokens > tpm
    {
        return Err(LimitExceeded::TokensPerMinute);
    }
    if let Some(rpm) = limits.rpm_limit
        && window.requests + 1 > rpm
    {
        return Err(LimitExceeded::RequestsPerMinute);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn key(limits: KeyLimits) -> Identity {
        Identity::VirtualKey {
            key_hash: "hash".to_string(),
            limits: Arc::new(limits),
        }
    }

    #[test]
    fn master_key_is_never_limited() {
        let limits = Limits::default();
        assert_eq!(limits.reserve(&Identity::Master, usize::MAX), Ok(()));
    }

    #[test]
    fn tpm_window_accumulates_and_rejects_on_overflow() {
        let limits = Limits::default();
        let identity = key(KeyLimits {
            tpm_limit: Some(100),
            ..KeyLimits::default()
        });
        assert_eq!(limits.reserve(&identity, 60), Ok(()));
        assert_eq!(
            limits.reserve(&identity, 50),
            Err(LimitExceeded::TokensPerMinute)
        );
        assert_eq!(limits.reserve(&identity, 40), Ok(()));
    }

    #[test]
    fn rpm_counts_requests() {
        let limits = Limits::default();
        let identity = key(KeyLimits {
            rpm_limit: Some(2),
            ..KeyLimits::default()
        });
        assert_eq!(limits.reserve(&identity, 1), Ok(()));
        assert_eq!(limits.reserve(&identity, 1), Ok(()));
        assert_eq!(
            limits.reserve(&identity, 1),
            Err(LimitExceeded::RequestsPerMinute)
        );
    }

    #[test]
    fn budget_includes_prior_spend_and_local_reservations() {
        let limits = Limits::default();
        let identity = key(KeyLimits {
            max_budget: Some(1.0),
            spend: 0.5,
            ..KeyLimits::default()
        });
        let tokens_for_quarter_dollar = (0.25 / DEFAULT_INPUT_COST_PER_TOKEN) as usize;
        assert_eq!(limits.reserve(&identity, tokens_for_quarter_dollar), Ok(()));
        assert_eq!(limits.reserve(&identity, tokens_for_quarter_dollar), Ok(()));
        assert_eq!(
            limits.reserve(&identity, tokens_for_quarter_dollar),
            Err(LimitExceeded::Budget)
        );
    }

    #[test]
    fn a_rejected_request_reserves_nothing() {
        let limits = Limits::default();
        let identity = key(KeyLimits {
            tpm_limit: Some(10),
            rpm_limit: Some(5),
            ..KeyLimits::default()
        });
        assert_eq!(
            limits.reserve(&identity, 11),
            Err(LimitExceeded::TokensPerMinute)
        );
        assert_eq!(limits.reserve(&identity, 10), Ok(()));
    }
}
