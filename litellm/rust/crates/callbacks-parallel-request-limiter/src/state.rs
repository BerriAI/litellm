use std::{
    collections::BTreeSet,
    sync::{Mutex, MutexGuard, PoisonError},
};

use litellm_redis_store::RequestRedis;
use serde::{Deserialize, Serialize};

use crate::{
    engine::{Op, SlotAcquisition},
    enqueued::EnqueuedReservation,
    limiter::PreCallFacts,
    ptu::Ceiling,
    status::{Response, WindowIdentity},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimitedModel {
    pub requested: String,
    pub group: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapDecision {
    pub explicit: bool,
    pub exempt: bool,
    pub responses: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Stash {
    pub pre_call: Option<PreCallFacts>,
    pub cap: CapDecision,
    pub owned: bool,
    pub first_pass_admitted: bool,
    pub owner_call_id: Option<String>,
    pub identity: Option<String>,
    pub rate_limit_response: Option<Response>,
    pub parallel_slot: Option<SlotAcquisition>,
    pub reserved_tokens: i64,
    pub reserved_model: Option<RateLimitedModel>,
    pub reserved_scopes: BTreeSet<(String, String)>,
    pub fallback_models: BTreeSet<String>,
    pub reservation_released: bool,
    pub tpm_limited_tags: BTreeSet<String>,
    pub ptu_ceiling: Option<Ceiling>,
    pub ptu_reserved_tokens: i64,
    pub ptu_windows: Vec<WindowIdentity>,
    pub itpm: IoReservation,
    pub otpm: IoReservation,
    pub io_reservation_released: bool,
    pub batch_tpd_refunds: Vec<Op>,
    pub batch_enqueued: Option<EnqueuedReservation>,
}

#[derive(Clone, Debug, Default)]
pub struct IoReservation {
    pub tokens: i64,
    pub scopes: BTreeSet<(String, String)>,
    pub windows: Vec<WindowIdentity>,
}

impl Stash {
    pub fn discard_released_tpm_reservation(&mut self) {
        if self.reservation_released {
            self.reserved_tokens = 0;
            self.ptu_reserved_tokens = 0;
            self.reservation_released = false;
        }
    }

    pub fn discard_released_io_reservation(&mut self) {
        if self.io_reservation_released {
            self.itpm.tokens = 0;
            self.otpm.tokens = 0;
            self.io_reservation_released = false;
        }
    }

    pub fn clear_reserved_tokens(&mut self) {
        self.reserved_tokens = 0;
        self.ptu_reserved_tokens = 0;
        self.itpm.tokens = 0;
        self.otpm.tokens = 0;
    }

    pub fn mark_reservation_released(&mut self) {
        self.reservation_released = true;
        self.io_reservation_released = true;
    }

    pub fn for_call(&self, call_id: Option<&str>) -> bool {
        match (&self.owner_call_id, call_id) {
            (Some(owner), Some(call_id)) => owner == call_id,
            _ => true,
        }
    }
}

#[derive(Default)]
pub struct LimiterState {
    stash: Mutex<Stash>,
    request: Mutex<Option<RequestRedis>>,
    pub(crate) release: tokio::sync::Mutex<()>,
}

impl LimiterState {
    pub fn stash(&self) -> MutexGuard<'_, Stash> {
        self.stash.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn snapshot(&self) -> Stash {
        self.stash().clone()
    }

    pub fn bind_request(&self, request: Option<RequestRedis>) {
        *self.request.lock().unwrap_or_else(PoisonError::into_inner) = request;
    }

    pub fn request(&self) -> Option<RequestRedis> {
        self.request
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}
