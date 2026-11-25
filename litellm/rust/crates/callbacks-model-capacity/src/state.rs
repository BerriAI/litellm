use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;

use crate::model::{BorrowerKind, Descriptor, Window};

#[derive(Debug, PartialEq)]
pub struct Hold {
    pub descriptor: Option<Descriptor>,
    pub reserved_tokens: i64,
    pub window: Option<Window>,
    pub cap: Option<Descriptor>,
    pub cap_reserved_tokens: i64,
    pub cap_window: Option<Window>,
    pub share: Option<Descriptor>,
}

impl Hold {
    /// The hold a borrower with no limit on a fallback target takes, so the target's usage never settles to its
    /// share on the admitting model.
    pub fn empty() -> Self {
        Self {
            descriptor: None,
            reserved_tokens: 0,
            window: None,
            cap: None,
            cap_reserved_tokens: 0,
            cap_window: None,
            share: None,
        }
    }
}

/// One borrower's share on the admitting model for this request: what was enforced, what was held on it, and the
/// holds taken on router fallback targets.
#[derive(Debug, Default)]
pub struct ShareState {
    pub limit: Option<Descriptor>,
    pub reserved_tokens: i64,
    pub reserved_window: Option<Window>,
    pub share: Option<Descriptor>,
    pub cap: Option<Descriptor>,
    pub cap_reserved_tokens: i64,
    pub cap_reserved_window: Option<Window>,
    pub fallback_holds: Vec<(String, Arc<Hold>)>,
}

/// What the pre-call hook hands to the settlement callbacks of the same request.
#[derive(Debug, Default)]
pub struct Stash {
    pub priority_class_share: Option<Descriptor>,
    pub priority_class_share_holder: Option<String>,
    pub priority: Option<Value>,
    pub borrowing: [ShareState; 3],
    pub output_cap: Option<(i64, Option<String>)>,
    pub turn_settled_tokens: i64,
    pub session_settled: bool,
    pub turn_model_group: Option<String>,
}

impl Stash {
    pub fn share(&self, kind: BorrowerKind) -> &ShareState {
        &self.borrowing[kind.index()]
    }

    pub fn share_mut(&mut self, kind: BorrowerKind) -> &mut ShareState {
        &mut self.borrowing[kind.index()]
    }
}

impl ShareState {
    pub fn hold(&self, model: &str) -> Option<Arc<Hold>> {
        self.fallback_holds
            .iter()
            .find(|(name, _)| name == model)
            .map(|(_, hold)| Arc::clone(hold))
    }

    pub fn put_hold(&mut self, model: &str, hold: &Arc<Hold>) {
        match self
            .fallback_holds
            .iter_mut()
            .find(|(name, _)| name == model)
        {
            Some((_, slot)) => *slot = Arc::clone(hold),
            None => self
                .fallback_holds
                .push((model.to_owned(), Arc::clone(hold))),
        }
    }

    pub fn clear_reservation(&mut self) {
        self.reserved_tokens = 0;
        self.cap_reserved_tokens = 0;
        self.reserved_window = None;
        self.cap_reserved_window = None;
    }

    pub fn drop_hold(&mut self, hold: Option<&Arc<Hold>>) {
        let Some(hold) = hold else {
            return;
        };
        self.fallback_holds
            .retain(|(_, held)| !Arc::ptr_eq(held, hold));
    }
}

pub type SharedStash = Arc<Mutex<Stash>>;
