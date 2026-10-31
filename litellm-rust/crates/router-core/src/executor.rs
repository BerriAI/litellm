//! Executors: each attempt resolves one after its deployment is picked. An (op, provider) pair in
//! the route-rule table with a registered [`NativeRoute`] runs in-process; anything else goes
//! through `Host::invoke`. An attempt runs on exactly one executor and is never replayed on
//! another: a native failure is that attempt's failure, and a retry selects afresh.

use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    config::Op,
    error::{AttemptError, Unsupported},
    host::{Attempt, Output},
};

/// The (op, provider) pairs that have a Rust route machine.
pub const NATIVE_RULES: &[(Op, &str)] = &[(Op::Messages, "anthropic")];

/// A Rust-native route machine (an `inference-*` crate) adapted to an attempt. The embedder
/// registers one per native rule; for Messages on Anthropic that is an adapter over
/// `litellm_inference_messages::MessagesRoute::execute`.
#[async_trait]
pub trait NativeRoute: Send + Sync {
    async fn execute(&self, attempt: &Attempt) -> Result<Output, AttemptError>;
}

#[derive(Clone)]
pub enum Executor {
    Native(Arc<dyn NativeRoute>),
    Host,
}

#[derive(Clone, Default)]
pub struct Executors {
    native: Vec<(Op, String, Arc<dyn NativeRoute>)>,
    host_invokes: bool,
}

impl Executors {
    pub fn new(host_invokes: bool) -> Self {
        Self {
            native: Vec::new(),
            host_invokes,
        }
    }

    /// Register a native route for a rule in [`NATIVE_RULES`]. A pair outside the table stays
    /// on the host, so a route cannot silently take over an operation not yet ported.
    pub fn register(&mut self, op: Op, provider: &str, route: Arc<dyn NativeRoute>) -> bool {
        let ruled = NATIVE_RULES
            .iter()
            .any(|(rule_op, rule_provider)| *rule_op == op && *rule_provider == provider);
        if ruled {
            self.native.push((op, provider.to_owned(), route));
        }
        ruled
    }

    pub fn resolve(&self, op: &Op, provider: &str) -> Result<Executor, Unsupported> {
        if let Some((_, _, route)) = self
            .native
            .iter()
            .find(|(rule_op, rule_provider, _)| rule_op == op && rule_provider == provider)
        {
            return Ok(Executor::Native(route.clone()));
        }
        if self.host_invokes {
            return Ok(Executor::Host);
        }
        Err(Unsupported::Executor {
            op: op.clone(),
            provider: provider.to_owned(),
        })
    }
}
