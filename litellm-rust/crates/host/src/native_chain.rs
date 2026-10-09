use crate::{
    error::HookError,
    hooks::NativeHooks,
    interceptors::{ExecutionFacts, RequestContext, WireRequest},
    lifecycle::CallEvent,
};

pub struct NativeChain(Vec<Box<dyn NativeHooks>>);

impl NativeChain {
    pub fn new(hooks: impl IntoIterator<Item = Box<dyn NativeHooks>>) -> Self {
        Self(hooks.into_iter().collect())
    }
}

impl NativeHooks for NativeChain {
    fn before_provider_request(
        &self,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<Box<WireRequest>, HookError> {
        self.0.iter().try_fold(wire, |wire, hooks| {
            hooks.before_provider_request(wire, context)
        })
    }

    fn result_ready(&self, facts: &ExecutionFacts) -> Result<(), HookError> {
        self.0
            .iter()
            .try_for_each(|hooks| hooks.result_ready(facts))
    }

    fn on_event(&self, event: &CallEvent) {
        match event {
            CallEvent::Started { .. } => self.0.iter().for_each(|hooks| hooks.on_event(event)),
            _ => self.0.iter().rev().for_each(|hooks| hooks.on_event(event)),
        }
    }
}
