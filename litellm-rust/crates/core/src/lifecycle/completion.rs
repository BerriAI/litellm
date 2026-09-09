use serde_json::Value;

use crate::integrations::custom_logger::CallbackTiming;

use super::{
    CallLifecycleContext, Clock, TerminalClassification, TerminalDispatcher, TerminalRecord,
};

pub(crate) struct CompletionOwner<'a> {
    context: CallLifecycleContext,
    start_time: f64,
    dispatcher: &'a dyn TerminalDispatcher,
    clock: &'a dyn Clock,
    finished: bool,
}

impl<'a> CompletionOwner<'a> {
    pub(crate) fn new(
        context: CallLifecycleContext,
        start_time: f64,
        dispatcher: &'a dyn TerminalDispatcher,
        clock: &'a dyn Clock,
    ) -> Self {
        Self {
            context,
            start_time,
            dispatcher,
            clock,
            finished: false,
        }
    }

    pub(crate) fn update(&mut self, context: &CallLifecycleContext) {
        self.context = context.clone();
    }

    pub(crate) fn finish(&mut self, terminal: &TerminalRecord) {
        if !self.finished {
            self.finished = true;
            self.dispatcher.record(terminal);
        }
    }

    pub(crate) fn transfer(&mut self) {
        self.finished = true;
    }
}

impl Drop for CompletionOwner<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let terminal = self.context.terminal(
            CallbackTiming::new(self.start_time, self.clock.now()),
            TerminalClassification::Cancelled {
                message: "call execution was cancelled".into(),
            },
            Value::Null,
        );
        self.finish(&terminal);
    }
}

pub(crate) async fn dispatch<D: TerminalDispatcher + ?Sized>(
    dispatcher: &D,
    terminal: &TerminalRecord,
) {
    if let Err(error) = dispatcher.dispatch(terminal).await {
        tracing::warn!(target: "litellm::lifecycle", call_id = %terminal.call_id,
            attempt = terminal.attempt, error_kind = %error.kind, "terminal dispatch failed");
    }
}

trait CompletionServices: TerminalDispatcher + Clock {}
impl<T: TerminalDispatcher + Clock> CompletionServices for T {}

type Snapshot = Box<dyn Fn(&mut CallLifecycleContext) + Send + Sync>;

pub(crate) struct SessionCompletion {
    services: std::sync::Arc<dyn CompletionServices>,
    context: Option<CallLifecycleContext>,
    start_time: f64,
    snapshot: Snapshot,
    runtime: Option<tokio::runtime::Handle>,
}

impl SessionCompletion {
    pub(crate) fn new<S: TerminalDispatcher + Clock + 'static>(
        services: std::sync::Arc<S>,
        context: CallLifecycleContext,
        start_time: f64,
        snapshot: impl Fn(&mut CallLifecycleContext) + Send + Sync + 'static,
    ) -> Self {
        Self {
            services,
            context: Some(context),
            start_time,
            snapshot: Box::new(snapshot),
            runtime: tokio::runtime::Handle::try_current().ok(),
        }
    }

    pub(crate) async fn settle(
        &mut self,
        classification: TerminalClassification,
    ) -> TerminalRecord {
        let terminal = self.terminal(classification);
        let services = self.services.clone();
        let dispatched = terminal.clone();
        if let Some(runtime) = &self.runtime {
            let task = runtime.spawn(async move {
                dispatch(&*services, &dispatched).await;
            });
            if let Err(error) = task.await {
                tracing::warn!(target: "litellm::lifecycle", call_id = %terminal.call_id,
                    cancelled = error.is_cancelled(), "terminal delivery task stopped");
            }
        } else {
            dispatch(&*services, &dispatched).await;
        }
        terminal
    }

    fn terminal(&mut self, classification: TerminalClassification) -> TerminalRecord {
        let mut context = self
            .context
            .take()
            .expect("session completion consumed once");
        (self.snapshot)(&mut context);
        if classification == TerminalClassification::Success
            && let super::terminal::UsageObservation::Partial(usage) = context.provider_usage
        {
            context.provider_usage = super::terminal::UsageObservation::Final(usage);
        }
        let projection = match &classification {
            TerminalClassification::Success => Value::Null,
            TerminalClassification::Failure { kind, message } => {
                serde_json::json!({"kind": kind, "message": message})
            }
            TerminalClassification::Cancelled { message }
            | TerminalClassification::Incomplete { message } => {
                serde_json::json!({"kind": classification.kind(), "message": message})
            }
        };
        let terminal = context.terminal(
            CallbackTiming::new(self.start_time, self.services.now()),
            classification,
            projection,
        );
        self.services.record(&terminal);
        terminal
    }
}

impl Drop for SessionCompletion {
    fn drop(&mut self) {
        if self.context.is_none() {
            return;
        }
        let message = match self
            .context
            .as_ref()
            .map(|context| context.call_type.as_str())
        {
            Some("realtime") => "realtime session was cancelled before completion",
            _ => "Responses WebSocket session was cancelled before completion",
        };
        let terminal = self.terminal(TerminalClassification::Cancelled {
            message: message.into(),
        });
        if let Some(runtime) = &self.runtime {
            let services = self.services.clone();
            runtime.spawn(async move {
                dispatch(&*services, &terminal).await;
            });
        }
    }
}
