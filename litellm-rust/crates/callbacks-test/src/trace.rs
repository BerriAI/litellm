use litellm_callbacks::event::CallEvent;

/// One op the host answered, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observed {
    Route(String),
    BeforeSend,
    Emit(&'static str),
}

pub fn event_name(event: &CallEvent) -> &'static str {
    match event {
        CallEvent::AttemptStarted { .. } => "attempt_started",
        CallEvent::AttemptFailed { .. } => "attempt_failed",
        CallEvent::ResponseReceived { .. } => "response_received",
        CallEvent::Succeeded { .. } => "succeeded",
        CallEvent::Failed { .. } => "failed",
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trace(pub Vec<Observed>);

impl Trace {
    /// Compact form for assertions: `route:send`, `before_send`, `emit:succeeded`.
    pub fn lines(&self) -> Vec<String> {
        self.0
            .iter()
            .map(|observed| match observed {
                Observed::Route(op) => format!("route:{op}"),
                Observed::BeforeSend => "before_send".into(),
                Observed::Emit(name) => format!("emit:{name}"),
            })
            .collect()
    }

    pub fn route_ops(&self) -> Vec<&str> {
        self.0
            .iter()
            .filter_map(|observed| match observed {
                Observed::Route(op) => Some(op.as_str()),
                _ => None,
            })
            .collect()
    }

    pub fn emitted(&self, name: &str) -> usize {
        self.0
            .iter()
            .filter(|observed| matches!(observed, Observed::Emit(n) if *n == name))
            .count()
    }

    /// Every terminal machine must satisfy this whatever the inner machines did.
    pub fn assert_one_terminal(&self) {
        let terminals = self.emitted("succeeded") + self.emitted("failed");
        assert_eq!(
            terminals,
            1,
            "expected exactly one terminal event in {:?}",
            self.lines()
        );
        let last_emit = self
            .0
            .iter()
            .rposition(|observed| matches!(observed, Observed::Emit("succeeded" | "failed")))
            .unwrap();
        assert!(
            !self.0[last_emit + 1..]
                .iter()
                .any(|o| matches!(o, Observed::Route(_))),
            "route ops after the terminal event in {:?}",
            self.lines()
        );
    }
}

#[macro_export]
macro_rules! assert_trace {
    ($trace:expr, [$($line:expr),* $(,)?]) => {
        assert_eq!($trace.lines(), vec![$($line.to_string()),*]);
    };
}
