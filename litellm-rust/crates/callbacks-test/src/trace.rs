use litellm_callbacks::event::CallEvent;
use litellm_callbacks::failure::FailureClass;

/// One thing the host answered or observed, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observed {
    Route(String),
    BeforeSend,
    Emit(&'static str),
    Yield(String),
    AttemptFailed { index: u32, class: FailureClass },
}

pub fn event_name(event: &CallEvent) -> &'static str {
    match event {
        CallEvent::AttemptStarted { .. } => "attempt_started",
        CallEvent::ResponseReceived { .. } => "response_received",
        CallEvent::Succeeded { .. } => "succeeded",
        CallEvent::Failed { .. } => "failed",
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trace(pub Vec<Observed>);

impl Trace {
    /// Compact form for assertions, one vocabulary for every stack: `route:send`,
    /// `before_send`, `emit:succeeded`, `yield:chunk`, `failed:0:RateLimited`.
    pub fn lines(&self) -> Vec<String> {
        self.0
            .iter()
            .map(|observed| match observed {
                Observed::Route(op) => format!("route:{op}"),
                Observed::BeforeSend => "before_send".into(),
                Observed::Emit(name) => format!("emit:{name}"),
                Observed::Yield(chunk) => format!("yield:{chunk}"),
                Observed::AttemptFailed { index, class } => format!("failed:{index}:{class}"),
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

    pub fn attempt_failures(&self) -> usize {
        self.0
            .iter()
            .filter(|observed| matches!(observed, Observed::AttemptFailed { .. }))
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
                .any(|o| matches!(o, Observed::Route(_) | Observed::Yield(_))),
            "route ops or chunks after the terminal event in {:?}",
            self.lines()
        );
    }
}

#[macro_export]
macro_rules! assert_trace {
    ($trace:expr, [$($line:expr),* $(,)?]) => {
        assert_eq!($trace.lines(), vec![$($line.to_string()),*]);
    };
    ($trace:expr, $lines:expr) => {
        assert_eq!(
            $trace.lines(),
            $lines.iter().map(ToString::to_string).collect::<Vec<String>>()
        );
    };
}
