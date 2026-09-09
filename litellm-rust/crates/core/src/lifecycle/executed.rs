use super::TerminalRecord;

#[derive(Debug)]
pub enum ExecutedCall<R, E> {
    Deferred {
        response: R,
        completion: PendingCompletion,
    },
    Success {
        response: R,
        terminal: TerminalRecord,
    },
    Failure {
        error: E,
        terminal: TerminalRecord,
    },
}

impl<R, E> ExecutedCall<R, E> {
    /// Accepts a deferred result. Match `Deferred` to retain control over acceptance.
    pub fn into_result(self) -> Result<R, E> {
        match self {
            Self::Success { response, .. } => Ok(response),
            Self::Deferred {
                response,
                completion,
            } => {
                completion.accept();
                Ok(response)
            }
            Self::Failure { error, .. } => Err(error),
        }
    }

    pub fn terminal(&self) -> &TerminalRecord {
        match self {
            Self::Success { terminal, .. } | Self::Failure { terminal, .. } => terminal,
            Self::Deferred { completion, .. } => completion.record(),
        }
    }
}

pub trait TerminalRecorder: Send + Sync {
    fn record(&self, terminal: &TerminalRecord);
}

pub struct PendingCompletion {
    recorder: std::sync::Arc<dyn TerminalRecorder>,
    terminal: Option<TerminalRecord>,
}

impl std::fmt::Debug for PendingCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PendingCompletion")
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

impl PendingCompletion {
    pub(crate) fn new(
        terminal: TerminalRecord,
        recorder: std::sync::Arc<dyn TerminalRecorder>,
    ) -> Self {
        Self {
            terminal: Some(terminal),
            recorder,
        }
    }

    pub fn record(&self) -> &TerminalRecord {
        self.terminal
            .as_ref()
            .expect("pending completion owns its record")
    }

    pub fn accept(mut self) -> TerminalRecord {
        let terminal = self.terminal.take().expect("completion accepted once");
        self.recorder.record(&terminal);
        terminal
    }

    pub fn reject(mut self, kind: String, message: String) -> TerminalRecord {
        let mut terminal = self.terminal.take().expect("completion rejected once");
        terminal.classification = super::TerminalClassification::Failure { kind, message };
        self.recorder.record(&terminal);
        terminal
    }
}

impl Drop for PendingCompletion {
    fn drop(&mut self) {
        if let Some(mut terminal) = self.terminal.take() {
            terminal.classification = super::TerminalClassification::Cancelled {
                message: "deferred completion was abandoned".into(),
            };
            self.recorder.record(&terminal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_logger::CallbackTiming;
    use crate::integrations::types::Usage;
    use crate::lifecycle::terminal::CostInputs;
    use crate::lifecycle::{RouteProjection, TerminalClassification};
    use serde_json::json;

    #[test]
    fn failure_retains_its_terminal_record() {
        let call = ExecutedCall::<(), _>::Failure {
            error: "provider failed",
            terminal: TerminalRecord {
                call_id: "call-1".to_string(),
                trace_id: None,
                attempt: 1,
                call_type: "ocr".to_string(),
                model: "model".to_string(),
                provider: "provider".to_string(),
                timing: CallbackTiming::new(1.0, 2.0),
                usage: Usage::default(),
                provider_usage: Default::default(),
                cost_inputs: CostInputs::default(),
                classification: TerminalClassification::Failure {
                    kind: "ProviderError".to_string(),
                    message: "provider failed".to_string(),
                },
                projection: RouteProjection::Ocr { value: json!({}) },
            },
        };

        assert_eq!(call.terminal().call_id, "call-1");
        assert_eq!(call.into_result(), Err("provider failed"));
    }
}
