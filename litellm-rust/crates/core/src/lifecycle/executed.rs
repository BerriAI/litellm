use super::TerminalRecord;

#[derive(Clone, Debug)]
pub enum ExecutedCall<R, E> {
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
    pub fn into_result(self) -> Result<R, E> {
        match self {
            Self::Success { response, .. } => Ok(response),
            Self::Failure { error, .. } => Err(error),
        }
    }

    pub fn terminal(&self) -> &TerminalRecord {
        match self {
            Self::Success { terminal, .. } | Self::Failure { terminal, .. } => terminal,
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
