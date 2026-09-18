use std::collections::VecDeque;

use litellm_callbacks::host::{HostOp, HostResult};
use litellm_callbacks::machine::{HostFailure, Interrupted, Machine, MachineStep, Step};

use crate::route::TestRoute;

/// What one scripted machine does: yield these ops in order, then these chunks, then
/// complete or fail.
#[derive(Clone, Debug)]
pub struct Script<E> {
    pub ops: Vec<String>,
    pub chunks: Vec<String>,
    pub outcome: Result<String, E>,
}

impl<E> Script<E> {
    pub fn ok(ops: &[&str], value: &str) -> Self {
        Self::stream(ops, &[], Ok(value))
    }

    pub fn err(ops: &[&str], error: E) -> Self {
        Self::stream(ops, &[], Err(error))
    }

    /// A streaming attempt: after its ops it yields `chunks`, then completes or fails.
    pub fn stream(ops: &[&str], chunks: &[&str], outcome: Result<&str, E>) -> Self {
        Self {
            ops: ops.iter().map(ToString::to_string).collect(),
            chunks: chunks.iter().map(ToString::to_string).collect(),
            outcome: outcome.map(String::from),
        }
    }
}

/// Plays a [`Script`]. Records the answers the host gave so a test can assert the host's
/// results reached the machine in order.
pub struct Scripted<E> {
    ops: VecDeque<String>,
    chunks: VecDeque<String>,
    outcome: Option<Result<String, E>>,
    pub answers: Vec<String>,
}

impl<E> Scripted<E> {
    pub fn new(script: Script<E>) -> Self {
        Self {
            ops: script.ops.into(),
            chunks: script.chunks.into(),
            outcome: Some(script.outcome),
            answers: Vec::new(),
        }
    }
}

impl<E: Clone + Send + Sync + 'static> Machine for Scripted<E> {
    type Route = TestRoute<E>;
    type Complete = String;

    fn resume(&mut self, result: Option<HostResult<TestRoute<E>>>) -> Step<'_, Self> {
        Box::pin(async move {
            if let Some(HostResult::Route(answer)) = result {
                self.answers.push(answer);
            }
            if let Some(op) = self.ops.pop_front() {
                return Ok(MachineStep::Host(HostOp::Route(op)));
            }
            if let Some(chunk) = self.chunks.pop_front() {
                return Ok(MachineStep::Yield(chunk));
            }
            self.outcome
                .take()
                .expect("scripted machine resumed after completion")
                .map(MachineStep::Complete)
        })
    }

    fn interrupt(&mut self, failure: HostFailure<E>) -> Interrupted<'_, Self> {
        Box::pin(async move { Err(failure.into_error()) })
    }
}
