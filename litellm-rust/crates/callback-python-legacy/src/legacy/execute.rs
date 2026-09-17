use super::vocabulary::{CallbackInvocation, InvocationOutcome, LoggedMarker};
use pyo3::prelude::*;

pub trait DispatchHost {
    fn prepare_logging(&mut self, py: Python<'_>) -> PyResult<()>;

    fn invoke(
        &mut self,
        py: Python<'_>,
        invocation: CallbackInvocation,
    ) -> PyResult<InvocationOutcome>;

    fn mark_logged(&mut self, py: Python<'_>, marker: LoggedMarker) -> PyResult<()>;
}
