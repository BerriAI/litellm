use std::sync::OnceLock;

use litellm_host::{
    machine::{HostFailure, Interrupted, Machine, Step},
    protocol::Protocol,
};
use litellm_tracing::Logger;
use pyo3::Python;

pub(crate) struct LoggedMachine<M> {
    machine: M,
    logger: OnceLock<Logger>,
}

impl<M> LoggedMachine<M> {
    pub(crate) fn new(machine: M) -> Self {
        Self {
            machine,
            logger: OnceLock::new(),
        }
    }
}

impl<M: Machine> Machine for LoggedMachine<M> {
    type Protocol = M::Protocol;
    type Complete = M::Complete;

    fn resume(&mut self) -> Step<'_, Self> {
        let logger = self.logger.get_or_init(|| Python::attach(super::capture));
        Box::pin(logger.instrument(logger.scope(|| self.machine.resume())))
    }

    fn interrupt(
        &mut self,
        failure: HostFailure<<Self::Protocol as Protocol>::Error>,
    ) -> Interrupted<'_, Self> {
        let logger = self.logger.get_or_init(|| Python::attach(super::capture));
        Box::pin(logger.instrument(logger.scope(|| self.machine.interrupt(failure))))
    }
}
