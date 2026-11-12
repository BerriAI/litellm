use futures_util::future::BoxFuture;
use litellm_host::machine::HostServices;
use litellm_inference_messages::{Error, HostAnswer};

use super::host::{MessagesHostCall, MessagesHosted};

pub(super) struct PythonAnswer(HostServices<MessagesHosted>);

impl PythonAnswer {
    pub(super) fn new(services: HostServices<MessagesHosted>) -> Self {
        Self(services)
    }
}

impl HostAnswer for PythonAnswer {
    fn answer(&self, stream: bool) -> BoxFuture<'_, Result<(), Error>> {
        Box::pin(
            self.0
                .call(move |reply| MessagesHostCall::Answer { stream, reply }),
        )
    }

    fn intercept(&self, stream: bool) -> BoxFuture<'_, Result<bool, Error>> {
        Box::pin(
            self.0
                .call(move |reply| MessagesHostCall::Intercept { stream, reply }),
        )
    }
}
