use futures_util::future::BoxFuture;
use litellm_host::{
    machine::HostServices,
    protocol::{Protocol, Reply},
};
use litellm_inference_messages::{Error, HostAnswer};

use crate::cache::CacheCall;

pub(super) enum MessagesHostCall {
    Cache(CacheCall),
    Answer { stream: bool, reply: Reply<()> },
}

impl From<CacheCall> for MessagesHostCall {
    fn from(call: CacheCall) -> Self {
        Self::Cache(call)
    }
}

pub(super) struct PythonAnswer<P: Protocol>(HostServices<P>);

impl<P: Protocol> PythonAnswer<P> {
    pub(super) fn new(services: HostServices<P>) -> Self {
        Self(services)
    }
}

impl<P> HostAnswer for PythonAnswer<P>
where
    P: Protocol<HostCall = MessagesHostCall, Error = Error>,
{
    fn answer(&self, stream: bool) -> BoxFuture<'_, Result<(), Error>> {
        Box::pin(
            self.0
                .call(move |reply| MessagesHostCall::Answer { stream, reply }),
        )
    }
}
