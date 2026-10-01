#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallOutcome {
    Succeeded,
    Failed,
    Cancelled,
}

pub trait CallInterceptors<P: crate::protocol::Protocol>:
    super::interceptors::ProviderInterceptors<P::Error>
{
    fn transform_response(
        &mut self,
        response: P::Response,
    ) -> impl std::future::Future<Output = Result<P::Response, crate::HookError>> + Send {
        std::future::ready(Ok(response))
    }

    fn on_stream_chunk(
        &mut self,
        _chunk: &P::Chunk,
    ) -> impl std::future::Future<Output = Result<(), crate::HookError>> + Send {
        std::future::ready(Ok(()))
    }

    fn on_terminal(
        &mut self,
        _outcome: CallOutcome,
    ) -> impl std::future::Future<Output = Result<(), crate::HookError>> + Send {
        std::future::ready(Ok(()))
    }

    fn on_cancel(&mut self) {}
}

impl<P: crate::protocol::Protocol> CallInterceptors<P> for () {}

/// Pairs a provider-stage interceptor with native call hooks so both observe
/// the call when a driver accepts one call-interceptor value.
impl<P, I, C> CallInterceptors<P> for (I, C)
where
    P: crate::protocol::Protocol,
    I: super::interceptors::ProviderInterceptors<P::Error>,
    C: CallInterceptors<P>,
{
    fn transform_response(
        &mut self,
        response: P::Response,
    ) -> impl std::future::Future<Output = Result<P::Response, crate::HookError>> + Send {
        self.1.transform_response(response)
    }

    fn on_stream_chunk(
        &mut self,
        chunk: &P::Chunk,
    ) -> impl std::future::Future<Output = Result<(), crate::HookError>> + Send {
        self.1.on_stream_chunk(chunk)
    }

    fn on_terminal(
        &mut self,
        outcome: CallOutcome,
    ) -> impl std::future::Future<Output = Result<(), crate::HookError>> + Send {
        self.1.on_terminal(outcome)
    }

    fn on_cancel(&mut self) {
        self.1.on_cancel();
    }
}
