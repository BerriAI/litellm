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
