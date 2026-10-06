use std::future::Future;

use axum::{extract::Request, middleware::Next, response::Response};

pub trait GatewayHooks: Send + Sync {
    fn pre_call(&self, request: Request) -> impl Future<Output = Result<Request, Response>> + Send {
        async { Ok(request) }
    }

    fn post_call(&self, response: Response) -> impl Future<Output = Response> + Send {
        async { response }
    }
}

impl GatewayHooks for () {}

pub(crate) async fn run<H: GatewayHooks>(hooks: &H, request: Request, next: Next) -> Response {
    let request = match hooks.pre_call(request).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    hooks.post_call(next.run(request).await).await
}
