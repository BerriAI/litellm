use std::future::Future;
use std::pin::Pin;

pub mod anthropic;
pub mod auth;
pub mod azure_ai;
#[cfg(feature = "bedrock-auth")]
pub mod bedrock;
pub mod dispatch;
pub mod mistral;
pub mod openai;
pub mod reducto;
pub mod vertex_ai;

pub type AuthorizationFuture<'a> = Pin<
    Box<dyn Future<Output = Result<crate::lifecycle::AuthorizedBody, crate::Error>> + Send + 'a>,
>;
