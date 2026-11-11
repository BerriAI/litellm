use futures_util::future::BoxFuture;

use super::{Error, MessagesBody};

pub trait McpGateway: Send + Sync {
    fn serve(&self) -> BoxFuture<'_, Result<MessagesBody, Error>>;
}
