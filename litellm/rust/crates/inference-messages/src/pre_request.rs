use futures_util::future::BoxFuture;

use super::{Error, MessagesCall};

pub trait PreRequestHooks: Send + Sync {
    fn run(&self, call: MessagesCall) -> BoxFuture<'_, Result<MessagesCall, Error>>;
}
