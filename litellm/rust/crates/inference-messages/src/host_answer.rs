use futures_util::future::BoxFuture;

use crate::Error;

pub trait HostAnswer: Send + Sync {
    fn answer(&self, stream: bool) -> BoxFuture<'_, Result<(), Error>>;

    fn intercept(&self, stream: bool) -> BoxFuture<'_, Result<bool, Error>> {
        Box::pin(async move { self.answer(stream).await.map(|()| true) })
    }
}
