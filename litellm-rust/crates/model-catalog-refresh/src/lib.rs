mod error;
mod file;
mod http;
mod policy;
mod revalidate;
mod s3;
mod source;
mod step;

use std::future::Future;
use std::time::Duration;

pub use error::*;
pub use policy::*;
pub use revalidate::*;
pub use s3::*;
pub use source::*;
pub use step::*;

/// The transports a fetch can use. The HTTP client comes from the caller's pool, the S3 client is built per source
pub struct Transports<'a> {
    pub http: &'a litellm_http::Client,
    pub s3: Option<&'a S3Source>,
}

/// One attempt at `source`, never retried here
pub async fn fetch(source: &Source, request: FetchRequest, transports: &Transports<'_>) -> Reply {
    match source {
        Source::Http(url) => http::fetch(transports.http, url, request).await,
        Source::File(path) => file::fetch(path).await,
        Source::S3(object) => match transports.s3 {
            Some(s3) => s3.fetch(request).await,
            None => {
                S3Source::from_environment(object.clone(), transports.http.clone())
                    .await
                    .fetch(request)
                    .await
            }
        },
    }
}

/// Drives `Revalidation` to `Adopt` or `Done`, sleeping between attempts with the injected `sleep`
pub async fn fetch_with_retry<S, F>(
    source: &Source,
    installed: &Installed,
    policy: RetryPolicy,
    seed: Option<u64>,
    transports: &Transports<'_>,
    sleep: S,
) -> Step
where
    S: Fn(Duration) -> F,
    F: Future<Output = ()>,
{
    let (mut revalidation, mut step) = Revalidation::new(source, installed, policy, seed);
    loop {
        step = match step {
            Step::Fetch(request) => revalidation.on_reply(fetch(source, request, transports).await),
            Step::Sleep(delay) => {
                sleep(delay).await;
                revalidation.on_slept()
            }
            done => return done,
        };
    }
}
