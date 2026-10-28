use std::time::Duration;

use reqwest::header::{ETAG, IF_NONE_MATCH, RETRY_AFTER};
use url::Url;

use crate::step::{FetchRequest, Reply};

pub(crate) async fn fetch(
    client: &litellm_http::Client,
    url: &Url,
    request: FetchRequest,
) -> Reply {
    let builder = client.get(url.clone()).timeout(request.timeout);
    let builder = match &request.if_none_match {
        Some(etag) => builder.header(IF_NONE_MATCH, etag),
        None => builder,
    };
    let response = match builder.send().await {
        Ok(response) => response,
        Err(error) => {
            return Reply::Transport {
                reason: format!("fetching {url} failed: {error}"),
            };
        }
    };
    let status = response.status().as_u16();
    let etags: Vec<&str> = response
        .headers()
        .get_all(ETAG)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .collect();
    let etag = (!etags.is_empty()).then(|| etags.join(", "));
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(retry_after_seconds);
    if status != 200 {
        return Reply::Response {
            status,
            etag,
            retry_after,
            body: None,
        };
    }
    match response.bytes().await {
        Ok(body) => Reply::Response {
            status,
            etag,
            retry_after,
            body: Some(body.to_vec()),
        },
        Err(error) => Reply::Transport {
            reason: format!("reading {url} failed: {error}"),
        },
    }
}

/// Main reads `Retry-After` as a number of seconds and ignores an HTTP date or a negative number
fn retry_after_seconds(header: &str) -> Option<Duration> {
    let seconds = header.trim().parse::<f64>().ok()?;
    (seconds.is_finite() && seconds >= 0.0)
        .then(|| Duration::try_from_secs_f64(seconds).ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::seconds("7", Some(Duration::from_secs(7)))]
    #[case::fraction("0.5", Some(Duration::from_millis(500)))]
    #[case::negative("-1", None)]
    #[case::http_date("Wed, 21 Oct 2015 07:28:00 GMT", None)]
    fn retry_after_is_seconds(#[case] header: &str, #[case] expected: Option<Duration>) {
        assert_eq!(retry_after_seconds(header), expected);
    }
}
