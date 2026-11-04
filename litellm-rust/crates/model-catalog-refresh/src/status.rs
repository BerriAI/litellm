use litellm_error::Report;

use crate::{FetchRequest, Reply};

/// Main's `RETRYABLE_FETCH_STATUS_CODES`
pub const RETRYABLE_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

/// What one attempt came to, by main's `_classify_fetch_response` and `_classify_fetch_error`
#[derive(Clone, Debug, PartialEq)]
pub enum Attempt {
    Fetched {
        body: Vec<u8>,
        etag: Option<String>,
    },
    /// A 304 to a request that sent `If-None-Match`
    NotModified,
    Retryable {
        reason: String,
        retry_after: Option<f64>,
    },
    Failed {
        reason: String,
    },
}

/// A retryable status waits out its `Retry-After`, any other 4xx or 5xx is final, and every other status is a
/// document. A transport failure is retried and a refused request is final
pub fn classify(reply: Reply, location: &str, request: &FetchRequest) -> Attempt {
    match reply {
        Reply::Response { status, .. } if RETRYABLE_STATUSES.contains(&status) => {
            let Reply::Response { retry_after, .. } = reply else {
                unreachable!("matched a response")
            };
            Attempt::Retryable {
                reason: format!("HTTP {status} from {location}"),
                retry_after: retry_after
                    .as_deref()
                    .and_then(litellm_python_compat::number::float),
            }
        }
        Reply::Response { status, .. } if (400..=599).contains(&status) => Attempt::Failed {
            reason: format!("HTTP {status} from {location}"),
        },
        Reply::Response { status: 304, .. } if request.if_none_match.is_some() => {
            Attempt::NotModified
        }
        Reply::Response { body, etag, .. } => Attempt::Fetched { body, etag },
        Reply::Transport { error } => Attempt::Retryable {
            reason: format!("could not fetch {location}: {}", Report::new(&error)),
            retry_after: None,
        },
        Reply::Refused { error } => refused(location, &error),
    }
}

/// The final failure for a request that cannot succeed
pub fn refused(location: &str, error: &crate::Error) -> Attempt {
    Attempt::Failed {
        reason: format!("could not fetch {location}: {}", Report::new(error)),
    }
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::time::Duration;

    use rstest::rstest;

    use super::*;

    const URL: &str = "https://example.com/map.json";

    fn request(if_none_match: Option<&str>) -> FetchRequest {
        FetchRequest {
            if_none_match: if_none_match.map(str::to_owned),
            timeout: Duration::from_secs(5),
        }
    }

    fn response(status: u16, retry_after: Option<&str>) -> Reply {
        Reply::Response {
            status,
            etag: Some("\"v1\"".to_owned()),
            retry_after: retry_after.map(str::to_owned),
            body: b"{}".to_vec(),
        }
    }

    fn retryable(status: u16, retry_after: Option<f64>) -> Attempt {
        Attempt::Retryable {
            reason: format!("HTTP {status} from {URL}"),
            retry_after,
        }
    }

    fn failed(status: u16) -> Attempt {
        Attempt::Failed {
            reason: format!("HTTP {status} from {URL}"),
        }
    }

    fn fetched() -> Attempt {
        Attempt::Fetched {
            body: b"{}".to_vec(),
            etag: Some("\"v1\"".to_owned()),
        }
    }

    /// The same cases as `test_fetch_status_rule_matches_main` in `test_get_model_cost_map.py`, which runs them
    /// against main's `_classify_fetch_response`
    #[rstest]
    #[case::ok(200, None, fetched())]
    #[case::created(201, None, fetched())]
    #[case::redirect_is_a_document(302, None, fetched())]
    #[case::past_599_is_a_document(600, None, fetched())]
    #[case::not_modified_without_an_etag_is_a_document(304, None, fetched())]
    #[case::too_many_requests(429, None, retryable(429, None))]
    #[case::server_error(500, None, retryable(500, None))]
    #[case::bad_gateway(502, None, retryable(502, None))]
    #[case::unavailable(503, None, retryable(503, None))]
    #[case::gateway_timeout(504, None, retryable(504, None))]
    #[case::not_implemented_is_final(501, None, failed(501))]
    #[case::not_found_is_final(404, None, failed(404))]
    #[case::bad_request_is_final(400, None, failed(400))]
    #[case::last_server_error_is_final(599, None, failed(599))]
    #[case::retry_after_seconds(503, Some("4.5"), retryable(503, Some(4.5)))]
    #[case::retry_after_padded(503, Some(" 2 "), retryable(503, Some(2.0)))]
    #[case::retry_after_negative(503, Some("-1"), retryable(503, Some(-1.0)))]
    #[case::retry_after_infinite(503, Some("inf"), retryable(503, Some(f64::INFINITY)))]
    #[case::retry_after_http_date(503, Some("Wed, 21 Oct 2026 07:28:00 GMT"), retryable(503, None))]
    #[case::retry_after_repeated(503, Some("1, 2"), retryable(503, None))]
    #[case::retry_after_ignored_on_a_final_status(404, Some("1"), failed(404))]
    fn status_rule_matches_main(
        #[case] status: u16,
        #[case] retry_after: Option<&str>,
        #[case] expected: Attempt,
    ) {
        assert_eq!(
            classify(response(status, retry_after), URL, &request(None)),
            expected
        );
    }

    #[rstest]
    fn not_modified_needs_a_conditional_request() {
        assert_eq!(
            classify(response(304, None), URL, &request(Some("\"v1\""))),
            Attempt::NotModified
        );
    }

    #[rstest]
    fn transport_failures_retry_and_refusals_are_final() {
        let transport = classify(
            Reply::Transport {
                error: io::Error::from(io::ErrorKind::ConnectionReset).into(),
            },
            URL,
            &request(None),
        );
        assert!(matches!(
            transport,
            Attempt::Retryable {
                retry_after: None,
                ..
            }
        ));
        let refused = classify(
            Reply::Refused {
                error: crate::Error::unsupported_scheme("ftp".to_owned()),
            },
            URL,
            &request(None),
        );
        assert_eq!(
            refused,
            Attempt::Failed {
                reason: format!("could not fetch {URL}: unsupported catalog URL scheme \"ftp\""),
            }
        );
    }
}
