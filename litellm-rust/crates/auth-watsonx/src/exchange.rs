use litellm_auth_types::{Error, ErrorDetail, SecretValue};
use litellm_http::CredentialClient;
use serde::Deserialize;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{config::WatsonxIamConfig, error::IamHttpStatus, service::TokenEntry};

pub(crate) const REFRESH_BUFFER: Duration = Duration::from_secs(10);

#[derive(Deserialize)]
struct IamTokenResponse {
    access_token: SecretValue,
    expires_in: u64,
    #[serde(default)]
    expiration: Option<u64>,
}

pub(crate) async fn acquire_token(
    config: &WatsonxIamConfig,
    http: &CredentialClient,
) -> Result<std::sync::Arc<TokenEntry>, Error> {
    let request_start = Instant::now();
    let response = http
        .post(config.endpoint())
        .header("Accept", "application/json")
        .form(&[
            ("grant_type", "urn:ibm:params:oauth:grant-type:apikey"),
            ("apikey", config.api_key().expose()),
        ])
        .send()
        .await
        .map_err(|error| Error::CredentialRequest {
            status: error.status().map(|status| status.as_u16()),
            source: ErrorDetail::failed("Watsonx IAM token request", error.without_url()),
        })?;
    let status = response.status();
    if !status.is_success() {
        return Err(Error::CredentialRequest {
            status: Some(status.as_u16()),
            source: ErrorDetail::failed(
                "Watsonx IAM token request",
                IamHttpStatus {
                    status: status.as_u16(),
                },
            ),
        });
    }
    let body = response
        .bytes()
        .await
        .map_err(|error| Error::CredentialRequest {
            status: error.status().map(|status| status.as_u16()),
            source: ErrorDetail::failed("Watsonx IAM token response", error.without_url()),
        })?;
    let received_at = Instant::now();
    let received_wall = SystemTime::now();
    let response: IamTokenResponse =
        serde_json::from_slice(&body).map_err(|_| invalid_token_response())?;
    let usable_until = usable_deadline(
        request_start,
        received_at,
        received_wall,
        response.expires_in,
        response.expiration,
    )?;
    if response.access_token.expose().trim().is_empty() || usable_until <= received_at {
        return Err(invalid_token_response());
    }
    Ok(std::sync::Arc::new(TokenEntry {
        secret: response.access_token,
        usable_until,
    }))
}

fn usable_deadline(
    request_start: Instant,
    received_at: Instant,
    received_wall: SystemTime,
    expires_in: u64,
    expiration: Option<u64>,
) -> Result<Instant, Error> {
    let lifetime = Duration::from_secs(expires_in)
        .checked_sub(REFRESH_BUFFER)
        .ok_or_else(invalid_token_response)?;
    if lifetime.is_zero() {
        return Err(invalid_token_response());
    }
    let relative = request_start
        .checked_add(lifetime)
        .ok_or_else(invalid_token_response)?;
    let Some(expiration) = expiration else {
        return Ok(relative);
    };
    let absolute = UNIX_EPOCH
        .checked_add(Duration::from_secs(expiration))
        .ok_or_else(invalid_token_response)?;
    let remaining = absolute
        .duration_since(received_wall)
        .map_err(|_| invalid_token_response())?;
    let absolute_lifetime = remaining
        .checked_sub(REFRESH_BUFFER)
        .ok_or_else(invalid_token_response)?;
    if absolute_lifetime.is_zero() {
        return Err(invalid_token_response());
    }
    let absolute_deadline = received_at
        .checked_add(absolute_lifetime)
        .ok_or_else(invalid_token_response)?;
    Ok(relative.min(absolute_deadline))
}

fn invalid_token_response() -> Error {
    Error::InvalidCredentialResponse(ErrorDetail::InvalidType {
        field: "Watsonx IAM token response".into(),
        expected: "a nonempty access_token and usable expiration",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::absolute_deadline_is_minimum(Some(1_100), 93)]
    #[case::relative_deadline_is_minimum(Some(10_000), 3_590)]
    fn chooses_the_earliest_usable_deadline(
        #[case] expiration: Option<u64>,
        #[case] expected_seconds_after_start: u64,
    ) {
        let start = Instant::now();
        let received = start + Duration::from_secs(3);
        let wall = UNIX_EPOCH + Duration::from_secs(1_000);
        let deadline = usable_deadline(start, received, wall, 3_600, expiration).unwrap();
        assert_eq!(
            deadline.duration_since(start),
            Duration::from_secs(expected_seconds_after_start)
        );
    }

    #[rstest]
    #[case::already_expired(999)]
    #[case::inside_refresh_buffer(1_009)]
    #[case::overflow(u64::MAX)]
    fn rejects_an_unusable_absolute_deadline(#[case] expiration: u64) {
        assert!(matches!(
            usable_deadline(
                Instant::now(),
                Instant::now(),
                UNIX_EPOCH + Duration::from_secs(1_000),
                3_600,
                Some(expiration),
            ),
            Err(Error::InvalidCredentialResponse(_))
        ));
    }

    #[rstest]
    #[case::within_buffer(10)]
    #[case::relative_overflow(u64::MAX)]
    fn rejects_an_unusable_relative_deadline(#[case] expires_in: u64) {
        assert!(matches!(
            usable_deadline(Instant::now(), Instant::now(), UNIX_EPOCH, expires_in, None,),
            Err(Error::InvalidCredentialResponse(_))
        ));
    }
}
