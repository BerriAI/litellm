use super::public::PublicError;

/// `_map_exception_by_status`, the one place a provider status picks a class. Statuses
/// below 400 are not failures the table claims.
pub(super) fn classify(status: u16) -> Option<PublicError> {
    let error = match status {
        ..400 => return None,
        401 => PublicError::Authentication,
        403 => PublicError::PermissionDenied,
        404 => PublicError::NotFound,
        408 | 504 => PublicError::Timeout { status },
        429 => PublicError::RateLimit,
        500 => PublicError::InternalServer,
        502 => PublicError::BadGateway,
        503 => PublicError::ServiceUnavailable,
        400..500 => PublicError::BadRequest,
        _ => PublicError::Api { status },
    };
    Some(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    #[case::below_client_errors(399, None)]
    #[case::lowest_client_error(400, Some(PublicError::BadRequest))]
    #[case::authentication(401, Some(PublicError::Authentication))]
    #[case::permission_denied(403, Some(PublicError::PermissionDenied))]
    #[case::not_found(404, Some(PublicError::NotFound))]
    #[case::request_timeout(408, Some(PublicError::Timeout { status: 408 }))]
    #[case::other_client_error(409, Some(PublicError::BadRequest))]
    #[case::unprocessable(422, Some(PublicError::BadRequest))]
    #[case::rate_limited(429, Some(PublicError::RateLimit))]
    #[case::highest_client_error(499, Some(PublicError::BadRequest))]
    #[case::internal_server(500, Some(PublicError::InternalServer))]
    #[case::other_server_error(501, Some(PublicError::Api { status: 501 }))]
    #[case::bad_gateway(502, Some(PublicError::BadGateway))]
    #[case::service_unavailable(503, Some(PublicError::ServiceUnavailable))]
    #[case::gateway_timeout(504, Some(PublicError::Timeout { status: 504 }))]
    #[case::highest_server_error(599, Some(PublicError::Api { status: 599 }))]
    fn every_mapped_status_and_the_fallback(
        #[case] status: u16,
        #[case] expected: Option<PublicError>,
    ) {
        assert_eq!(classify(status), expected);
    }
}
