use crate::{Error, ErrorDetail};

pub(crate) fn validated_region(region: String) -> Result<String, Error> {
    if !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Ok(region);
    }
    Err(Error::InvalidRequest(ErrorDetail::InvalidValue {
        field: "aws_region_name",
        expected: "lowercase letters, digits, and hyphens",
        actual: region.into(),
    }))
}
