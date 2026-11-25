//! `base64.b64decode(text, validate=True)`.

use ::base64::{
    Engine,
    alphabet::STANDARD,
    engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig},
};

/// Python's validating decoder: canonical padding is required, and the unused bits of the last
/// character may be set.
const VALIDATING: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_padding_mode(DecodePaddingMode::RequireCanonical)
        .with_decode_allow_trailing_bits(true),
);

/// `base64.b64decode(text, validate=True)`, or `None` where it raises `binascii.Error`.
pub fn b64decode_validate(text: &str) -> Option<Vec<u8>> {
    VALIDATING.decode(text).ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::b64decode_validate;

    #[rstest]
    #[case::canonical("YQ==", Some(b"a".as_slice()))]
    #[case::trailing_bits("YR==", Some(b"a".as_slice()))]
    #[case::missing_padding("YQ", None)]
    #[case::whitespace(" YQ==", None)]
    #[case::not_base64("not base64!", None)]
    fn decodes_as_strictly_as_python(#[case] text: &str, #[case] expected: Option<&[u8]>) {
        assert_eq!(b64decode_validate(text).as_deref(), expected);
    }
}
