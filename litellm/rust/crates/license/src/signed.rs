use base64::Engine;
use base64::alphabet::STANDARD;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use jiff::Zoned;
use jiff::civil::{Date, DateTime, Time};
use jiff::fmt::strtime;
use rsa::RsaPublicKey;
use rsa::pkcs8::DecodePublicKey;
use rsa::pss::Pss;
use rsa::traits::PublicKeyParts;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Error, License};

type Claims = serde_json::Map<String, Value>;

const LENIENT_BASE64: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new()
        .with_decode_padding_mode(DecodePaddingMode::Indifferent)
        .with_decode_allow_trailing_bits(true),
);

pub struct PublicKey(RsaPublicKey);

impl PublicKey {
    pub fn from_pem(pem: &str) -> Result<Self, Error> {
        RsaPublicKey::from_public_key_pem(pem)
            .map(Self)
            .map_err(Error::PublicKey)
    }

    pub fn litellm() -> Self {
        Self::from_pem(include_str!("../public_key.pem"))
            .expect("the bundled license public key is a valid PEM")
    }

    /// Checks `license` against the local wall clock, which is how the proxy has always read `expiration_date`
    pub fn verify_now(&self, license: &str) -> Result<License, Error> {
        self.verify(license, Zoned::now().datetime())
    }

    /// A base64 `payload.signature` license, signed with RSA-PSS over SHA-256 at the maximum salt length, stays
    /// valid until local midnight starting its `expiration_date`
    pub fn verify(&self, license: &str, now: DateTime) -> Result<License, Error> {
        let encoded: String = license
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '='))
            .collect();
        let decoded = LENIENT_BASE64.decode(encoded).map_err(Error::Encoding)?;
        let dot = decoded
            .iter()
            .position(|byte| *byte == b'.')
            .ok_or(Error::Unsigned)?;
        let (message, signature) = (&decoded[..dot], &decoded[dot + 1..]);
        self.0
            .verify(self.max_salt_pss(), &Sha256::digest(message), signature)
            .map_err(Error::Signature)?;
        let claims: Claims = serde_json::from_slice(message).map_err(Error::Payload)?;
        let expiration_date = expiration_date(&claims)?;
        if expiration_date.to_datetime(Time::midnight()) < now {
            return Err(Error::Expired);
        }
        Ok(License::from_claims(expiration_date, &claims))
    }

    fn max_salt_pss(&self) -> Pss {
        let encoded_message_len = (self.0.n().bits() - 1).div_ceil(8);
        Pss::new_with_salt::<Sha256>(
            encoded_message_len.saturating_sub(<Sha256 as Digest>::output_size() + 2),
        )
    }
}

fn expiration_date(claims: &Claims) -> Result<Date, Error> {
    let text = claims
        .get("expiration_date")
        .and_then(Value::as_str)
        .ok_or(Error::ExpirationDate(None))?;
    strtime::parse("%Y-%m-%d", text)
        .and_then(|parsed| parsed.to_date())
        .map_err(|error| Error::ExpirationDate(Some(error)))
}

#[cfg(test)]
mod tests {
    use base64::engine::general_purpose::STANDARD;
    use jiff::civil::date;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::{OTHER_SIGNER, SIGNER, license, public_key, sign, sign_with};

    /// Signed once by Python's `cryptography` with `PSS.MAX_LENGTH`, the way LiteLLM issues licenses, using a
    /// throwaway key whose private half was discarded
    const PYTHON_PUBLIC_KEY: &str = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAkBrrav0fhJV8PCexp13M\nhM/M5YbXUrl6VndP9U9C2nQ9TJC5sZy3f7OQ3H/oCkQXo5r86gk7zgShZtzJW+0E\n2/V4yvYSHdvRtBkbKuBZQT8tGT5bRNPYnD0Zbz8MfPtdZRhU5wg0XP2NqL4ip7kM\nALHpz+z6BuBkMLqFyK8ww++UJl37MmY6WiGLkO1mChe8jvrUOBJfokHrwvW7d0Iq\nMWFcZbBNFhu10X6n0OQGZ/0NCKr2PyroAC836xLd6Ti2lVBq1iXNTmla/u1qUrCK\nNDsxHpoNUYGDm4LP1COkdw3ulI8STDrelbzr/QLfML2ohKM6jKcO8iyanzTEGmqy\nmwIDAQAB\n-----END PUBLIC KEY-----\n";
    const PYTHON_SIGNED_LICENSE: &str = "eyJleHBpcmF0aW9uX2RhdGUiOiAiMjk5OS0wMS0wMSIsICJ1c2VyX2lkIjogInB5dGhvbi1zaWduZWQiLCAiYWxsb3dlZF9mZWF0dXJlcyI6IFsiYXV0b19yb3V0ZXIiXSwgIm1heF91c2VycyI6IDcsICJtYXhfdGVhbXMiOiAzfS4RmJm6SLvfZv0m7UumByEW16ViJg6Zzktviaa8s82LdGtrsi1XQrj6k+dOFG99hoUF9kL1eR2x2w9pzycqkuo1BeoSeEut6cWp7Aq84n6r800n842GozRGONb873ivHXq2KvEEmkdKo9WGUzn/JNL1Pq3xugvccV2cufU0oFaDQO1ZBcbktMS/liSdDwOyQOnEeDBoR3zDV20ZBy+QnptTRwQBdz8MvyxrJFpSOgtdTJtQ5invj8CPl4qdJDwCrUKMIbVprPw0DWKql5CNUvgHPMBv7ckqvlD8VbJPFj9FOPo3pd+rg4i4mKqRT2nuguwzLUYydJCouTwU9cu5znUG";

    fn now() -> DateTime {
        date(2026, 11, 13).at(12, 0, 0, 0)
    }

    #[rstest]
    fn verifies_a_license_python_cryptography_signed() {
        let license = PublicKey::from_pem(PYTHON_PUBLIC_KEY)
            .unwrap()
            .verify(PYTHON_SIGNED_LICENSE, now())
            .unwrap();

        assert_eq!(
            license,
            License {
                expiration_date: date(2999, 1, 1),
                user_id: Some("python-signed".into()),
                allowed_features: vec!["auto_router".into()],
                max_users: Some(7),
                max_teams: Some(3),
            }
        );
    }

    #[rstest]
    fn returns_the_signed_license() {
        let message = json!({
            "expiration_date": "2027-11-12",
            "user_id": "u",
            "allowed_features": ["auto_router"],
            "max_users": 5,
            "max_teams": 2,
        });

        let license = public_key()
            .verify(&sign(message.to_string().as_bytes()), now())
            .unwrap();

        assert_eq!(
            license,
            License {
                expiration_date: date(2027, 11, 12),
                user_id: Some("u".into()),
                allowed_features: vec!["auto_router".into()],
                max_users: Some(5),
                max_teams: Some(2),
            }
        );
    }

    #[rstest]
    #[case::padded(|license: String| license)]
    #[case::unpadded(|license: String| license.trim_end_matches('=').to_owned())]
    #[case::trailing_newline(|license: String| license + "\n")]
    fn accepts_the_encodings_python_decodes(#[case] encode: fn(String) -> String) {
        let encoded = encode(license("2027-11-12"));

        assert!(public_key().verify(&encoded, now()).is_ok());
    }

    #[rstest]
    #[case::day_before(date(2027, 11, 11).at(23, 59, 59, 0), true)]
    #[case::at_midnight(date(2027, 11, 12).at(0, 0, 0, 0), true)]
    #[case::just_after_midnight(date(2027, 11, 12).at(0, 0, 0, 1), false)]
    #[case::day_after(date(2027, 11, 13).at(0, 0, 0, 0), false)]
    fn expires_at_the_start_of_its_expiration_date(#[case] now: DateTime, #[case] valid: bool) {
        let verified = public_key().verify(&license("2027-11-12"), now);

        assert_eq!(verified.is_ok(), valid);
        assert!(valid || matches!(verified, Err(Error::Expired)));
    }

    #[rstest]
    fn rejects_a_license_another_key_signed() {
        let forged = sign_with(&OTHER_SIGNER, br#"{"expiration_date": "2027-11-12"}"#);

        assert!(matches!(
            public_key().verify(&forged, now()),
            Err(Error::Signature(_))
        ));
    }

    #[rstest]
    fn rejects_a_payload_changed_after_signing() {
        let decoded = STANDARD.decode(license("2027-11-12")).unwrap();
        let dot = decoded.iter().position(|byte| *byte == b'.').unwrap();
        let tampered = [
            br#"{"expiration_date":"2999-11-12","max_users":5}"#.as_slice(),
            &decoded[dot..],
        ]
        .concat();

        assert!(matches!(
            public_key().verify(&STANDARD.encode(tampered), now()),
            Err(Error::Signature(_))
        ));
    }

    #[rstest]
    fn rejects_a_signature_with_a_digest_length_salt() {
        let message = br#"{"expiration_date": "2027-11-12"}"#;
        let signature = SIGNER
            .sign_with_rng(
                &mut rand::thread_rng(),
                Pss::new::<Sha256>(),
                &Sha256::digest(message),
            )
            .unwrap();
        let license = STANDARD.encode([message.as_slice(), b".", &signature].concat());

        assert!(matches!(
            public_key().verify(&license, now()),
            Err(Error::Signature(_))
        ));
    }

    #[rstest]
    #[case::not_base64("a")]
    #[case::invalid_length("abcde")]
    fn rejects_text_that_is_not_base64(#[case] license: &str) {
        assert!(matches!(
            public_key().verify(license, now()),
            Err(Error::Encoding(_))
        ));
    }

    #[rstest]
    fn rejects_a_license_without_a_signature() {
        assert!(matches!(
            public_key().verify(&STANDARD.encode(b"no separator"), now()),
            Err(Error::Unsigned)
        ));
    }

    #[rstest]
    #[case::array(br#"["expiration_date", "2027-11-12"]"#)]
    #[case::not_json(b"expiration_date=2027-11-12")]
    fn rejects_a_signed_payload_that_is_not_a_json_object(#[case] message: &[u8]) {
        assert!(matches!(
            public_key().verify(&sign(message), now()),
            Err(Error::Payload(_))
        ));
    }

    #[rstest]
    #[case::missing(json!({"max_users": 5}))]
    #[case::number(json!({"expiration_date": 20271112}))]
    #[case::slashes(json!({"expiration_date": "2027/11/12"}))]
    #[case::datetime(json!({"expiration_date": "2027-11-12T00:00:00"}))]
    #[case::impossible_day(json!({"expiration_date": "2027-02-30"}))]
    fn rejects_an_expiration_date_python_cannot_parse(#[case] message: Value) {
        assert!(matches!(
            public_key().verify(&sign(message.to_string().as_bytes()), now()),
            Err(Error::ExpirationDate(_))
        ));
    }

    #[rstest]
    fn rejects_a_public_key_that_is_not_pem() {
        assert!(matches!(
            PublicKey::from_pem("not a key"),
            Err(Error::PublicKey(_))
        ));
    }

    #[rstest]
    fn bundled_public_key_rejects_a_license_it_did_not_sign() {
        assert!(matches!(
            PublicKey::litellm().verify_now(&license("2999-01-01")),
            Err(Error::Signature(_))
        ));
    }
}
