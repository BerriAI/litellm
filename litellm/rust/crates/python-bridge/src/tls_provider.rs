//! The process TLS crypto provider of the FIPS build: rustls-openssl, so handshakes run in the system OpenSSL.

use openssl::rsa::Rsa;
use rustls::{
    crypto::{CryptoProvider, WebPkiSupportedAlgorithms},
    pki_types::{AlgorithmIdentifier, InvalidSignature, SignatureVerificationAlgorithm, alg_id},
};

/// The RSA key sizes, in bits, whose signatures the FIPS build verifies: the FIPS module's 2048-bit floor and ring's
/// 8192-bit ceiling.
const RSA_KEY_BITS: std::ops::RangeInclusive<i32> = 2048..=8192;

/// rustls-openssl's provider, with RSA signatures from keys outside `RSA_KEY_BITS` refused.
/// rustls-openssl leaves the key size to OpenSSL, which outside FIPS mode also verifies with 1024-bit keys.
pub(crate) fn openssl_provider() -> CryptoProvider {
    let provider = rustls_openssl::default_provider();
    let algorithms = provider.signature_verification_algorithms;
    CryptoProvider {
        signature_verification_algorithms: WebPkiSupportedAlgorithms {
            all: leak(algorithms.all.iter().map(|algorithm| sized(*algorithm))),
            mapping: leak(algorithms.mapping.iter().map(|(scheme, algorithms)| {
                (
                    *scheme,
                    leak(algorithms.iter().map(|algorithm| sized(*algorithm))),
                )
            })),
        },
        ..provider
    }
}

/// Built once per process, when the module is imported, so the leak is bounded.
fn leak<T>(items: impl Iterator<Item = T>) -> &'static [T] {
    items.collect::<Vec<_>>().leak()
}

fn sized(
    algorithm: &'static dyn SignatureVerificationAlgorithm,
) -> &'static dyn SignatureVerificationAlgorithm {
    if algorithm.public_key_alg_id() == alg_id::RSA_ENCRYPTION {
        Box::leak(Box::new(RsaKeySize(algorithm)))
    } else {
        algorithm
    }
}

#[derive(Debug)]
struct RsaKeySize(&'static dyn SignatureVerificationAlgorithm);

impl SignatureVerificationAlgorithm for RsaKeySize {
    fn verify_signature(
        &self,
        public_key: &[u8],
        message: &[u8],
        signature: &[u8],
    ) -> Result<(), InvalidSignature> {
        let bits = Rsa::public_key_from_der_pkcs1(public_key)
            .map_err(|_| InvalidSignature)?
            .n()
            .num_bits();
        if !RSA_KEY_BITS.contains(&bits) {
            return Err(InvalidSignature);
        }
        self.0.verify_signature(public_key, message, signature)
    }

    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
        self.0.public_key_alg_id()
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
        self.0.signature_alg_id()
    }

    fn fips(&self) -> bool {
        self.0.fips()
    }
}

#[cfg(test)]
mod tests {
    use openssl::{hash::MessageDigest, pkey::PKey, sign::Signer};
    use rustls::SignatureScheme;

    use super::*;

    fn verifier(scheme: SignatureScheme) -> &'static dyn SignatureVerificationAlgorithm {
        let provider = openssl_provider();
        let (_, algorithms) = provider
            .signature_verification_algorithms
            .mapping
            .iter()
            .find(|(candidate, _)| *candidate == scheme)
            .expect("the provider verifies this scheme");
        algorithms[0]
    }

    #[rstest::rstest]
    #[case::rsa_1024(1024, false)]
    #[case::rsa_2047_under_the_fips_floor(2047, false)]
    #[case::rsa_2048(2048, true)]
    #[case::rsa_4096(4096, true)]
    fn verifies_rsa_signatures_only_from_keys_in_range(#[case] bits: u32, #[case] verified: bool) {
        let rsa = Rsa::generate(bits).expect("OpenSSL generates the RSA key");
        let public_key = rsa
            .public_key_to_der_pkcs1()
            .expect("OpenSSL encodes the public key");
        let key = PKey::from_rsa(rsa).expect("OpenSSL wraps the RSA key");
        let mut signer =
            Signer::new(MessageDigest::sha256(), &key).expect("OpenSSL signs with SHA-256");
        let signature = signer
            .sign_oneshot_to_vec(b"handshake")
            .expect("OpenSSL signs");

        let result = verifier(SignatureScheme::RSA_PKCS1_SHA256).verify_signature(
            &public_key,
            b"handshake",
            &signature,
        );

        assert_eq!(result.is_ok(), verified);
    }
}
