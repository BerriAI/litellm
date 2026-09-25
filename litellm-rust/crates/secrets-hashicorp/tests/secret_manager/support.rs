use super::*;

pub(super) fn config(server: &MockServer, values: &[(&str, &str)]) -> HashicorpVaultConfig {
    let mut environment_values: HashMap<String, String> = values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    environment_values.insert("HCP_VAULT_ADDR".to_owned(), server.uri());
    let environment: Arc<dyn Lookup + Send + Sync> =
        Arc::new(move |name: &str| environment_values.get(name).cloned());
    HashicorpVaultConfig::from_environment(environment.as_ref()).unwrap()
}

pub(super) fn manager(server: &MockServer, values: &[(&str, &str)]) -> HashicorpVault {
    HashicorpVault::from_config(config(server, values), true).unwrap()
}

pub(super) fn auth_response(token: &str, lease_duration: u64) -> serde_json::Value {
    json!({
        "auth": {
            "client_token": token,
            "accessor": "",
            "policies": [],
            "token_policies": [],
            "metadata": null,
            "lease_duration": lease_duration,
            "renewable": false,
            "entity_id": "",
            "token_type": "service",
            "orphan": false
        },
        "lease_id": "",
        "lease_duration": lease_duration,
        "renewable": false,
        "request_id": "",
        "warnings": null,
        "wrap_info": null
    })
}

pub(super) fn read_response(data: serde_json::Value) -> serde_json::Value {
    json!({
        "data": {
            "data": data,
            "metadata": {
                "created_time": "",
                "deletion_time": "",
                "custom_metadata": null,
                "destroyed": false,
                "version": 1
            }
        },
        "lease_id": "",
        "lease_duration": 0,
        "renewable": false,
        "request_id": "",
        "warnings": null,
        "wrap_info": null
    })
}

pub(super) fn metadata_response(version: u64) -> serde_json::Value {
    json!({
        "data": {
            "cas_required": true,
            "created_time": "",
            "current_version": version,
            "delete_version_after": "0s",
            "max_versions": 0,
            "oldest_version": 1,
            "updated_time": "",
            "custom_metadata": null,
            "versions": {}
        },
        "lease_id": "",
        "lease_duration": 0,
        "renewable": false,
        "request_id": "",
        "warnings": null,
        "wrap_info": null
    })
}

pub(super) fn write_response(version: u64) -> serde_json::Value {
    json!({
        "data": {
            "created_time": "",
            "deletion_time": "",
            "custom_metadata": null,
            "destroyed": false,
            "version": version
        },
        "lease_id": "",
        "lease_duration": 0,
        "renewable": false,
        "request_id": "",
        "warnings": null,
        "wrap_info": null
    })
}

#[fixture]
pub(super) fn token_values() -> Vec<(&'static str, &'static str)> {
    vec![("HCP_VAULT_TOKEN", "token")]
}

#[derive(Deserialize)]
pub(super) struct ParityCase {
    pub(super) env: HashMap<String, String>,
    pub(super) expected_secret_url: String,
    pub(super) expected_login_url: Option<String>,
    pub(super) expected_login_namespace: Option<String>,
    pub(super) expected_secret_namespace: Option<String>,
    pub(super) secret_name: String,
}

#[fixture]
pub(super) fn parity_cases() -> Vec<ParityCase> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/test_litellm/secret_managers/hashicorp_vault_parity.json"
    )))
    .unwrap()
}

pub(super) const TEST_CERTIFICATE: &str = "-----BEGIN CERTIFICATE-----
MIIDDzCCAfegAwIBAgIUeMzLFLM/mRbPGbNAew5N2UTscocwDQYJKoZIhvcNAQEL
BQAwFzEVMBMGA1UEAwwMbGl0ZWxsbS10ZXN0MB4XDTI2MDkyMTIwMjA1OVoXDTI2
MDkyMjIwMjA1OVowFzEVMBMGA1UEAwwMbGl0ZWxsbS10ZXN0MIIBIjANBgkqhkiG
9w0BAQEFAAOCAQ8AMIIBCgKCAQEAveYoSUJXybmkHmQsBfhBcv2Ob5Oy8ejZu+B3
vTnrPumW4ANi1XXKBSazRGB3fEtAgr+3KhKeHaSKEQeBwJkAEBfdmQv0tpXICwHs
1kFNtU0owy54HVW5/ia+LMszsFcPzVIoMnbUOuiKr9RaV7P+IEFzILPBVuV4DoYH
yocjD3+9QNqokWgNL8LK37JijmNEFVaKFz0X6SyL2VRDlfPWTEBK52Gp/pvDgA6G
eTSfyI+kCm9h5ECTYUAtmatk9WPVS8sWOqV1EXVanFyYBU+mDxoywAS1/6CHeIPh
bNmCOZjPoO9qWBJ7ZyGhOconBigXY8qnlXymev+44IPHrx4urwIDAQABo1MwUTAd
BgNVHQ4EFgQUvaZrZ6HKtbr3ekeZmgy4b5Pq95QwHwYDVR0jBBgwFoAUvaZrZ6HK
tbr3ekeZmgy4b5Pq95QwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOC
AQEAEejrD8d1qDxW55XxQ4IC31rufoEvDV955jyvh2kALPaN/i5oWsBGI+UAQZna
aaoQXwzlmHrtDUBWl0LztVTUamIleUep2+PLLauqqt43vxppxMX8Jn2mnPO20YE/
hIzGx0jN/LBG8PDyLSvHdlgjP9ofA4Vg4rTQugdXRgOvlCE/epnH/MADcg9KYJtJ
C1RObCIkL3LcdUbjStJRCY/U/FeWcgyncEPz95OFDkbrlNDajb6o6CkYfouqvhTc
8XlgjjAVKIbAbRgbVu3elsquuFM97x2DzWDjkrMNmDt1FJ9ubK36gL6B3o0UMaoQ
00R7x/eqvH+EkWa/2ekW9lpleQ==
-----END CERTIFICATE-----
";

pub(super) const TEST_PRIVATE_KEY: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC95ihJQlfJuaQe
ZCwF+EFy/Y5vk7Lx6Nm74He9Oes+6ZbgA2LVdcoFJrNEYHd8S0CCv7cqEp4dpIoR
B4HAmQAQF92ZC/S2lcgLAezWQU21TSjDLngdVbn+Jr4syzOwVw/NUigydtQ66Iqv
1FpXs/4gQXMgs8FW5XgOhgfKhyMPf71A2qiRaA0vwsrfsmKOY0QVVooXPRfpLIvZ
VEOV89ZMQErnYan+m8OADoZ5NJ/Ij6QKb2HkQJNhQC2Zq2T1Y9VLyxY6pXURdVqc
XJgFT6YPGjLABLX/oId4g+Fs2YI5mM+g72pYEntnIaE5yicGKBdjyqeVfKZ6/7jg
g8evHi6vAgMBAAECggEAGdJjlP6b8Fa5bdaCM/ebcrbuuNZVJVbb0JPHxGfNSLs7
pE9hj5QaOdQW2Uviw3h6F61ZCzQH4xD+Iy2po5ZKb2XHYKnDB1bboj+LRGER337T
9aJqe9at2VTMVEv3Rdm40NsEk0QcPLxlK16NQFK90gYEUSSQPDAswJDSG2R/zHn+
vADI907mW/goEJHeLn8PWGlNlSiR6x+5JJtq+GXCzUzVvJYQSCLGxCSl2x2H+0g7
NhFI0zPpdzNmO/h+yhzaFb6Rp5U8+ZsnZ3qYjQ/03gw1myTDKJt1YaO9JvArnNYX
hcJQQ8Rt0bHhcrZA16bBOpqZlo5pKCicwI/netgN8QKBgQDcFz7AzdJ26sMSV32V
rwrMgIoggt8qDjO1ARwqW35A1TIge0FoW4M4KpsXQGGfT341uU1esXEcyZ/1L/5X
3ql2gX4DbOYLZLWYzZGR2hq33oi8HkhN98QrEwL9emSH8NqYX3Xxja3PrmCrSYJe
Zbnd9TIm2XkxyMoyXJu6M/QvnwKBgQDc4dzqTbxoGEGa5MuJoGmMwPnqgdG9UM5J
eExVnh7osxc2sOdsiPeRjjQTxs9v2kJwctC359OJoo9yGaaJeSghU4LEWJo1sqnA
fzSCLammYvtVAtniyNv5Mxk/6Uimi4NNDKaAKB+m4K2uSn3U9AmY7KPYMGaSbS9W
XSnobjxm8QKBgC8bPpAvvWs8ZhIn7bY659nLbUT2HeO3dHO6UBf0yzn/J6JyHxbB
93zvCZDZc8uQTRgcmCW7XtVlhjoJUqvl+Wlm39zF0xr/LCsPXKfWAb/2/lcdOCaP
8Emz4QD10EyUTYUtcWYJB/mafhBLRH8F0Nlj4J8WDu2L51MOJTqeYhZLAoGAWffN
icocAbJPlo22sdoa4+/+W5yBF8GAJMDRJtZ+9H1t6SLpQHYRkMIBSETkXUTjZvX9
Ocs9iIQkNW9pO/mTdO+VBfCo71JUfknR02xR+6m5gYjlws/ZeYlssXGN2/hbhNiw
QOcW7Vv6olFJK6Iy/oz0t6wPO3kpnN3Zogi0paECgYEAwo44M1DdYCtV0snhmYM9
5u0mPfYt5P2SVLXyUbr+vFTfrTL/WKnXIJgbsnj3Gvf+GIZv9tKcXhSNmEHQCYX4
X3w9iTPddCHuvZ1fpufi2TyArJh0OkoNtLXJHTKrHjf2N+61AQzFiv5WieJrdE+H
qr32PTUuVGPyO9LyTY4/RL0=
-----END PRIVATE KEY-----
";
