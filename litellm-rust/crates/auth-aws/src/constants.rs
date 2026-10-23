use litellm_auth_types::AwsKeys;

pub const AWS_ACCESS_KEY_ID: &str = AwsKeys::ACCESS_KEY_ID.env[0];
pub const AWS_SECRET_ACCESS_KEY: &str = AwsKeys::SECRET_ACCESS_KEY.env[0];
pub const AWS_SESSION_TOKEN: &str = AwsKeys::SESSION_TOKEN.env[0];
pub const AWS_REGION_NAME: &str = AwsKeys::REGION.env[0];
pub const AWS_REGION: &str = AwsKeys::REGION.env[1];
pub const AWS_DEFAULT_REGION: &str = "AWS_DEFAULT_REGION";
pub const AWS_BEDROCK_RUNTIME_ENDPOINT: &str = AwsKeys::BEDROCK_RUNTIME_ENDPOINT.env[0];
pub const AWS_SESSION_NAME: &str = AwsKeys::SESSION_NAME.env[0];
pub const AWS_PROFILE_NAME: &str = AwsKeys::PROFILE_NAME.env[0];
pub const AWS_ROLE_NAME: &str = AwsKeys::ROLE_NAME.env[0];
pub const AWS_WEB_IDENTITY_TOKEN: &str = AwsKeys::WEB_IDENTITY_TOKEN.env[0];
pub const AWS_ROLE_ARN: &str = "AWS_ROLE_ARN";
pub const AWS_WEB_IDENTITY_TOKEN_FILE: &str = "AWS_WEB_IDENTITY_TOKEN_FILE";
pub const AWS_STS_ENDPOINT: &str = AwsKeys::STS_ENDPOINT.env[0];
pub const AWS_EXTERNAL_ID: &str = AwsKeys::EXTERNAL_ID.env[0];
pub const AWS_BEARER_TOKEN_BEDROCK: &str = "AWS_BEARER_TOKEN_BEDROCK";
/// Headers SigV4 covers, beyond the `x-amz-` / `x-amzn-` prefixes. Mirrors
/// Python's `_filter_headers_for_aws_signature` allowlist.
pub const AWS_SIGNED_HEADER_NAMES: &[&str] = &[
    "host",
    "content-type",
    "date",
    "x-amz-date",
    "x-amz-security-token",
    "x-amz-content-sha256",
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-signedheaders",
    "x-amz-signature",
];
/// Headers the signer emits itself. Mirrors Python's `SIGV4_COMPUTED_HEADERS`,
/// which the reattach loop skips so a caller's copy cannot ride alongside the
/// computed one.
pub const SIGV4_COMPUTED_HEADER_NAMES: &[&str] = &[
    "authorization",
    "x-amz-date",
    "x-amz-security-token",
    "date",
];
pub const BEDROCK_SERVICE: &str = "bedrock";
pub const DEFAULT_SESSION_NAME_PREFIX: &str = "litellm-session";
pub const DEFAULT_BEDROCK_REGION: &str = "us-west-2";
pub const BEDROCK_RUNTIME_ENDPOINT_TEMPLATE: &str =
    "https://bedrock-runtime.{region}.amazonaws.com";
