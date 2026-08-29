//! Rate limiting via Redis counters with time windows.
//!
//! Checks RPM (requests per minute), TPM (tokens per minute), and
//! max_parallel_requests limits. Uses fixed window approach with TTL.

use litellm_core::auth::KeyObject;
use litellm_core::persistence::{CacheStore, RedisStore};

/// Rate limit check result.
#[derive(Debug)]
pub enum RateLimitResult {
    Allowed,
    RpmExceeded { limit: i64, current: f64 },
    TpmExceeded { limit: i64, current: f64 },
    ParallelExceeded { limit: i64, current: f64 },
}

/// Build a Redis rate-limit key on the stack (no heap allocation).
/// Format: `{api_key:<hashed_token>}:<suffix>`
/// Returns the key as a &str slice into the provided buffer.
fn rate_limit_key<'a>(buf: &'a mut [u8; 128], hashed_token: &str, suffix: &str) -> &'a str {
    let prefix = b"{api_key:";
    let mid = b"}:";
    let mut pos = 0;
    buf[pos..pos + prefix.len()].copy_from_slice(prefix);
    pos += prefix.len();
    let token_bytes = hashed_token.as_bytes();
    buf[pos..pos + token_bytes.len()].copy_from_slice(token_bytes);
    pos += token_bytes.len();
    buf[pos..pos + mid.len()].copy_from_slice(mid);
    pos += mid.len();
    let suffix_bytes = suffix.as_bytes();
    buf[pos..pos + suffix_bytes.len()].copy_from_slice(suffix_bytes);
    pos += suffix_bytes.len();
    unsafe { std::str::from_utf8_unchecked(&buf[..pos]) }
}

/// Write a u64 into a stack buffer, returning the formatted &str.
fn write_u64(buf: &mut [u8; 20], val: u64) -> &str {
    if val == 0 {
        return "0";
    }
    let mut pos = 20;
    let mut v = val;
    while v > 0 {
        pos -= 1;
        buf[pos] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    unsafe { std::str::from_utf8_unchecked(&buf[pos..]) }
}

pub async fn check_request_limits(
    redis: &RedisStore,
    key_object: &KeyObject,
    hashed_token: &str,
) -> RateLimitResult {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let window = now / 60;

    let mut key_buf = [0u8; 128];
    let mut window_buf = [0u8; 20];
    let window_str = write_u64(&mut window_buf, window);

    if let Some(rpm_limit) = key_object.rpm_limit {
        let mut suffix_buf = [0u8; 32];
        suffix_buf[..4].copy_from_slice(b"rpm:");
        suffix_buf[4..4 + window_str.len()].copy_from_slice(window_str.as_bytes());
        let suffix = unsafe { std::str::from_utf8_unchecked(&suffix_buf[..4 + window_str.len()]) };
        let key = rate_limit_key(&mut key_buf, hashed_token, suffix);
        match redis.incr_with_ttl(key, 1.0, 120).await {
            Ok(current) if current > rpm_limit as f64 => {
                return RateLimitResult::RpmExceeded {
                    limit: rpm_limit,
                    current,
                };
            }
            _ => {}
        }
    }

    if let Some(max_parallel) = key_object.max_parallel_requests {
        let key = rate_limit_key(&mut key_buf, hashed_token, "parallel");
        match redis.incr_with_ttl(key, 1.0, 300).await {
            Ok(current) if current > max_parallel as f64 => {
                let _ = redis.incr_by_float(key, -1.0).await;
                return RateLimitResult::ParallelExceeded {
                    limit: max_parallel,
                    current,
                };
            }
            _ => {}
        }
    }

    RateLimitResult::Allowed
}

pub async fn release_parallel_slot(redis: &RedisStore, hashed_token: &str) {
    let mut key_buf = [0u8; 128];
    let key = rate_limit_key(&mut key_buf, hashed_token, "parallel");
    let _ = redis.incr_by_float(key, -1.0).await;
}

pub async fn check_token_limits(
    redis: &RedisStore,
    key_object: &KeyObject,
    hashed_token: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> bool {
    if let Some(tpm_limit) = key_object.tpm_limit {
        let total_tokens = prompt_tokens + completion_tokens;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let window = now / 60;

        let mut key_buf = [0u8; 128];
        let mut window_buf = [0u8; 20];
        let window_str = write_u64(&mut window_buf, window);
        let mut suffix_buf = [0u8; 32];
        suffix_buf[..4].copy_from_slice(b"tpm:");
        suffix_buf[4..4 + window_str.len()].copy_from_slice(window_str.as_bytes());
        let suffix = unsafe { std::str::from_utf8_unchecked(&suffix_buf[..4 + window_str.len()]) };
        let key = rate_limit_key(&mut key_buf, hashed_token, suffix);

        match redis.incr_with_ttl(key, total_tokens as f64, 120).await {
            Ok(current) => current <= tpm_limit as f64,
            Err(_) => true,
        }
    } else {
        true
    }
}
