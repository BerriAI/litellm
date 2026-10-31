mod error;
pub mod parallel_slots;
pub mod pod_lock;
pub mod response_polling;
pub mod router_cooldowns;
pub mod spend_counters;

pub use error::Error;

fn as_strs(keys: &[String]) -> Vec<&str> {
    keys.iter().map(String::as_str).collect()
}
