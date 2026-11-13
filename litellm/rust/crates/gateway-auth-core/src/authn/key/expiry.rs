use jiff::Timestamp;

use crate::{KeyDenial, token::Shown};

use super::{KeyRecord, KeySettings};

pub(super) fn expired(
    key: &KeyRecord,
    now: Timestamp,
    shown: Shown<'_>,
    settings: &KeySettings,
) -> Option<KeyDenial> {
    key.expires
        .filter(|expires| *expires < now)
        .map(|expires| KeyDenial::Expired {
            expires,
            now,
            key: shown.abbreviated(settings.min_custom_key_length),
        })
}

/// What the key row alone settles, before any read about who holds it.
pub(super) fn usable(
    key: &KeyRecord,
    now: Timestamp,
    shown: Shown<'_>,
    settings: &KeySettings,
) -> Result<(), KeyDenial> {
    if key.blocked {
        return Err(KeyDenial::Blocked);
    }
    if let Some(denial) = expired(key, now, shown, settings) {
        return Err(denial);
    }
    if key.token.is_none() {
        return Err(KeyDenial::NoToken);
    }
    Ok(())
}
