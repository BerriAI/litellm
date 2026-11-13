use crate::{AuthHost, KeyDenial, authn::key::KeyRecord, host::UserRow};

/// The key owner, refusing a key whose owner SCIM deactivated. An owner that cannot be read is absent.
pub(super) async fn read<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<UserRow<H>>, KeyDenial> {
    let Some(user_id) = record.user_id.as_deref() else {
        return Ok(None);
    };
    match host.key_owner(user_id).await.ok().flatten() {
        Some(owner) if owner.deactivated => Err(KeyDenial::UserDeactivated {
            user_id: user_id.to_owned(),
        }),
        owner => Ok(owner.map(|owner| owner.row)),
    }
}
