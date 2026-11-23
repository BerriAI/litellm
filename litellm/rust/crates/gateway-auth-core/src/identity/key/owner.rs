use crate::{AuthHost, Error, KeyDenial, authn::key::KeyRecord, host::UserRow};

/// The key owner, refusing a key whose owner SCIM deactivated. An owner that cannot be read is absent.
pub(super) async fn read<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<UserRow<H>>, Error> {
    let Some(user_id) = record.user_id.as_deref() else {
        return Ok(None);
    };
    match Error::absent_unless_declined(host.key_owner(user_id).await)? {
        Some(owner) if owner.deactivated => Err(KeyDenial::UserDeactivated {
            user_id: user_id.to_owned(),
        }
        .into()),
        owner => Ok(owner.map(|owner| owner.row)),
    }
}
