use crate::{AuthHost, Error, KeyDenial, Named, authn::key::KeyRecord, host::UserRow};

/// The key owner, refusing a key whose owner SCIM deactivated. An owner that cannot be read is
/// named by its id alone.
pub(super) async fn read<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<Named<UserRow<H>>>, Error> {
    let Some(user_id) = record.user_id.as_deref() else {
        return Ok(None);
    };
    match Error::absent_unless_declined(host.key_owner(user_id).await)? {
        Some(owner) if owner.deactivated => Err(KeyDenial::UserDeactivated {
            user_id: user_id.to_owned(),
        }
        .into()),
        owner => Ok(Named::of(
            Some(user_id.to_owned()),
            owner.map(|owner| owner.row),
        )),
    }
}
