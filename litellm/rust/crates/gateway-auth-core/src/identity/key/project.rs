use crate::{AuthHost, Error, authn::key::KeyRecord, host::ProjectRow};

pub(super) async fn read<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<ProjectRow<H>>, Error> {
    match record.project_id.as_deref() {
        None => Ok(None),
        Some(project_id) => host.project(project_id).await,
    }
}
