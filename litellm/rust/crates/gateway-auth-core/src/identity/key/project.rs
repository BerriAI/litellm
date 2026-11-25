use super::ProjectRecord;
use crate::{AuthHost, Error, authn::key::KeyRecord};

pub(super) async fn read<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<ProjectRecord>, Error> {
    match record.project_id.as_deref() {
        None => Ok(None),
        Some(project_id) => host.project(project_id).await,
    }
}
