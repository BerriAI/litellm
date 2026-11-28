use crate::{AuthHost, Error, Named, authn::key::KeyRecord, host::ProjectRow};

pub(crate) async fn read<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
) -> Result<Option<Named<ProjectRow<H>>>, Error> {
    let Some(project_id) = record.project_id.as_deref() else {
        return Ok(None);
    };
    let project = host.project(project_id).await?;
    Ok(Named::of(Some(project_id.to_owned()), project))
}
