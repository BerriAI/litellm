//! The stored objects repository: whether the proxy's database holds each kind of object Python
//! loads beside the stored config that changes how it answers a request.

use litellm_db::Call;

use super::queries::{self, StoredObjects};
use crate::Database;

const FIND_STORED_OBJECTS: Call = Call {
    call_type: "find_stored_objects",
    table: None,
    operation: None,
};

/// Reads which stored objects the database holds.
pub struct StoredObjectsRepository {
    database: Database,
}

impl StoredObjectsRepository {
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Which kinds the database holds, read where the stored models are, so a poll sees an object
    /// the moment it sees the models saved beside it.
    pub async fn find(&self) -> Result<StoredObjects, litellm_db::Error> {
        self.database
            .run(FIND_STORED_OBJECTS, || async {
                let mut session = self.database.writer_preferred_session().await?;
                queries::find_stored_objects(session.executor()).await
            })
            .await
    }
}
