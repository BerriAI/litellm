use litellm_jobs::{HolderId, JobName};

pub trait LeaseNaming: Send + Sync + 'static {
    fn key(&self, job: &JobName) -> String;

    fn owner(&self, holder: &HolderId) -> String;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PythonLeaseNaming;

impl LeaseNaming for PythonLeaseNaming {
    fn key(&self, job: &JobName) -> String {
        format!("cronjob_lock:{}", job.as_str())
    }

    fn owner(&self, holder: &HolderId) -> String {
        format!("\"{}\"", holder.as_str())
    }
}
