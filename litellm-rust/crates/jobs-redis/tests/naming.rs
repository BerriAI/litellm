use litellm_jobs::{HolderId, JobName};
use litellm_jobs_redis::{LeaseNaming, PythonLeaseNaming};

#[test]
fn python_naming_matches_the_lock_python_pods_take() {
    let holder = HolderId::random();

    assert_eq!(
        PythonLeaseNaming.key(&JobName::new("db_spend_update_job")),
        "cronjob_lock:db_spend_update_job"
    );
    assert_eq!(
        serde_json::from_str::<String>(&PythonLeaseNaming.owner(&holder)).unwrap(),
        holder.as_str()
    );
}
