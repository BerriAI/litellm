use litellm_host_python::{reserve_process_for_forking, run_isolated, runtime_started};

#[test]
fn an_isolated_run_leaves_the_process_free_to_reserve_for_forking() {
    run_isolated(async { tokio::task::spawn_blocking(|| ()).await.unwrap() }).unwrap();

    assert!(!runtime_started());
    assert!(reserve_process_for_forking().is_ok());
}
