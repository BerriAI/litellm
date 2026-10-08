use litellm_lens::{
    config::http_client,
    control::{Control, JobClient},
    evidence::Workspace,
    wire,
};
use rstest::rstest;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{method, path},
};

async fn workspace(text: Arc<Mutex<String>>) -> (MockServer, Workspace, wire::Execution) {
    let server = MockServer::start().await;
    let sample: wire::Sample = serde_json::from_str(include_str!("fixtures/sample.json")).unwrap();
    let execution = sample.executions[0].clone();
    let response_execution = execution.clone();
    Mock::given(method("GET"))
        .and(path("/lens/worker/lens/job/content"))
        .respond_with(move |request: &Request| {
            let offset: usize = request.url.query_pairs().find(|(key, _)| key == "offset").unwrap().1.parse().unwrap();
            assert!(offset >= 1);
            let text = text.lock().unwrap();
            let start = offset - 1;
            ResponseTemplate::new(200).set_body_json(json!({
                "execution":response_execution,
                "parts":[{"execution_id":"run-test","span_id":"span-test","parent_span_id":"root",
                    "name":"tool","kind":"tool","content":text.chars().skip(start).take(8000).collect::<String>(),
                    "truncated":start+8000<text.chars().count(),
                    "start_time":"2026-10-03 10:00:00.200000009","end_time":"2026-10-03 10:00:00.200000019"}]
            }))
        }).mount(&server).await;
    let client = JobClient::new(
        Control::new(
            http_client().unwrap(),
            server.uri().parse().unwrap(),
            "token".into(),
        ),
        "lens",
        "job",
        2,
    )
    .unwrap();
    (server, Workspace::new(sample.executions, client), execution)
}

#[rstest]
#[tokio::test]
async fn reads_search_citations_and_python_preserve_original_unicode_across_pages() {
    let original = format!(
        "{}boundary evidence{}",
        "é".repeat(7995),
        "終".repeat(12000)
    );
    let (_server, workspace, _) = workspace(Arc::new(Mutex::new(original.clone()))).await;
    let read: wire::EvidenceRequest =
        serde_json::from_value(json!({"action":"read","execution_id":"run-test"})).unwrap();
    let reply = workspace.respond(&read).await.unwrap();
    assert_eq!(reply["parts"][0]["content"], original);
    assert_eq!(reply["parts"][0]["parent_span_id"], "root");
    assert_eq!(
        reply["parts"][0]["start_time"],
        "2026-10-03 10:00:00.200000009"
    );
    let search: wire::EvidenceRequest =
        serde_json::from_value(json!({"action":"search","query":"BOUNDARY EVIDENCE"})).unwrap();
    assert_eq!(
        workspace.respond(&search).await.unwrap()["parts"][0]["content"],
        original
    );
    let quote: wire::Evidence = serde_json::from_value(
        json!({"execution_id":"run-test","span_id":"span-test","quote":"boundary evidence"}),
    )
    .unwrap();
    assert!(workspace.valid(&quote).await.unwrap());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input.json");
    let mut file = tokio::fs::File::create(&path).await.unwrap();
    let request: wire::PythonRequest =
        serde_json::from_value(json!({"action":"python","code":"print(data)"})).unwrap();
    workspace.python_input(&request, &mut file).await.unwrap();
    let data: Value = serde_json::from_slice(&tokio::fs::read(path).await.unwrap()).unwrap();
    assert_eq!(data["sessions"][0]["parts"][0]["content"], original);
    assert_eq!(
        data["sessions"][0]["parts"][0]["end_time"],
        "2026-10-03 10:00:00.200000019"
    );
    assert_eq!(data["sessions"][0]["partial"], false);
}

#[rstest]
#[case::first_character(0)]
#[case::within_first_page(3000)]
#[case::end_of_first_page(7999)]
#[case::start_of_second_page(8000)]
#[case::within_second_page(12000)]
#[case::last_character(19999)]
#[tokio::test]
async fn equal_length_edits_on_every_page_invalidate_reuse(#[case] position: usize) {
    let text = Arc::new(Mutex::new("x".repeat(20000)));
    let (_server, workspace, execution) = workspace(text.clone()).await;
    let baseline = workspace.fingerprint(&execution).await.unwrap();
    assert_eq!(workspace.fingerprint(&execution).await.unwrap(), baseline);
    text.lock()
        .unwrap()
        .replace_range(position..position + 1, "y");
    assert_ne!(workspace.fingerprint(&execution).await.unwrap(), baseline);
}

#[rstest]
#[case::joined("startend")]
#[case::omission_marker("start\n[... content omitted ...]\nend")]
#[tokio::test]
async fn citations_cannot_join_across_omitted_content(#[case] quote: &str) {
    let original = format!("{}start\n[... content omitted ...]\nend", "x".repeat(7990));
    let (_server, workspace, _) = workspace(Arc::new(Mutex::new(original))).await;
    let citation: wire::Evidence = serde_json::from_value(
        json!({"execution_id":"run-test","span_id":"span-test","quote":quote}),
    )
    .unwrap();
    assert!(!workspace.valid(&citation).await.unwrap());
}
