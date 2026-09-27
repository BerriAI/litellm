use std::sync::atomic::Ordering;

use super::support::{Harness, context, harness};
use litellm_gateway_mcp::{Error, McpServer, Operation, Operations};
use rmcp::{ServiceExt, model::*};
use rstest::rstest;
use serde_json::{Value, json};

#[rstest]
#[tokio::test]
async fn sdk_client_discovers_and_calls_prefixed_tools(#[future] harness: Harness) {
    let harness = harness.await;
    let (server, client) = tokio::io::duplex(65536);
    let gateway = McpServer::new(harness.gateway.clone());
    let server = tokio::spawn(async move {
        gateway
            .serve(server)
            .await
            .unwrap()
            .waiting()
            .await
            .unwrap()
    });
    let client = ().serve(client).await.unwrap();
    let tools = client.list_all_tools().await.unwrap();
    let names: Vec<_> = tools.iter().map(|tool| tool.name.as_ref()).collect();
    assert_eq!(
        names,
        [
            "alpha-echo-value",
            "alpha-fail",
            "alpha-beta-echo-value",
            "alpha-beta-fail"
        ]
    );
    let arguments = serde_json::from_value(json!({"value": [1, true, "hello"]})).unwrap();
    let mut request = CallToolRequestParams::new("alpha-beta-echo-value").with_arguments(arguments);
    request.meta = Some(serde_json::from_value(json!({"traceparent":"00-0123456789abcdef0123456789abcdef-0123456789abcdef-01", "custom":"preserved"})).unwrap());
    let result = client.call_tool(request).await.unwrap();
    let text = result.content[0].as_text().unwrap();
    let forwarded: Value = serde_json::from_str(&text.text).unwrap();
    assert_eq!(forwarded["name"], "echo-value");
    assert_eq!(forwarded["arguments"], json!({"value": [1, true, "hello"]}));
    assert_eq!(forwarded["meta"]["custom"], "preserved");
    assert_eq!(harness.calls.load(Ordering::SeqCst), 1);
    client.cancel().await.unwrap();
    server.await.unwrap();
}

#[rstest]
#[case::hidden("alpha-hidden", None)]
#[case::unknown("unknown-echo-value", None)]
#[case::bare_ambiguous("echo-value", None)]
#[case::cross_scope("alpha-beta-echo-value", Some("alpha"))]
#[case::unknown_scope("echo-value", Some("missing"))]
#[tokio::test]
async fn denies_calls_outside_visible_catalog(
    #[future] harness: Harness,
    #[case] name: &str,
    #[case] scope: Option<&str>,
) {
    let harness = harness.await;
    let result = harness
        .gateway
        .execute(
            Operation::CallTool(CallToolRequestParams::new(name.to_owned())),
            context(scope),
        )
        .await;
    assert!(matches!(result, Err(Error::Forbidden)));
    assert_eq!(harness.calls.load(Ordering::SeqCst), 0);
}

#[rstest]
#[tokio::test]
async fn preserves_tool_failures_as_tool_results(#[future] harness: Harness) {
    let harness = harness.await;
    let result = harness
        .gateway
        .execute(
            Operation::CallTool(CallToolRequestParams::new("alpha-fail")),
            context(None),
        )
        .await
        .unwrap();
    let ServerResult::CallToolResult(result) = result else {
        panic!("expected tool result")
    };
    assert_eq!(result.is_error, Some(true));
    assert_eq!(
        result.content[0].as_text().unwrap().text,
        "fixture tool failed"
    );
}

#[rstest]
#[tokio::test]
async fn routes_prompts_and_resources_without_rewriting_uris(#[future] harness: Harness) {
    let harness = harness.await;
    let prompts = harness
        .gateway
        .execute(Operation::ListPrompts(None), context(Some("alpha-beta")))
        .await
        .unwrap();
    let ServerResult::ListPromptsResult(prompts) = prompts else {
        panic!("expected prompts")
    };
    assert_eq!(prompts.prompts[0].name, "alpha-beta-review-code");
    let prompt = harness
        .gateway
        .execute(
            Operation::GetPrompt(GetPromptRequestParams::new("alpha-beta-review-code")),
            context(None),
        )
        .await
        .unwrap();
    let ServerResult::GetPromptResult(prompt) = prompt else {
        panic!("expected prompt")
    };
    assert_eq!(
        serde_json::to_value(prompt).unwrap()["messages"][0]["content"]["text"],
        "review-code"
    );
    let resources = harness
        .gateway
        .execute(Operation::ListResources(None), context(Some("alpha")))
        .await
        .unwrap();
    let ServerResult::ListResourcesResult(resources) = resources else {
        panic!("expected resources")
    };
    assert_eq!(resources.resources[0].name, "alpha-document");
    assert_eq!(resources.resources[0].uri, "fixture://document");
    let templates = harness
        .gateway
        .execute(
            Operation::ListResourceTemplates(None),
            context(Some("alpha")),
        )
        .await
        .unwrap();
    let ServerResult::ListResourceTemplatesResult(templates) = templates else {
        panic!("expected templates")
    };
    assert_eq!(templates.resource_templates[0].name, "alpha-files");
    assert_eq!(
        templates.resource_templates[0].uri_template,
        "fixture://{file}"
    );
    let read = Operation::ReadResource(ReadResourceRequestParams::new("fixture://document"));
    assert!(matches!(
        harness.gateway.execute(read.clone(), context(None)).await,
        Err(Error::InvalidRequest(_))
    ));
    let result = harness
        .gateway
        .execute(read, context(Some("alpha")))
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(result).unwrap()["contents"][0]["text"],
        "fixture contents"
    );
}

#[rstest]
#[tokio::test]
async fn retains_healthy_catalogs_and_reports_failed_servers(#[future] harness: Harness) {
    let harness = harness.await;
    let [first, second] = <[_; 2]>::try_from(harness.connections).ok().unwrap();
    second.cancel().await.unwrap();
    let result = harness
        .gateway
        .execute(Operation::ListTools(None), context(None))
        .await
        .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(result["tools"].as_array().unwrap().len(), 2);
    assert_eq!(
        result["_meta"]["litellm.ai/server_outcomes"]["alpha"]["tool_count"],
        2
    );
    assert_eq!(
        result["_meta"]["litellm.ai/server_outcomes"]["alpha-beta"]["status"],
        "unreachable"
    );
    let prompts = harness
        .gateway
        .execute(Operation::ListPrompts(None), context(None))
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(prompts).unwrap()["prompts"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert!(matches!(
        harness
            .gateway
            .rest_tools(context(Some("alpha-beta")))
            .await,
        Err(Error::Upstream(_))
    ));
    first.cancel().await.unwrap();
}
