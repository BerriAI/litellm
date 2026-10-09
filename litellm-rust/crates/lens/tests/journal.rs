use litellm_lens::{
    Error,
    journal::{Journal, Turn},
    wire,
};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case::with_initial(true, 0, None)]
#[case::without_initial(false, 0, None)]
#[case::second_turn(true, 1, Some(2))]
#[tokio::test]
async fn excerpts_match_the_serialized_history(
    #[case] include_initial: bool,
    #[case] turn_start: u64,
    #[case] turn_end: Option<u64>,
) {
    let mut journal = Journal::new(&json!({"task": "Read é終🦀 and \"quotes\"\n"}))
        .await
        .unwrap();
    for response in ["first é終🦀", "second \"reply\"\n"] {
        journal
            .push(&Turn {
                response: response.into(),
                tool_results: vec![json!({"value": "é終🦀"}).to_string()],
                validation_error: String::new(),
            })
            .await
            .unwrap();
    }
    let mut request: wire::EvidenceRequest = serde_json::from_value(json!({
        "action": "history", "include_initial": include_initial,
        "turn_start": turn_start, "turn_end": turn_end,
    }))
    .unwrap();
    let full = journal.reply(&request).await.unwrap().to_string();
    request.char_start = 7;
    request.char_end = Some(full.chars().count() as u64 - 9);
    let excerpt = journal.reply(&request).await.unwrap();
    assert_eq!(excerpt["characters"], full.chars().count());
    assert_eq!(
        excerpt["excerpt"],
        full.chars()
            .skip(7)
            .take(full.chars().count() - 16)
            .collect::<String>()
    );
    assert_eq!(excerpt["request"], serde_json::to_value(request).unwrap());
}

#[rstest]
#[case::initial_context(true)]
#[case::archived_turn(false)]
#[tokio::test]
async fn small_unicode_excerpts_are_readable_from_history_over_32_mib(
    #[case] initial_context: bool,
) {
    let content = "é終🦀".repeat(4 * 1024 * 1024);
    let initial = if initial_context {
        json!({"task": content})
    } else {
        json!({"task": "Read archived tools"})
    };
    let mut journal = Journal::new(&initial).await.unwrap();
    if !initial_context {
        journal
            .push(&Turn {
                response: String::new(),
                tool_results: vec![content],
                validation_error: String::new(),
            })
            .await
            .unwrap();
    }
    let mut request: wire::EvidenceRequest = serde_json::from_value(json!({
        "action": "history", "include_initial": initial_context,
    }))
    .unwrap();
    assert!(journal.reply(&request).await.is_err());
    request.char_start = 6 * 1024 * 1024;
    request.char_end = Some(request.char_start + 30);
    let reply = journal.reply(&request).await.unwrap();
    let excerpt = reply["excerpt"].as_str().unwrap();
    assert_eq!(excerpt.chars().count(), 30);
    assert_eq!(excerpt.chars().filter(|ch| *ch == 'é').count(), 10);
    assert_eq!(excerpt.chars().filter(|ch| *ch == '終').count(), 10);
    assert_eq!(excerpt.chars().filter(|ch| *ch == '🦀').count(), 10);
    assert!(reply["characters"].as_u64().unwrap() > 12 * 1024 * 1024);
    request.char_end = None;
    request.char_start = 1;
    assert!(matches!(
        journal.reply(&request).await,
        Err(Error::ToolOutputTooLarge)
    ));
}
