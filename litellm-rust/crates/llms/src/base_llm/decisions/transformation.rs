use std::borrow::Cow;

use litellm_llms_types::formats::decisions::{
    ChoiceProbability, DecisionAnswer, DecisionChoice, DecisionLevel, DecisionQuestion,
    DecisionsContentPart, DecisionsInput, DecisionsInputMessage, DecisionsMessageContent,
    DecisionsRequest, DecisionsResponse, DecisionsUsage, ScoreProbability,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::{Error, Invalid};
use crate::{ErrorDetail, ErrorSource};

pub const MAX_QUESTIONS: usize = 128;
const CHOICES: (usize, usize) = (2, 255);
const LEVELS: (usize, usize) = (2, 10);
const DEFAULT_PREDICATE: &str = "Is this true of the input?";

pub fn parse_request<'de, D>(body: D) -> Result<DecisionsRequest, Error>
where
    D: serde::Deserializer<'de>,
    D::Error: Send + Sync + 'static,
{
    let request = DecisionsRequest::deserialize(body)
        .map_err(|error| Error::InvalidRequest(Invalid::Body(ErrorSource::new(error))))?;
    check(&request).map_err(Error::InvalidRequest)?;
    Ok(request)
}

fn check(request: &DecisionsRequest) -> Result<(), Invalid> {
    let count = request.questions.len();
    if !(1..=MAX_QUESTIONS).contains(&count) {
        return Err(Invalid::QuestionCount { count });
    }
    request
        .questions
        .iter()
        .enumerate()
        .try_for_each(|(index, question)| match question {
            DecisionQuestion::Choice { choices, .. } => {
                check_count(index, "choices", CHOICES, choices.len())?;
                match distinct_keys(choices) {
                    true => Ok(()),
                    false => Err(Invalid::DuplicateChoices(index)),
                }
            }
            DecisionQuestion::Score { levels, .. } => {
                check_count(index, "levels", LEVELS, levels.len())
            }
            DecisionQuestion::Predicate { .. } | DecisionQuestion::Box { .. } => Ok(()),
        })
}

fn check_count(
    index: usize,
    field: &'static str,
    (min, max): (usize, usize),
    count: usize,
) -> Result<(), Invalid> {
    match (min..=max).contains(&count) {
        true => Ok(()),
        false => Err(Invalid::OptionCount {
            index,
            field,
            min,
            max,
            count,
        }),
    }
}

fn distinct_keys(choices: &[DecisionChoice]) -> bool {
    choices.iter().enumerate().all(|(index, choice)| {
        choices[..index]
            .iter()
            .all(|earlier| earlier.value.key() != choice.value.key())
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    pub provider: &'static str,
    pub model: &'a str,
    pub requested_model: &'a str,
    pub supports_box_questions: bool,
    pub drop_params: bool,
}

#[derive(Debug, Serialize)]
pub struct ProviderBody<'a> {
    pub model: &'a str,
    pub input: Cow<'a, DecisionsInput>,
    pub questions: Vec<Cow<'a, DecisionQuestion>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<&'a str>,
}

#[derive(Debug)]
pub struct ProviderRequest<'a> {
    pub body: ProviderBody<'a>,
    pub sent: Vec<usize>,
}

pub fn provider_request<'a>(
    request: &'a DecisionsRequest,
    target: Target<'a>,
) -> Result<ProviderRequest<'a>, Error> {
    if !target.drop_params && request.images.as_ref().is_some_and(truthy) {
        return Err(Error::ImagesNextToInput);
    }
    let safety_identifier = match &request.safety_identifier {
        None | Some(Value::Null) => None,
        Some(Value::String(identifier)) => Some(identifier.as_str()),
        Some(_) if target.drop_params => None,
        Some(_) => return Err(Error::InvalidRequest(Invalid::SafetyIdentifier)),
    };
    let input = well_formed_input(&request.input, target.drop_params)?;
    let refuses_box = !target.supports_box_questions
        && request
            .questions
            .iter()
            .any(|question| matches!(question, DecisionQuestion::Box { .. }));
    let media = |name: &'static str, clips: &Option<Value>| {
        (!target.drop_params && clips.as_ref().is_some_and(truthy)).then_some(name)
    };
    let unsupported: Vec<&'static str> = [
        refuses_box.then_some("box questions"),
        media("audio", &request.audio),
        media("videos", &request.videos),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !unsupported.is_empty() && !target.drop_params {
        return Err(Error::UnsupportedParams {
            provider: target.provider,
            params: unsupported,
            model: target.requested_model.to_owned(),
        });
    }
    let sent: Vec<usize> = request
        .questions
        .iter()
        .enumerate()
        .filter(|(_, question)| !(refuses_box && matches!(question, DecisionQuestion::Box { .. })))
        .map(|(index, _)| index)
        .collect();
    if sent.is_empty() {
        return Err(Error::NoQuestionsLeft {
            provider: target.provider,
            params: unsupported,
            model: target.requested_model.to_owned(),
        });
    }
    Ok(ProviderRequest {
        body: ProviderBody {
            model: target.model,
            input,
            questions: sent
                .iter()
                .map(|index| provider_question(&request.questions[*index]))
                .collect(),
            safety_identifier,
        },
        sent,
    })
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

fn provider_question(question: &DecisionQuestion) -> Cow<'_, DecisionQuestion> {
    match question {
        DecisionQuestion::Predicate { name, instructions } if instructions.is_empty() => {
            Cow::Owned(DecisionQuestion::Predicate {
                name: name.clone(),
                instructions: DEFAULT_PREDICATE.to_owned(),
            })
        }
        other => Cow::Borrowed(other),
    }
}

fn well_formed_input(
    input: &DecisionsInput,
    drop_params: bool,
) -> Result<Cow<'_, DecisionsInput>, Error> {
    let DecisionsInput::Messages(messages) = input else {
        return Ok(Cow::Borrowed(input));
    };
    let malformed = messages
        .iter()
        .flat_map(image_urls)
        .filter(|url| !is_well_formed_image(url))
        .count();
    if malformed == 0 {
        return Ok(Cow::Borrowed(input));
    }
    let kept: Vec<DecisionsInputMessage> = messages
        .iter()
        .filter_map(without_malformed_images)
        .collect();
    match drop_params && !kept.is_empty() {
        true => Ok(Cow::Owned(DecisionsInput::Messages(kept))),
        false => Err(Error::InvalidRequest(Invalid::MalformedImages(malformed))),
    }
}

fn image_urls(message: &DecisionsInputMessage) -> impl Iterator<Item = &str> {
    let parts = match &message.content {
        DecisionsMessageContent::Parts(parts) => parts.as_slice(),
        DecisionsMessageContent::Text(_) => &[],
    };
    parts.iter().filter_map(|part| match part {
        DecisionsContentPart::InputImage { image_url, .. } => Some(image_url.as_str()),
        DecisionsContentPart::InputText { .. } => None,
    })
}

fn without_malformed_images(message: &DecisionsInputMessage) -> Option<DecisionsInputMessage> {
    let DecisionsMessageContent::Parts(parts) = &message.content else {
        return Some(message.clone());
    };
    let kept: Vec<DecisionsContentPart> = parts
        .iter()
        .filter(|part| match part {
            DecisionsContentPart::InputImage { image_url, .. } => is_well_formed_image(image_url),
            DecisionsContentPart::InputText { .. } => true,
        })
        .cloned()
        .collect();
    (!kept.is_empty()).then(|| DecisionsInputMessage {
        content: DecisionsMessageContent::Parts(kept),
        ..message.clone()
    })
}

fn strip_prefix_ignore_case<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.get(..prefix.len())
        .filter(|head| head.eq_ignore_ascii_case(prefix))
        .map(|_| &text[prefix.len()..])
}

fn is_well_formed_image(url: &str) -> bool {
    if let Some(rest) = strip_prefix_ignore_case(url, "https://")
        .or_else(|| strip_prefix_ignore_case(url, "http://"))
    {
        return !rest.is_empty() && !rest.chars().any(char::is_whitespace);
    }
    let Some(rest) = strip_prefix_ignore_case(url, "data:image/") else {
        return false;
    };
    let Some((header, payload)) = rest.split_once(',') else {
        return false;
    };
    let segments: Vec<&str> = header.split(';').collect();
    let header_ok = segments.len() >= 2
        && segments
            .last()
            .is_some_and(|last| last.eq_ignore_ascii_case("base64"))
        && segments
            .iter()
            .all(|segment| !segment.is_empty() && !segment.chars().any(char::is_whitespace));
    let data = payload.trim_end_matches('=');
    header_ok
        && payload.len() - data.len() <= 2
        && !data.is_empty()
        && data
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'_' | b'-'))
}

pub fn caller_response(
    payload: &[u8],
    request: &DecisionsRequest,
    sent: &[usize],
    requested_model: &str,
    provider: &'static str,
) -> Result<DecisionsResponse, Error> {
    let invalid = |detail: ErrorDetail| Error::InvalidResponse { provider, detail };
    let upstream: DecisionsResponse = serde_json::from_slice(payload)
        .map_err(|error| invalid(ErrorDetail::invalid("Decisions response", error)))?;
    let mut upstream_answers = upstream.answers.into_iter();
    let answered: Vec<DecisionAnswer> = sent
        .iter()
        .map(|index| answer(&request.questions[*index], upstream_answers.next()))
        .collect();
    if answered.iter().any(lacks_confidence) {
        return Err(invalid("a choice or score answer has no confidence".into()));
    }
    let mut answered = answered.into_iter();
    let mut sent = sent.iter().peekable();
    let answers = request
        .questions
        .iter()
        .enumerate()
        .map(
            |(index, question)| match sent.next_if(|sent| **sent == index) {
                Some(_) => answered.next().unwrap_or_else(|| refusal(question)),
                None => refusal(question),
            },
        )
        .collect();
    let DecisionsUsage {
        input_tokens,
        output_tokens,
        ..
    } = upstream.usage;
    Ok(DecisionsResponse {
        model: match upstream.model.is_empty() {
            true => requested_model.to_owned(),
            false => upstream.model,
        },
        answers,
        usage: DecisionsUsage {
            total_tokens: input_tokens + output_tokens,
            ..upstream.usage
        },
    })
}

fn lacks_confidence(answer: &DecisionAnswer) -> bool {
    matches!(
        answer,
        DecisionAnswer::Choice {
            confidence: None,
            ..
        } | DecisionAnswer::Score {
            confidence: None,
            ..
        }
    )
}

fn refusal(question: &DecisionQuestion) -> DecisionAnswer {
    DecisionAnswer::Refusal {
        name: question.name().map(str::to_owned),
    }
}

fn answer(question: &DecisionQuestion, answer: Option<DecisionAnswer>) -> DecisionAnswer {
    let name = || question.name().map(str::to_owned);
    match (question, answer) {
        (
            DecisionQuestion::Predicate { .. },
            Some(DecisionAnswer::Predicate { probability, .. }),
        ) => DecisionAnswer::Predicate {
            name: name(),
            probability,
        },
        (
            DecisionQuestion::Choice { choices, .. },
            Some(DecisionAnswer::Choice {
                choice,
                probabilities,
                confidence,
                ..
            }),
        ) => DecisionAnswer::Choice {
            name: name(),
            choice,
            probabilities: choice_probabilities(choices, &probabilities),
            confidence,
        },
        (
            DecisionQuestion::Score { levels, .. },
            Some(DecisionAnswer::Score {
                score,
                probabilities,
                confidence,
                ..
            }),
        ) => DecisionAnswer::Score {
            name: name(),
            score,
            probabilities: score_probabilities(levels, &probabilities),
            confidence,
        },
        (
            DecisionQuestion::Box { .. },
            Some(DecisionAnswer::Box {
                region, confidence, ..
            }),
        ) => DecisionAnswer::Box {
            name: name(),
            region,
            confidence,
        },
        _ => refusal(question),
    }
}

fn choice_probabilities(
    choices: &[DecisionChoice],
    probabilities: &[ChoiceProbability],
) -> Vec<ChoiceProbability> {
    choices
        .iter()
        .filter_map(|choice| {
            let found = probabilities
                .iter()
                .rev()
                .find(|item| item.value.key() == choice.value.key())?;
            Some(ChoiceProbability {
                value: choice.value.clone(),
                probability: found.probability,
            })
        })
        .collect()
}

fn score_probabilities(
    levels: &[DecisionLevel],
    probabilities: &[ScoreProbability],
) -> Vec<ScoreProbability> {
    levels
        .iter()
        .zip(0_i64..)
        .filter_map(|(level, index)| {
            let found = probabilities
                .iter()
                .rev()
                .find(|item| item.value == index)?;
            Some(ScoreProbability {
                value: index,
                label: level.label.clone(),
                probability: found.probability,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    fn request(body: Value) -> DecisionsRequest {
        parse_request(&body).unwrap()
    }

    fn target(drop_params: bool, supports_box_questions: bool) -> Target<'static> {
        Target {
            provider: "openai",
            model: "gpt-6-luna",
            requested_model: "openai/gpt-6-luna",
            supports_box_questions,
            drop_params,
        }
    }

    #[fixture]
    fn mixed() -> DecisionsRequest {
        request(json!({
            "input": "Customer wrote: refund order 4411 today",
            "questions": [
                {"type": "predicate", "name": "refund", "instructions": "Refund?"},
                {"type": "box", "name": "mug", "instructions": "the mug"},
                {"type": "choice", "name": "urgency", "instructions": "How urgent?",
                 "choices": [{"value": "low"}, {"value": "high", "description": "today"}]},
                {"type": "score", "name": "mood", "instructions": "Mood?",
                 "levels": [{"label": "calm"}, {"label": "annoyed"}, {"label": "angry"}]},
            ],
        }))
    }

    #[rstest]
    #[case::no_questions(json!({"input": "x", "questions": []}), "questions should have 1 to 128 items, got 0")]
    #[case::one_choice(json!({"input": "x", "questions": [{"type": "choice", "instructions": "q", "choices": [{"value": "a"}]}]}), "questions[0].choices should have 2 to 255 items, got 1")]
    #[case::eleven_levels(json!({"input": "x", "questions": [{"type": "score", "instructions": "q", "levels": (0..11).map(|n| json!({"label": n.to_string()})).collect::<Vec<_>>()}]}), "questions[0].levels should have 2 to 10 items, got 11")]
    #[case::bool_shares_text(json!({"input": "x", "questions": [{"type": "predicate", "instructions": "q"}, {"type": "choice", "instructions": "q", "choices": [{"value": true}, {"value": "true"}]}]}), "questions[1]: Choice values must be unique")]
    fn requests_breaking_the_question_rules_are_refused(
        #[case] body: Value,
        #[case] message: &str,
    ) {
        let error = parse_request(&body).unwrap_err().to_string();

        assert!(
            error.starts_with(&format!("Invalid Decisions request: {message}")),
            "{error}"
        );
    }

    #[rstest]
    fn too_many_questions_are_refused() {
        let questions: Vec<Value> = (0..=MAX_QUESTIONS)
            .map(|_| json!({"type": "predicate", "instructions": "q"}))
            .collect();

        assert!(matches!(
            parse_request(&json!({"input": "x", "questions": questions})),
            Err(Error::InvalidRequest(Invalid::QuestionCount { count: 129 }))
        ));
    }

    #[rstest]
    fn the_provider_body_carries_only_the_decision_fields(mixed: DecisionsRequest) {
        let mut body = serde_json::to_value(&mixed).unwrap();
        body["safety_identifier"] = json!("user-1");
        let with_identifier = request(body);

        let prepared = provider_request(&with_identifier, target(false, true)).unwrap();

        assert_eq!(prepared.sent, [0, 1, 2, 3]);
        assert_eq!(
            serde_json::to_value(&prepared.body).unwrap(),
            json!({
                "model": "gpt-6-luna",
                "input": "Customer wrote: refund order 4411 today",
                "questions": serde_json::to_value(&mixed.questions).unwrap(),
                "safety_identifier": "user-1",
            })
        );
    }

    #[rstest]
    fn an_empty_predicate_gets_the_default_question() {
        let request = request(
            json!({"input": "x", "questions": [{"type": "predicate", "instructions": ""}]}),
        );

        let prepared = provider_request(&request, target(false, false)).unwrap();

        assert_eq!(
            serde_json::to_value(&prepared.body.questions).unwrap(),
            json!([{"type": "predicate", "instructions": "Is this true of the input?"}])
        );
    }

    #[rstest]
    fn box_questions_are_refused_where_the_model_cannot_answer_them(mixed: DecisionsRequest) {
        let error = provider_request(&mixed, target(false, false)).unwrap_err();

        assert_eq!(
            error,
            Error::UnsupportedParams {
                provider: "openai",
                params: vec!["box questions"],
                model: "openai/gpt-6-luna".into(),
            }
        );
    }

    #[rstest]
    fn dropped_box_questions_are_left_out_and_refused_in_the_answers(mixed: DecisionsRequest) {
        let prepared = provider_request(&mixed, target(true, false)).unwrap();
        assert_eq!(prepared.sent, [0, 2, 3]);
        let upstream = json!({
            "model": "gpt-6-luna-2026",
            "answers": [
                {"type": "predicate", "name": "ignored", "probability": 0.9},
                {"type": "choice", "name": null, "choice": "high", "confidence": 0.8,
                 "probabilities": [{"value": "high", "probability": 0.8}, {"value": "low", "probability": 0.2}]},
                {"type": "score", "name": null, "score": 1.2, "confidence": 0.6,
                 "probabilities": [{"value": 2, "label": "x", "probability": 0.3}, {"value": 7, "label": "y", "probability": 0.1}, {"value": 0, "label": "z", "probability": 0.6}]},
            ],
            "usage": {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 100}, "output_tokens": 20, "total_tokens": 1},
        });

        let response = caller_response(
            &serde_json::to_vec(&upstream).unwrap(),
            &mixed,
            &prepared.sent,
            "openai/gpt-6-luna",
            "openai",
        )
        .unwrap();

        assert_eq!(
            serde_json::to_value(&response).unwrap(),
            json!({
                "model": "gpt-6-luna-2026",
                "answers": [
                    {"type": "predicate", "name": "refund", "probability": 0.9},
                    {"type": "refusal", "name": "mug"},
                    {"type": "choice", "name": "urgency", "choice": "high",
                     "probabilities": [{"value": "low", "probability": 0.2}, {"value": "high", "probability": 0.8}], "confidence": 0.8},
                    {"type": "score", "name": "mood", "score": 1.2,
                     "probabilities": [{"value": 0, "label": "calm", "probability": 0.6}, {"value": 2, "label": "angry", "probability": 0.3}], "confidence": 0.6},
                ],
                "usage": {"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 0},
                          "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2020},
            })
        );
    }

    #[rstest]
    fn only_box_questions_dropped_leave_nothing_to_ask() {
        let request =
            request(json!({"input": "x", "questions": [{"type": "box", "instructions": "mug"}]}));

        assert!(matches!(
            provider_request(&request, target(true, false)),
            Err(Error::NoQuestionsLeft { .. })
        ));
    }

    #[rstest]
    #[case::refused(false, true)]
    #[case::dropped(true, false)]
    fn media_is_refused_unless_dropped(#[case] drop_params: bool, #[case] refused: bool) {
        let request = request(
            json!({"input": "x", "questions": [{"type": "predicate", "instructions": "q"}], "audio": ["data:audio/wav;base64,AA=="]}),
        );

        let result = provider_request(&request, target(drop_params, false));

        assert_eq!(
            matches!(result, Err(Error::UnsupportedParams { ref params, .. }) if params == &["audio"]),
            refused
        );
    }

    #[rstest]
    #[case::kept(false, true)]
    #[case::dropped(true, false)]
    fn images_beside_input_belong_to_the_other_format(
        #[case] drop_params: bool,
        #[case] refused: bool,
    ) {
        let request = request(
            json!({"input": "x", "questions": [{"type": "predicate", "instructions": "q"}], "images": ["https://x/a.png"]}),
        );

        assert_eq!(
            matches!(
                provider_request(&request, target(drop_params, false)),
                Err(Error::ImagesNextToInput)
            ),
            refused
        );
    }

    #[rstest]
    #[case::string(json!("u"), false, Some(Some("u")))]
    #[case::null(Value::Null, false, Some(None))]
    #[case::number(json!(3), false, None)]
    #[case::number_dropped(json!(3), true, Some(None))]
    fn the_safety_identifier_must_be_a_string(
        #[case] identifier: Value,
        #[case] drop_params: bool,
        #[case] sent: Option<Option<&str>>,
    ) {
        let request = request(
            json!({"input": "x", "questions": [{"type": "predicate", "instructions": "q"}], "safety_identifier": identifier}),
        );

        let prepared = provider_request(&request, target(drop_params, false));

        assert_eq!(
            prepared
                .map(|prepared| prepared.body.safety_identifier)
                .ok(),
            sent
        );
    }

    #[rstest]
    #[case::https("https://example.com/a.png", true)]
    #[case::upper_scheme("HTTP://example.com/a.png", true)]
    #[case::data("data:image/png;base64,iVBORw0KGgo=", true)]
    #[case::data_with_params("data:image/png;name=a;base64,AAAA", true)]
    #[case::url_safe_payload("data:image/png;base64,ab-_", true)]
    #[case::no_base64("data:image/png,AAAA", false)]
    #[case::three_pads("data:image/png;base64,AA===", false)]
    #[case::bad_payload("data:image/png;base64,a b", false)]
    #[case::not_an_image("data:text/plain;base64,AAAA", false)]
    #[case::bare_path("/tmp/a.png", false)]
    #[case::scheme_only("https://", false)]
    fn image_urls_are_data_urls_or_links(#[case] url: &str, #[case] well_formed: bool) {
        assert_eq!(is_well_formed_image(url), well_formed);
    }

    #[rstest]
    #[case::refused(false, Some(1))]
    #[case::dropped(true, None)]
    fn malformed_images_are_refused_unless_dropped(
        #[case] drop_params: bool,
        #[case] refused: Option<usize>,
    ) {
        let request = request(json!({"input": [{"content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "file:///etc/passwd"},
            {"type": "input_image", "image_url": "https://x/a.png"},
        ]}], "questions": [{"type": "predicate", "instructions": "q"}]}));

        let result = provider_request(&request, target(drop_params, false));

        match refused {
            Some(count) => assert_eq!(
                result.unwrap_err(),
                Error::InvalidRequest(Invalid::MalformedImages(count))
            ),
            None => assert_eq!(
                serde_json::to_value(&result.unwrap().body.input).unwrap(),
                json!([{"role": "user", "type": "message", "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "https://x/a.png"},
                ]}])
            ),
        }
    }

    #[rstest]
    fn a_message_of_only_malformed_images_cannot_be_dropped_to_nothing() {
        let request = request(
            json!({"input": [{"content": [{"type": "input_image", "image_url": "nope"}]}], "questions": [{"type": "predicate", "instructions": "q"}]}),
        );

        assert_eq!(
            provider_request(&request, target(true, false)).unwrap_err(),
            Error::InvalidRequest(Invalid::MalformedImages(1))
        );
    }

    fn respond(request: &DecisionsRequest, answers: Value) -> Result<DecisionsResponse, Error> {
        let upstream = json!({"model": "", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}});
        let sent: Vec<usize> = (0..request.questions.len()).collect();
        caller_response(
            &serde_json::to_vec(&upstream).unwrap(),
            request,
            &sent,
            "openai/gpt-6-luna",
            "openai",
        )
    }

    #[rstest]
    fn answers_of_the_wrong_type_or_missing_become_refusals(mixed: DecisionsRequest) {
        let response = respond(
            &mixed,
            json!([{"type": "choice", "name": "x", "choice": "a", "probabilities": [], "confidence": 1.0}]),
        )
        .unwrap();

        assert_eq!(response.model, "openai/gpt-6-luna");
        assert_eq!(
            serde_json::to_value(&response.answers).unwrap(),
            json!([
                {"type": "refusal", "name": "refund"},
                {"type": "refusal", "name": "mug"},
                {"type": "refusal", "name": "urgency"},
                {"type": "refusal", "name": "mood"},
            ])
        );
    }

    #[rstest]
    #[case::choice(json!([{"type": "choice", "name": null, "choice": "low", "probabilities": []}]), true)]
    #[case::unknown_answer_type(json!([{"type": "maybe", "name": null}]), true)]
    #[case::refusal(json!([{"type": "refusal", "name": null}]), false)]
    fn a_response_python_rejects_is_an_invalid_response(
        #[case] answers: Value,
        #[case] rejected: bool,
    ) {
        let request = request(
            json!({"input": "x", "questions": [{"type": "choice", "instructions": "q", "choices": [{"value": "low"}, {"value": "high"}]}]}),
        );

        let result = respond(&request, answers);

        assert_eq!(
            matches!(
                result,
                Err(Error::InvalidResponse {
                    provider: "openai",
                    ..
                })
            ),
            rejected
        );
    }
}
