use std::collections::{HashMap, HashSet};

use litellm_llms_types::{
    formats::messages::{
        CacheControl, CacheControlTtl, CacheControlType, ContentBlock, Message, MessageContent,
        MessagesOptionalParams, MessagesRequest, MessagesTool, SystemPrompt,
    },
    integrations::{CacheControlInjectionPoint, CacheControlMessageInjectionPoint},
    message_role::MessageRole,
    recognized::Recognized,
    serde_compat::ParsedInteger,
};
use serde_json::{Map, Value};

const MAX_CACHE_CONTROL_BLOCKS: usize = 4;
const CACHE_CONTROL_KEY: &str = "cache_control";

pub struct InjectionInputs<'a> {
    pub points: &'a [CacheControlInjectionPoint],
    pub enable_prompt_caching: bool,
    pub supports_anthropic_cache_control: bool,
    pub one_shot_subagent: bool,
    pub ttl: Option<CacheControlTtl>,
}

pub struct Injected {
    pub request: MessagesRequest,
    pub added: usize,
    pub remaining_points: Vec<CacheControlInjectionPoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Target<'a> {
    Index(i64),
    Role(&'a str),
    None,
}

pub fn maybe_inject_cache_control(
    request: MessagesRequest,
    inputs: &InjectionInputs<'_>,
) -> Injected {
    if inputs.points.iter().any(|point| match point {
        CacheControlInjectionPoint::Unknown(value) => value.is_object(),
        _ => true,
    }) {
        return apply_to_anthropic_messages_request(request, inputs.points);
    }
    let defaults = get_default_injection_points(&request, inputs);
    apply_to_anthropic_messages_request(request, &defaults)
}

fn get_default_injection_points(
    request: &MessagesRequest,
    inputs: &InjectionInputs<'_>,
) -> Vec<CacheControlInjectionPoint> {
    if !inputs.enable_prompt_caching
        || !inputs.supports_anthropic_cache_control
        || inputs.one_shot_subagent
        || request_has_cache_control(request)
    {
        return Vec::new();
    }
    let control = default_control(inputs.ttl.clone());
    vec![
        CacheControlInjectionPoint::Message(CacheControlMessageInjectionPoint::new(
            None,
            Some(MessageRole::System),
            Some(control.clone()),
        )),
        CacheControlInjectionPoint::Message(CacheControlMessageInjectionPoint::new(
            Some(-1),
            None,
            Some(control),
        )),
    ]
}

fn default_control(ttl: Option<CacheControlTtl>) -> CacheControl {
    CacheControl {
        cache_type: Some(Some(CacheControlType::Ephemeral)),
        ttl: ttl
            .filter(|ttl| matches!(ttl, CacheControlTtl::FiveMinutes | CacheControlTtl::OneHour))
            .map(Some),
        ..CacheControl::default()
    }
}

fn apply_to_anthropic_messages_request(
    request: MessagesRequest,
    injection_points: &[CacheControlInjectionPoint],
) -> Injected {
    if injection_points.is_empty() {
        return Injected {
            request,
            added: 0,
            remaining_points: Vec::new(),
        };
    }
    let (system_points, message_points): (Vec<_>, Vec<_>) = injection_points
        .iter()
        .filter_map(message_point)
        .partition(|point| is_system_point(point));
    let external_breakpoints = count_external_cache_breakpoints(&request);
    let max_blocks = MAX_CACHE_CONTROL_BLOCKS
        .saturating_sub(blocks_reserved_outside_messages(injection_points) + external_breakpoints);
    let used = count_request_cache_breakpoints(&request.messages, request.params.system.as_ref());

    let system_point = system_points.first().filter(|_| {
        used < max_blocks
            && request
                .params
                .system
                .as_ref()
                .and_then(system_marking_index)
                .is_some()
    });
    let system_marked = system_point.is_some();
    let system = match system_point {
        Some(point) => request
            .params
            .system
            .map(|system| system_block_with_breakpoint(system, point.control())),
        None => request.params.system,
    };
    let messages = request
        .messages
        .into_iter()
        .map(with_block_content)
        .collect();
    let (messages, added_to_messages) = apply_message_injections(
        &message_points,
        messages,
        max_blocks.saturating_sub(used + usize::from(system_marked)),
    );
    let added = usize::from(system_marked) + added_to_messages;
    Injected {
        request: MessagesRequest {
            messages,
            params: MessagesOptionalParams {
                system,
                ..request.params
            },
            ..request
        },
        added,
        remaining_points: points_with_a_slot_left(
            injection_points,
            used + added + external_breakpoints,
        ),
    }
}

fn message_point(point: &CacheControlInjectionPoint) -> Option<&CacheControlMessageInjectionPoint> {
    match point {
        CacheControlInjectionPoint::Message(point) => Some(point),
        _ => None,
    }
}

fn is_system_point(point: &CacheControlMessageInjectionPoint) -> bool {
    point.role().map(AsRef::as_ref) == Some(MessageRole::System.as_ref())
}

fn points_with_a_slot_left(
    points: &[CacheControlInjectionPoint],
    breakpoints_on_wire: usize,
) -> Vec<CacheControlInjectionPoint> {
    points
        .iter()
        .filter(|point| match point {
            CacheControlInjectionPoint::Message(_) => false,
            CacheControlInjectionPoint::ToolConfig(_) => {
                breakpoints_on_wire < MAX_CACHE_CONTROL_BLOCKS
            }
            CacheControlInjectionPoint::Unknown(value) => value.is_object(),
        })
        .cloned()
        .collect()
}

fn blocks_reserved_outside_messages(injection_points: &[CacheControlInjectionPoint]) -> usize {
    usize::from(
        injection_points
            .iter()
            .any(|point| matches!(point, CacheControlInjectionPoint::ToolConfig(_))),
    )
}

fn with_block_content(message: Message) -> Message {
    match message.content {
        MessageContent::Text(text) => Message {
            content: MessageContent::Blocks(vec![ContentBlock::text(text)]),
            ..message
        },
        MessageContent::Blocks(_) => message,
    }
}

fn apply_message_injections(
    points: &[&CacheControlMessageInjectionPoint],
    mut messages: Vec<Message>,
    max_blocks: usize,
) -> (Vec<Message>, usize) {
    let roles = role_indices(&messages);
    let mut seen = HashSet::new();
    let mut added = 0;
    for point in points {
        let target = message_target(point);
        if !seen.insert(target) {
            continue;
        }
        for index in resolve_target_indices(target, messages.len(), &roles) {
            if added >= max_blocks {
                return (messages, added);
            }
            if !message_has_cache_control(&messages[index])
                && safe_insert_cache_control_in_message(&mut messages[index], point.control())
            {
                added += 1;
            }
        }
    }
    (messages, added)
}

fn role_indices(messages: &[Message]) -> HashMap<String, Vec<usize>> {
    messages
        .iter()
        .enumerate()
        .fold(HashMap::new(), |mut roles, (index, message)| {
            roles.entry(message.role.clone()).or_default().push(index);
            roles
        })
}

fn message_target(point: &CacheControlMessageInjectionPoint) -> Target<'_> {
    match point.parsed_index() {
        ParsedInteger::Value(index) => Target::Index(index),
        ParsedInteger::OutOfRange => Target::None,
        ParsedInteger::Invalid => point
            .role()
            .map(|role| Target::Role(role.as_ref()))
            .unwrap_or(Target::None),
    }
}

fn resolve_target_indices<'a>(
    target: Target<'_>,
    length: usize,
    roles: &'a HashMap<String, Vec<usize>>,
) -> impl Iterator<Item = usize> + 'a {
    let (index, matching_roles): (Option<usize>, &[usize]) = match target {
        Target::None => (None, &[]),
        Target::Index(index) => {
            let resolved = if index < 0 {
                i64::try_from(length)
                    .unwrap_or(i64::MAX)
                    .saturating_add(index)
            } else {
                index
            };
            (
                usize::try_from(resolved)
                    .ok()
                    .filter(|resolved| *resolved < length),
                &[],
            )
        }
        Target::Role(role) => (None, roles.get(role).map(Vec::as_slice).unwrap_or_default()),
    };
    index.into_iter().chain(matching_roles.iter().copied())
}

fn safe_insert_cache_control_in_message(
    message: &mut Message,
    control: Option<&CacheControl>,
) -> bool {
    let MessageContent::Blocks(blocks) = &mut message.content else {
        return false;
    };
    blocks.last_mut().is_some_and(|block| {
        block.cache_control = Some(injected_control(control));
        true
    })
}

fn system_marking_index(system: &SystemPrompt) -> Option<usize> {
    match system {
        SystemPrompt::Text(_) => Some(0),
        SystemPrompt::Blocks(blocks) if count_cache_control_in_blocks(blocks) > 0 => None,
        SystemPrompt::Blocks(blocks) => blocks.len().checked_sub(1),
    }
}

fn system_block_with_breakpoint(
    system: SystemPrompt,
    control: Option<&CacheControl>,
) -> SystemPrompt {
    let index = system_marking_index(&system);
    let blocks = match system {
        SystemPrompt::Text(text) => vec![ContentBlock::text(text)],
        SystemPrompt::Blocks(blocks) => blocks,
    };
    SystemPrompt::Blocks(
        blocks
            .into_iter()
            .enumerate()
            .map(|(position, block)| {
                if Some(position) == index {
                    ContentBlock {
                        cache_control: Some(injected_control(control)),
                        ..block
                    }
                } else {
                    block
                }
            })
            .collect(),
    )
}

fn injected_control(control: Option<&CacheControl>) -> CacheControl {
    control
        .filter(|control| {
            control.cache_type.is_some()
                || control.ttl.is_some()
                || control.scope.is_some()
                || !control.extra.is_empty()
        })
        .cloned()
        .unwrap_or_else(|| default_control(None))
}

fn request_has_cache_control(request: &MessagesRequest) -> bool {
    count_request_cache_breakpoints(&request.messages, request.params.system.as_ref())
        + count_external_cache_breakpoints(request)
        > 0
}

fn count_request_cache_breakpoints(messages: &[Message], system: Option<&SystemPrompt>) -> usize {
    let system_blocks = match system {
        Some(SystemPrompt::Blocks(blocks)) => count_cache_control_in_blocks(blocks),
        Some(SystemPrompt::Text(_)) | None => 0,
    };
    system_blocks
        + messages
            .iter()
            .map(count_cache_control_blocks)
            .sum::<usize>()
}

fn count_external_cache_breakpoints(request: &MessagesRequest) -> usize {
    let tool_blocks = request
        .params
        .tools
        .iter()
        .flatten()
        .filter(|tool| tool_carries_cache_breakpoint(tool))
        .count();
    usize::from(carries_cache_breakpoint(&request.params.extra)) + tool_blocks
}

fn count_cache_control_blocks(message: &Message) -> usize {
    let content_blocks = match &message.content {
        MessageContent::Blocks(blocks) => count_cache_control_in_blocks(blocks),
        MessageContent::Text(_) => 0,
    };
    usize::from(carries_cache_breakpoint(&message.extra)) + content_blocks
}

fn message_has_cache_control(message: &Message) -> bool {
    count_cache_control_blocks(message) > 0
}

fn count_cache_control_in_blocks(blocks: &[ContentBlock]) -> usize {
    blocks
        .iter()
        .filter(|block| block.cache_control.is_some())
        .count()
}

fn tool_carries_cache_breakpoint(tool: &Recognized<MessagesTool>) -> bool {
    let fields = match tool {
        Recognized::Known(
            MessagesTool::Advisor { extra }
            | MessagesTool::ToolSearchRegex { extra }
            | MessagesTool::ToolSearchBm25 { extra },
        ) => extra,
        Recognized::Unrecognized(Value::Object(fields)) => fields,
        Recognized::Unrecognized(_) => return false,
    };
    carries_cache_breakpoint(fields)
        || fields
            .get("function")
            .and_then(Value::as_object)
            .is_some_and(carries_cache_breakpoint)
}

fn carries_cache_breakpoint(fields: &Map<String, Value>) -> bool {
    fields
        .get(CACHE_CONTROL_KEY)
        .is_some_and(|value| !value.is_null())
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use litellm_llms_types::serde_compat::PYTHON_DEFAULT_INT_MAX_STR_DIGITS;

    fn request(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid Messages request")
    }

    fn configured_points(value: Value) -> Vec<CacheControlInjectionPoint> {
        serde_json::from_value(value).expect("valid cache-control injection points")
    }

    fn configured(points: &[CacheControlInjectionPoint]) -> InjectionInputs<'_> {
        InjectionInputs {
            points,
            enable_prompt_caching: false,
            supports_anthropic_cache_control: false,
            one_shot_subagent: false,
            ttl: None,
        }
    }

    fn defaults(ttl: Option<CacheControlTtl>) -> InjectionInputs<'static> {
        InjectionInputs {
            points: &[],
            enable_prompt_caching: true,
            supports_anthropic_cache_control: true,
            one_shot_subagent: false,
            ttl,
        }
    }

    fn inject(request: MessagesRequest, inputs: &InjectionInputs<'_>) -> (Value, usize) {
        let injected = maybe_inject_cache_control(request, inputs);
        (
            serde_json::to_value(injected.request).unwrap(),
            injected.added,
        )
    }

    #[fixture]
    fn simple_request() -> MessagesRequest {
        request(json!({
            "model": "claude-test",
            "max_tokens": 32,
            "system": "system prompt",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "middle"},
                {"role": "user", "content": "last"}
            ]
        }))
    }

    #[rstest]
    #[case::primitives_only(json!([null, 7, true, "point", []]), 2)]
    #[case::unknown_object(json!([null, {"location": "future"}]), 0)]
    #[case::message_among_primitives(json!([null, {"location": "message", "index": " 0 "}, 7]), 1)]
    fn configured_points_filter_non_objects_before_selecting_defaults(
        simple_request: MessagesRequest,
        #[case] points: Value,
        #[case] expected_added: usize,
    ) {
        let points = configured_points(points);
        let inputs = InjectionInputs {
            points: &points,
            ..defaults(None)
        };

        let (value, added) = inject(simple_request, &inputs);

        assert_eq!(added, expected_added);
        let marked = match expected_added {
            2 => &value["messages"][2]["content"][0],
            1 => &value["messages"][0]["content"][0],
            _ => return,
        };
        assert_eq!(marked["cache_control"], json!({"type": "ephemeral"}));
    }

    #[rstest]
    #[case::disabled(false, true, false, 0)]
    #[case::unsupported(true, false, false, 0)]
    #[case::one_shot(true, true, true, 0)]
    #[case::enabled(true, true, false, 2)]
    fn defaults_require_all_gates(
        simple_request: MessagesRequest,
        #[case] enabled: bool,
        #[case] supported: bool,
        #[case] one_shot: bool,
        #[case] expected_added: usize,
    ) {
        let inputs = InjectionInputs {
            points: &[],
            enable_prompt_caching: enabled,
            supports_anthropic_cache_control: supported,
            one_shot_subagent: one_shot,
            ttl: None,
        };

        let (value, added) = inject(simple_request, &inputs);

        assert_eq!(added, expected_added);
        let expected_mark = (expected_added > 0).then(|| json!({"type": "ephemeral"}));
        assert_eq!(
            value["system"][0].get("cache_control").cloned(),
            expected_mark
        );
        assert_eq!(
            value["messages"][2]["content"][0]
                .get("cache_control")
                .cloned(),
            expected_mark
        );
    }

    #[rstest]
    #[case::non_object_points_fall_back(json!([null, 5, "x"]))]
    fn non_object_configured_points_fall_back_to_defaults(
        simple_request: MessagesRequest,
        #[case] point_values: Value,
    ) {
        let points = configured_points(point_values);
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: true,
            supports_anthropic_cache_control: true,
            one_shot_subagent: false,
            ttl: None,
        };
        let actual = inject(simple_request.clone(), &inputs);
        let expected = inject(simple_request, &defaults(None));

        assert_eq!(actual, expected);
        assert_eq!(actual.1, 2);
    }

    #[rstest]
    fn object_unknown_configured_points_prevent_defaults(simple_request: MessagesRequest) {
        let points = configured_points(json!([{"location": "future"}]));
        let inputs = InjectionInputs {
            points: &points,
            enable_prompt_caching: true,
            supports_anthropic_cache_control: true,
            one_shot_subagent: false,
            ttl: None,
        };
        let (actual, added) = inject(simple_request, &inputs);

        assert_eq!(added, 0);
        assert!(
            actual["messages"][2]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[rstest]
    #[case::message_block(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "marked", "cache_control": {"type": "ephemeral"}}]}]
    }))]
    #[case::message_level(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": "prompt", "cache_control": {"type": "ephemeral"}}]
    }))]
    #[case::system_block(json!({
        "model": "claude-test", "max_tokens": 32,
        "system": [{"type": "text", "text": "system prompt", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "prompt"}]
    }))]
    #[case::top_level(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": "prompt"}],
        "cache_control": {"type": "ephemeral"}
    }))]
    #[case::anthropic_tool(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}]
    }))]
    #[case::function_tool(json!({
        "model": "claude-test", "max_tokens": 32, "system": "system prompt",
        "messages": [{"role": "user", "content": "prompt"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "cache_control": {"type": "ephemeral"}}}]
    }))]
    fn defaults_stand_down_on_client_marks(#[case] request_value: Value) {
        let (value, added) = inject(request(request_value.clone()), &defaults(None));

        assert_eq!(added, 0);
        assert_eq!(value, request_value);
    }

    #[rstest]
    #[case::tool_schema_property(json!({
        "tools": [{"name": "lookup", "input_schema": {
            "type": "object",
            "properties": {"cache_control": {"type": "string"}}
        }}]
    }))]
    #[case::tool_use_input(json!({
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"cache_control": "x"}}
            ]},
            {"role": "user", "content": "last"}
        ]
    }))]
    #[case::metadata(json!({"metadata": {"cache_control": {"type": "ephemeral"}}}))]
    #[case::null_top_level(json!({"cache_control": null}))]
    #[case::null_message_level(json!({
        "messages": [{"role": "user", "content": "last", "cache_control": null}]
    }))]
    fn defaults_ignore_cache_control_outside_breakpoint_positions(#[case] overrides: Value) {
        let mut value = json!({
            "model": "claude-test", "max_tokens": 32, "system": "system prompt",
            "messages": [{"role": "user", "content": "last"}]
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(overrides.as_object().unwrap().clone());

        let (_, added) = inject(request(value), &defaults(None));

        assert_eq!(added, 2);
    }

    #[rstest]
    #[case::five_minutes(CacheControlTtl::FiveMinutes, "5m")]
    #[case::one_hour(CacheControlTtl::OneHour, "1h")]
    fn defaults_apply_ttl(
        simple_request: MessagesRequest,
        #[case] ttl: CacheControlTtl,
        #[case] expected: &str,
    ) {
        let (value, _) = inject(simple_request, &defaults(Some(ttl)));

        assert_eq!(
            value["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": expected})
        );
        assert_eq!(
            value["messages"][2]["content"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": expected})
        );
    }

    #[rstest]
    #[case::unknown_ttl(json!({"type": "ephemeral", "ttl": "future"}), json!({"type": "ephemeral", "ttl": "future"}))]
    #[case::missing_type(json!({"ttl": "1h"}), json!({"ttl": "1h"}))]
    #[case::null_type(json!({"type": null, "ttl": "1h"}), json!({"type": null, "ttl": "1h"}))]
    #[case::null_ttl_and_scope(json!({"type": "ephemeral", "ttl": null, "scope": null}), json!({"type": "ephemeral", "ttl": null, "scope": null}))]
    #[case::extension(json!({"custom": {"enabled": true}}), json!({"custom": {"enabled": true}}))]
    #[case::empty_control(json!({}), json!({"type": "ephemeral"}))]
    #[case::null_control(Value::Null, json!({"type": "ephemeral"}))]
    fn configured_controls_preserve_fields(#[case] control: Value, #[case] expected: Value) {
        let points =
            configured_points(json!([{"location": "message", "index": 0, "control": control}]));
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [{"role": "user", "content": "prompt"}]
        }));

        let (value, added) = inject(request, &configured(&points));

        assert_eq!(added, 1);
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"],
            expected
        );
    }

    #[rstest]
    #[case::unknown_ttl(CacheControlTtl::Other("future".into()))]
    #[case::empty_ttl(CacheControlTtl::Other(String::new()))]
    fn defaults_omit_unknown_ttl(simple_request: MessagesRequest, #[case] ttl: CacheControlTtl) {
        let actual = inject(simple_request.clone(), &defaults(Some(ttl)));
        let expected = inject(simple_request, &defaults(None));

        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::oversized_string(json!("9223372036854775808"), None)]
    #[case::oversized_number(json!(u64::MAX), None)]
    #[case::negative_overflow(json!("-9223372036854775809"), None)]
    #[case::underscores(json!("0_0"), Some(0))]
    #[case::python_digit_limit(json!("0".repeat(PYTHON_DEFAULT_INT_MAX_STR_DIGITS + 1)), Some(1))]
    #[case::python_unicode_digit_limit(json!("٠".repeat(PYTHON_DEFAULT_INT_MAX_STR_DIGITS + 1)), Some(1))]
    #[case::unicode_digits(json!("٠"), Some(0))]
    #[case::invalid_index_uses_role(json!("0__0"), Some(1))]
    #[case::missing_index_uses_role(Value::Null, Some(1))]
    fn configured_indices_follow_python_resolution(
        #[case] index: Value,
        #[case] expected_target: Option<usize>,
    ) {
        let points = configured_points(
            json!([{"location": "message", "index": index, "role": "assistant"}]),
        );
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"}
            ]
        }));

        let (value, added) = inject(request, &configured(&points));

        assert_eq!(added, usize::from(expected_target.is_some()));
        assert_eq!(
            value["messages"][0]["content"][0]
                .get("cache_control")
                .is_some(),
            expected_target == Some(0)
        );
        assert_eq!(
            value["messages"][1]["content"][0]
                .get("cache_control")
                .is_some(),
            expected_target == Some(1)
        );
    }

    #[rstest]
    #[case::with_system(Some("system prompt"))]
    #[case::without_system(None)]
    fn configured_system_role_precedes_index_beside_client_marks(#[case] system: Option<&str>) {
        let points =
            configured_points(json!([{"location": "message", "index": 0, "role": "system"}]));
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32, "system": system,
            "messages": [{"role": "user", "content": "prompt"}],
            "cache_control": {"type": "ephemeral"}
        }));

        let (value, added) = inject(request, &configured(&points));

        assert_eq!(added, usize::from(system.is_some()));
        assert_eq!(
            value["system"][0].get("cache_control").cloned(),
            system.map(|_| json!({"type": "ephemeral"}))
        );
        assert!(
            value["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[rstest]
    #[case::earlier_block(json!([
        {"type": "text", "text": "client marked", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "unmarked tail"}
    ]))]
    #[case::final_block(json!([
        {"type": "text", "text": "unmarked head"},
        {"type": "text", "text": "client marked", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]))]
    fn configured_point_leaves_an_already_marked_message_alone(#[case] content: Value) {
        let points = configured_points(json!([{"location": "message", "index": 0}]));
        let request_value = json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [{"role": "user", "content": content}]
        });

        let (value, added) = inject(request(request_value.clone()), &configured(&points));

        assert_eq!(added, 0);
        assert_eq!(value, request_value);
    }

    #[rstest]
    #[case::client_marks_fill_the_cap(
        json!("system prompt"),
        json!([
            {"role": "user", "content": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "d", "cache_control": {"type": "ephemeral"}}]}
        ])
    )]
    #[case::system_already_marked_on_an_earlier_block(
        json!([
            {"type": "text", "text": "client marked", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": "unmarked tail"}
        ]),
        json!([{"role": "user", "content": [{"type": "text", "text": "prompt"}]}])
    )]
    fn system_point_adds_nothing(#[case] system: Value, #[case] messages: Value) {
        let points = configured_points(json!([{"location": "message", "role": "system"}]));
        let request_value = json!({
            "model": "claude-test", "max_tokens": 32, "system": system, "messages": messages
        });

        let (value, added) = inject(request(request_value.clone()), &configured(&points));

        assert_eq!(added, 0);
        assert_eq!(value, request_value);
    }

    #[rstest]
    #[case::one_past_the_end(1)]
    #[case::far_past_the_end(99)]
    #[case::negative_past_the_start(-2)]
    fn out_of_range_index_marks_nothing(#[case] index: i64) {
        let points = configured_points(json!([{"location": "message", "index": index}]));
        let request_value = json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "only"}]}]
        });

        let (value, added) = inject(request(request_value.clone()), &configured(&points));

        assert_eq!(added, 0);
        assert_eq!(value, request_value);
    }

    #[rstest]
    fn system_point_is_applied_before_message_points_and_tool_config_reserves_a_slot() {
        let points = configured_points(json!([
            {"location": "message", "role": "user"},
            {"location": "message", "role": "system"},
            {"location": "tool_config"}
        ]));
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32, "system": "system prompt",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "user", "content": "third"}
            ]
        }));

        let (value, added) = inject(request, &configured(&points));

        assert_eq!(added, 3);
        assert!(value["system"][0].get("cache_control").is_some());
        assert!(
            value["messages"][0]["content"][0]
                .get("cache_control")
                .is_some()
        );
        assert!(
            value["messages"][1]["content"][0]
                .get("cache_control")
                .is_some()
        );
        assert!(
            value["messages"][2]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[rstest]
    fn configured_message_point_marks_only_the_final_content_block() {
        let points = configured_points(json!([{"location": "message", "index": 0}]));
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "last"}
            ]}]
        }));

        let (value, added) = inject(request, &configured(&points));

        assert_eq!(added, 1);
        assert!(
            value["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            value["messages"][0]["content"][1]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[rstest]
    #[case::no_external_marks(json!({}), 4)]
    #[case::marked_tool(json!({"tools": [{"name": "t", "input_schema": {}, "cache_control": {"type": "ephemeral"}}]}), 3)]
    #[case::automatic_caching(json!({"cache_control": {"type": "ephemeral"}}), 3)]
    fn gateway_injection_observes_the_four_block_cap(
        #[case] overrides: Value,
        #[case] expected: usize,
    ) {
        let points = configured_points(json!([{"location": "message", "role": "user"}]));
        let mut value = json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {"role": "user", "content": "third"},
                {"role": "user", "content": "fourth"},
                {"role": "user", "content": "fifth"}
            ]
        });
        value
            .as_object_mut()
            .unwrap()
            .extend(overrides.as_object().unwrap().clone());

        let (value, added) = inject(request(value), &configured(&points));

        assert_eq!(added, expected);
        assert!(
            value["messages"][expected - 1]["content"][0]
                .get("cache_control")
                .is_some()
        );
        assert!(
            value["messages"][expected]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[rstest]
    fn repeated_targets_keep_the_first_points_control() {
        let points = configured_points(json!([
            {"location": "message", "role": "user", "control": {"type": "ephemeral", "ttl": "1h"}},
            {"location": "message", "role": "user", "control": {"type": "ephemeral", "ttl": "5m"}},
            {"location": "message", "role": "assistant"}
        ]));
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32,
            "messages": [{"role": "user", "content": "only"}]
        }));

        let (value, added) = inject(request, &configured(&points));

        assert_eq!(added, 1);
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
    }

    #[rstest]
    fn all_message_strings_are_normalized_when_points_apply(simple_request: MessagesRequest) {
        let points = configured_points(json!([{"location": "message", "index": 99}]));

        let (value, added) = inject(simple_request, &configured(&points));

        assert_eq!(added, 0);
        assert_eq!(
            value["messages"][0]["content"],
            json!([{"type": "text", "text": "first"}])
        );
        assert_eq!(
            value["messages"][1]["content"],
            json!([{"type": "text", "text": "middle"}])
        );
    }

    #[rstest]
    fn non_message_points_are_never_placed_on_the_request(simple_request: MessagesRequest) {
        let points = configured_points(json!([
            {"location": "future_location", "target": "future", "control": {"x": 1}},
            {"location": "tool_config"}
        ]));

        let injected = maybe_inject_cache_control(simple_request, &configured(&points));
        let value = serde_json::to_value(injected.request).unwrap();

        assert_eq!(injected.added, 0);
        assert_eq!(injected.remaining_points, points);
        assert_eq!(value["system"], "system prompt");
        assert!(value.get("cache_control_injection_points").is_none());
        assert!(value.get("tool_config").is_none());
    }

    #[rstest]
    #[case::slot_available(3, 0, json!([
        {"location": "future", "custom": {"keep": true}},
        {"location": "tool_config", "control": null, "custom": "keep"}
    ]))]
    #[case::cap_full(4, 0, json!([{"location": "future", "custom": {"keep": true}}]))]
    #[case::cap_exceeded(5, 0, json!([{"location": "future", "custom": {"keep": true}}]))]
    #[case::cap_full_from_tools(3, 1, json!([{"location": "future", "custom": {"keep": true}}]))]
    fn pending_points_preserve_extensions_and_only_forward_tools_with_a_slot(
        #[case] marks: usize,
        #[case] tool_marks: usize,
        #[case] expected: Value,
    ) {
        let points = configured_points(json!([
            null,
            {"location": "message", "index": 0},
            {"location": "future", "custom": {"keep": true}},
            {"location": "tool_config", "control": null, "custom": "keep"},
            7
        ]));
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32,
            "tools": vec![json!({
                "name": "lookup", "input_schema": {}, "cache_control": {"type": "ephemeral"}
            }); tool_marks],
            "messages": [{"role": "user", "content": vec![json!({
                "type": "text", "text": "marked", "cache_control": {"type": "ephemeral"}
            }); marks]}]
        }));

        let injected = maybe_inject_cache_control(request, &configured(&points));

        assert_eq!(injected.added, 0);
        assert_eq!(
            serde_json::to_value(injected.remaining_points).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::trailing_billing_block(
        json!([
            {"type": "text", "text": "Instructions"},
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}
        ]),
        1
    )]
    #[case::only_billing_blocks(
        json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}]),
        0
    )]
    #[case::billing_text(json!("x-anthropic-billing-header: cc_version=1"), 0)]
    fn system_breakpoint_matches_python_final_block_placement(
        #[case] system: Value,
        #[case] marked_block: usize,
    ) {
        let points = configured_points(json!([{"location": "message", "role": "system"}]));
        let request = request(json!({
            "model": "claude-test", "max_tokens": 32, "system": system,
            "messages": [{"role": "user", "content": "prompt"}]
        }));

        let (value, added) = inject(request, &configured(&points));

        assert_eq!(added, 1);
        assert_eq!(
            value["system"][marked_block]["cache_control"],
            json!({"type": "ephemeral"})
        );
        let blocks = value["system"].as_array().unwrap();
        assert_eq!(
            blocks
                .iter()
                .filter(|block| block.get("cache_control").is_some())
                .count(),
            1
        );
    }
}
