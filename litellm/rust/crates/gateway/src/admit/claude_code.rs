//! The model ids `/v1/models` lists for Claude Code (`claude_code_model_id` in
//! `litellm/proxy/common_utils/model_listing_utils.py`), read back as the names they stand for the
//! way Python's auth does on any client's `/v1/messages` (`_normalize_claude_model`).

use litellm_router_core::registry::Registry;

const ENCODED: &str = "claude-router-";
const ONE_MILLION: &str = "[1m]";

/// The name `requested` stands for, its own name without `[1m]` first, then the one its hex
/// spells, when a route owns that name and none owns `requested`. Python matches a wildcard route
/// with the name's provider in front, so while one is configured every id stays as sent.
pub(super) fn named(registry: &Registry, requested: &str) -> Option<String> {
    let unmarked = unmarked(requested);
    let listed = unmarked != requested || requested.starts_with(ENCODED);
    if !listed || registry.has_wildcards() || owned(registry, requested) {
        return None;
    }
    (unmarked != requested && owned(registry, unmarked))
        .then(|| unmarked.to_owned())
        .or_else(|| decoded(unmarked).filter(|name| owned(registry, name)))
}

fn unmarked(name: &str) -> &str {
    name.len()
        .checked_sub(ONE_MILLION.len())
        .filter(|&cut| name.is_char_boundary(cut) && name[cut..].eq_ignore_ascii_case(ONE_MILLION))
        .map_or(name, |cut| &name[..cut])
}

fn decoded(unmarked: &str) -> Option<String> {
    let encoded = unmarked.strip_prefix(ENCODED)?;
    let name = String::from_utf8(hex::decode(encoded).ok()?).ok()?;
    (!name.is_empty() && hex::encode(&name) == encoded).then_some(name)
}

/// `ClaudeCodeRoutingNames`: a group, an alias, a routing group or a deployment id. The key and
/// team aliases and `litellm_settings.model_alias_map` it also reads send a request to Python
/// before its body is read.
fn owned(registry: &Registry, name: &str) -> bool {
    registry.alias_of(name).is_some()
        || registry.host_indexes().has_id(name)
        || registry.model_list_len(name) > 0
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use rstest::rstest;

    use super::*;

    const DEPLOYMENT: &str = "    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n";

    fn named_in(groups: &[&str], requested: &str) -> Option<String> {
        let models: String = groups
            .iter()
            .map(|group| format!("  - model_name: \"{group}\"\n{DEPLOYMENT}"))
            .collect();
        let yaml = format!(
            "model_list:\n{models}router_settings:\n  model_group_alias:\n    sonnet: claude\n"
        );
        let router = crate::boot::unhosted(&Config::from_yaml(&yaml).unwrap());
        named(&router.registry(), requested)
    }

    #[rstest]
    #[case::a_group_in_hex("claude-router-6770742d66617374", Some("gpt-fast"))]
    #[case::a_group_in_hex_marked("claude-router-6770742d66617374[1m]", Some("gpt-fast"))]
    #[case::a_marked_group("claude[1m]", Some("claude"))]
    #[case::a_marked_group_in_capitals("claude[1M]", Some("claude"))]
    #[case::an_alias_in_hex("claude-router-736f6e6e6574", Some("sonnet"))]
    #[case::a_group_named_with_the_mark("big[1m]", None)]
    #[case::a_marked_name_no_route_owns("nope[1m]", None)]
    #[case::a_name_no_route_owns_in_hex("claude-router-6e6f7065", None)]
    #[case::hex_in_capitals("claude-router-6770742D66617374", None)]
    #[case::no_hex("claude-router-gpt-fast", None)]
    #[case::nothing_after_the_prefix("claude-router-", None)]
    #[case::a_plain_group("gpt-fast", None)]
    fn a_listed_id_names_the_route_it_stands_for(
        #[case] requested: &str,
        #[case] stands_for: Option<&str>,
    ) {
        assert_eq!(
            named_in(&["claude", "gpt-fast", "big[1m]"], requested).as_deref(),
            stands_for
        );
    }

    #[rstest]
    #[case::marked("claude[1m]")]
    #[case::in_hex("claude-router-636c61756465")]
    fn no_id_is_read_back_while_a_wildcard_route_is_configured(#[case] requested: &str) {
        assert_eq!(named_in(&["claude", "anthropic/*"], requested), None);
    }
}
