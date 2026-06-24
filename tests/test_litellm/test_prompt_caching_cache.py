from litellm.router_utils.prompt_caching_cache import PromptCachingCache


def test_prompt_caching_affinity_ttl_respects_one_hour_cache_control():
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Long-lived prompt cache prefix",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        },
        {"role": "user", "content": "This is outside the cacheable prefix"},
    ]

    assert PromptCachingCache.get_prompt_caching_ttl_seconds(messages) == 3600
