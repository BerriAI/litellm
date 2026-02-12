# Claude Code - Prompt Cache Routing

Claude's [Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) feature helps to optimize API usage through attempting to cache prompts and re-use cached prompts during subsequent API calls. This feature is used by Claude Code. 

When LiteLLM [load balancing](../proxy/load_balancing.md) is enabled, to ensure this prompt caching feature still works with Claude Code, LiteLLM needs to be configured to use the `PromptCachingDeploymentCheck` pre-call check. This pre-call check will ensure that API calls that used prompt caching are remembered and that subsequent API calls that try to use that prompt caching are routed to the same model deployment where a cache write occurred. 

## Set Up

1. Configure the router so that it uses the `PromptCachingDeploymentCheck` (via setting the `optional_pre_call_checks` property), and configure the models so that they can access multiple deployments of Claude; below, we show an example for multiple AWS accounts (referred to as `account-1` and `account-2`, using the `aws_profile_name` property):
```yaml
router_settings:
  optional_pre_call_checks: ["prompt_caching"]

model_list:
- litellm_params:
    model: us.anthropic.claude-sonnet-4-5-20250929-v1:0
    aws_profile_name: account-1
    aws_region_name: us-west-2
  model_info:
    litellm_provider: bedrock
  model_name: us.anthropic.claude-sonnet-4-5-20250929-v1:0
- litellm_params:
    model: us.anthropic.claude-sonnet-4-5-20250929-v1:0
    aws_profile_name: account-2
    aws_region_name: us-west-2
  model_info:
    litellm_provider: bedrock
  model_name: us.anthropic.claude-sonnet-4-5-20250929-v1:0
```
2. Utilize Claude Code:
   1. Launch Claude Code, which will do a warm-up API call that tries to cache its warm-up prompt and its system prompt.
   2. Wait a few seconds, then quit Claude Code and re-open it.
   3. You'll notice that the warm-up API call successfully gets a cache hit (if using Claude Code in an IDE like VS Code, ensure that you don't do anything between step 2.1 and 2.2 here, otherwise there may not be a cache hit): 
      1. Go to the [LiteLLM Request Logs page](../proxy/ui_logs.md) in the Admin UI
      2. Click on the individual requests to see (a) the cache creation and cache read tokens; and (b) the Model ID. In particular, the API call from step 2.1 should show a cache write, and the API call from step 2.2 should show a cache read; in addition, the Model ID should be equal (meaning the API call is getting forwarded to the same AWS account).

## Related

- [Claude Code - Quickstart](./claude_responses_api.md)
- [Claude Code - Customer Tracking](./claude_code_customer_tracking.md)
- [Claude Code - Plugin Marketplace](./claude_code_plugin_marketplace.md)
- [Claude Code - WebSearch](./claude_code_websearch.md)
- [Proxy - Load Balancing](../proxy/load_balancing.md)
