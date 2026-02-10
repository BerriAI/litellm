# Stably LiteLLM Fork

This is Stably's fork of [LiteLLM](https://github.com/BerriAI/litellm). For general LiteLLM documentation, see the [official docs](https://docs.litellm.ai/).

---

## Version History & Bug Fixes

| Version | Commit | Description |
|---------|--------|-------------|
| `stably-2` | `ca2cd1789` | **Exclude cached tokens from TPM calculation** - AWS Bedrock and other providers don't count cached tokens toward rate limits. This feature adds `exclude_cached_tokens_from_tpm` config option to align LiteLLM's TPM calculation with provider behavior. |
| `stably-3` | `8fbe0c61b` | **Fix /responses API TPM bypass** - The Responses API (`/v1/responses`) was bypassing TPM rate limiting because it uses `ResponseAPIUsage` type with `input_tokens`/`output_tokens` instead of `prompt_tokens`/`completion_tokens`. Added proper handling for `ResponseAPIUsage` type and improved `model_group` resolution from multiple sources. |
| `stably-4` | `59151ac07` | **Dev setup optimization** - Optimized Dockerfile for better layer caching. Dependencies are now installed before copying source code, significantly speeding up rebuilds when only source code changes. |
| `stably-5` | `4b69b929b` | **Safe encoding for invalid unicode** - Fixed crashes when LLM responses contain invalid unicode sequences. Added `safe_encode()` utility that handles encoding errors gracefully instead of raising exceptions. |

### Bugs Fixed by Upstream (No Longer Needed)

| Original Commit | Issue | Status |
|-----------------|-------|--------|
| `3e7656f52` | Key duration calculation - keys were expiring at standardized intervals instead of exactly N time units from creation | **Fixed in upstream** - removed from fork |

---

## Rebasing to Latest Upstream

When you need to pull in new models or features from upstream LiteLLM:

### Quick Steps

```bash
# 1. Fetch upstream
git fetch upstream main

# 2. Check how far behind
git log --oneline HEAD..upstream/main | wc -l

# 3. Rebase
git rebase upstream/main

# 4. Resolve conflicts (AI can help with this)

# 5. Run tests (CRITICAL)
```

### Using AI for Rebase

You can use Claude Code or similar AI tools to help with rebasing:

```
帮我 rebase 到 upstream/main，处理所有 conflict
```

The AI will:
1. Identify conflict files
2. Understand both upstream changes and our fixes
3. Merge them correctly

**IMPORTANT: After rebase, you MUST run the proxy e2e tests:**

```bash
# Tests are in the noqa repo
cd /path/to/noqa/packages/ai-proxy/e2e
pnpm test
```

Do NOT deploy without passing these tests.

### Recommended Maintenance Schedule

- **Weekly**: Check `git log --oneline HEAD..upstream/main | wc -l`
- **If < 100 commits**: Rebase immediately (conflicts will be minimal)
- **If > 100 commits**: Schedule a dedicated rebase session

---

## Deployment

Deployment is currently **manual**. Follow these steps:

### 1. Build & Push to ECR

Go to AWS ECR: [litellm repository](https://us-west-2.console.aws.amazon.com/ecr/repositories/private/624143034767/litellm/_/details?region=us-west-2)

Click **"View push commands"** and follow the instructions:

```bash
# 1. Authenticate Docker to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 624143034767.dkr.ecr.us-west-2.amazonaws.com

# 2. Build the image
docker build -t litellm .

# 3. Tag the image
docker tag litellm:latest 624143034767.dkr.ecr.us-west-2.amazonaws.com/litellm:latest

# 4. Push to ECR
docker push 624143034767.dkr.ecr.us-west-2.amazonaws.com/litellm:latest
```

### 2. Deploy via noqa

Once the image is pushed to ECR:

1. Go to the noqa deployment system
2. Select **"Docker container deploy"**
3. Choose **"AI Proxy"**
4. Deploy the latest version

---

## File Structure (Stably-specific)

```
litellm/_version.py                    # Contains STABLY_FORK_VERSION
litellm/litellm_core_utils/
  core_helpers.py                      # get_tokens_for_tpm() function
  json_encoding_utils.py               # safe_encode() function
litellm/proxy/hooks/
  parallel_request_limiter_v3.py       # TPM rate limiting with ResponseAPIUsage support
litellm/llms/custom_httpx/
  http_handler.py                      # Safe unicode handling
```

---

## Contact

For issues with this fork, contact the Stably engineering team.
