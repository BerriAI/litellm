# LiteLLM + n8n Integration Summary

**Status**: ✅ **WORKING** (with some minor limitations)

## Executive Summary

LiteLLM **already works** with n8n (v1.113.3+) as an OpenAI-compatible provider. Users can access 100+ LLMs (Anthropic, Azure, Bedrock, etc.) through a single n8n workflow using LiteLLM proxy.

## What We've Done

### 1. ✅ Compatibility Fixes (Already Merged)
- **PR #13320** - Fixed null field serialization that broke n8n's AgentExecutor
- **Issue #13055** - tool_calls now returns empty array [] instead of null
- **Version**: LiteLLM v1.75.5+ has full n8n compatibility

### 2. ✅ Comprehensive Documentation
Created complete integration guide at `docs/my-website/docs/integrations/n8n.md`:
- Quick start guide
- Configuration steps
- Supported features
- Example workflows
- Troubleshooting guide
- Community resources

### 3. ✅ Community Package Analysis
Reviewed existing solutions:
- [@rlquilez/n8n-nodes-openai-litellm](https://github.com/rlquilez/n8n-nodes-openai-litellm) - Metadata injection
- [paulokuong/litellm-n8n-node](https://github.com/paulokuong/litellm-n8n-node) - Community integration
- [sebastianrueckerai/litellm-n8n](https://github.com/sebastianrueckerai/litellm-n8n) - Examples

## Current Status

### ✅ What Works
- Chat completions (streaming & non-streaming)
- Model listing via `/v1/models`
- Tool/function calling
- AI Agent node v1.7
- All 100+ LiteLLM supported providers
- Custom metadata injection (via community packages)

### ⚠️ Known Issues
1. **AI Agent v2.2 node** - Crashes with "Cannot read properties of null"
   - Tracked: [n8n #19712](https://github.com/n8n-io/n8n/issues/19712)
   - Workaround: Use AI Agent v1.7
   - Status: Closed as "not planned" by n8n team

## How to Use (User Guide)

### Step 1: Configure n8n Credentials
```
Type: OpenAI API
API Key: <your-litellm-api-key>
Base URL: http://localhost:4000 (or your LiteLLM proxy URL)
```

### Step 2: Use OpenAI Chat Model Node
- Select configured credential
- Choose model from dropdown (populated from LiteLLM)
- Build your workflow!

### Step 3: (Optional) Advanced Features
- Use community packages for metadata injection
- Configure fallbacks and load balancing in LiteLLM
- Set up observability with Langfuse/Datadog

## Engagement Plan with n8n Team

### Objective
Get LiteLLM listed as an officially supported AI provider in n8n's documentation and integrations page.

### Approach
1. **Demonstrate Value**
   - Already working with minimal configuration
   - Adds 100+ LLM providers through single integration
   - Active community interest (multiple community packages)
   - LiteLLM made specific compatibility fixes for n8n

2. **Provide Resources**
   - Comprehensive documentation ready to merge
   - Example workflows
   - Troubleshooting guide
   - Community support commitment

3. **Address Concerns**
   - All major compatibility issues already resolved
   - Only outstanding issue is AI Agent v2.2 (has workaround)
   - Willing to maintain integration documentation
   - Can provide support through LiteLLM Discord

### Proposed Actions

#### For n8n Team
1. **PR to n8n docs** - Add LiteLLM to official integrations list
2. **Test workflow** - Validate integration in n8n's test environment
3. **Blog post** - Joint announcement of official support
4. **Issue #19712** - Collaborate on fixing AI Agent v2.2 compatibility

#### For LiteLLM
1. ✅ Create integration documentation (DONE)
2. ✅ Document known issues and workarounds (DONE)
3. 📝 Create example n8n workflows repository
4. 📝 Add n8n to LiteLLM homepage/marketing materials
5. 📝 Consider official LiteLLM community node (@litellm/n8n-nodes-litellm)

## Next Steps

### Immediate (This Week)
1. **Share with Krrish** - Review and approve approach
2. **Post in n8n community** - Announce working integration with guide
3. **Create example repo** - Sample n8n workflows using LiteLLM
4. **Test validation** - Run through full integration test suite

### Short Term (This Month)
1. **Engage n8n team** - Reach out via GitHub/community forum
2. **Fix AI Agent v2.2** - Reproduce, report detailed bug with fix if possible
3. **Create demo video** - Show LiteLLM + n8n in action
4. **Official node** - Evaluate building @litellm/n8n-nodes-litellm

### Long Term
1. **Official partnership** - Explore deeper integration opportunities
2. **Joint webinar** - Educate users on LLM workflow automation
3. **Enterprise features** - SSO, audit logs, team permissions integration

## Success Metrics

- ✅ Documentation published
- ⏳ Listed on n8n integrations page
- ⏳ 100+ stars on example workflows repo
- ⏳ Featured in n8n newsletter/blog
- ⏳ Official LiteLLM node package published

## Key Contacts

- **LiteLLM**: Krrish Dholakia (CEO), Ishaan Jaffer (CTO)
- **n8n**: TBD (need to identify product/partnerships lead)
- **Community**: Authors of existing community packages

## Resources

- **Integration Guide**: `docs/my-website/docs/integrations/n8n.md`
- **LiteLLM Proxy**: https://docs.litellm.ai/docs/proxy/quick_start
- **n8n Docs**: https://docs.n8n.io/
- **Community Forum**: https://community.n8n.io/

---

**Prepared by**: Claude Code
**Date**: February 14, 2026
**Version**: 1.0
