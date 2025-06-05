# INSTRUCTIONS FOR LITELLM

This document provides comprehensive instructions for AI agents working in the LiteLLM repository.

## OVERVIEW

LiteLLM is a unified interface for 100+ LLMs that:
- Translates inputs to provider-specific completion, embedding, and image generation endpoints
- Provides consistent OpenAI-format output across all providers
- Includes retry/fallback logic across multiple deployments (Router)
- Offers a proxy server (LLM Gateway) with budgets, rate limits, and authentication
- Supports advanced features like function calling, streaming, caching, and observability

## REPOSITORY STRUCTURE

### Core Components
- `litellm/` - Main library code
  - `llms/` - Provider-specific implementations (OpenAI, Anthropic, Azure, etc.)
  - `proxy/` - Proxy server implementation (LLM Gateway)
  - `router_utils/` - Load balancing and fallback logic
  - `types/` - Type definitions and schemas
  - `integrations/` - Third-party integrations (observability, caching, etc.)

### Key Directories
- `tests/` - Comprehensive test suites
- `docs/my-website/` - Documentation website
- `ui/litellm-dashboard/` - Admin dashboard UI
- `enterprise/` - Enterprise-specific features

## DEVELOPMENT GUIDELINES

### MAKING CODE CHANGES

1. **Provider Implementations**: When adding/modifying LLM providers:
   - Follow existing patterns in `litellm/llms/{provider}/`
   - Implement proper transformation classes that inherit from `BaseConfig`
   - Support both sync and async operations
   - Handle streaming responses appropriately
   - Include proper error handling with provider-specific exceptions

2. **Type Safety**: 
   - Use proper type hints throughout
   - Update type definitions in `litellm/types/`
   - Ensure compatibility with both Pydantic v1 and v2

3. **Testing**:
   - Add tests in appropriate `tests/` subdirectories
   - Include both unit tests and integration tests
   - Test provider-specific functionality thoroughly
   - Consider adding load tests for performance-critical changes

### IMPORTANT PATTERNS

1. **Function/Tool Calling**:
   - LiteLLM standardizes tool calling across providers
   - OpenAI format is the standard, with transformations for other providers
   - See `litellm/llms/anthropic/chat/transformation.py` for complex tool handling

2. **Streaming**:
   - All providers should support streaming where possible
   - Use consistent chunk formatting across providers
   - Handle both sync and async streaming

3. **Error Handling**:
   - Use provider-specific exception classes
   - Maintain consistent error formats across providers
   - Include proper retry logic and fallback mechanisms

4. **Configuration**:
   - Support both environment variables and programmatic configuration
   - Use `BaseConfig` classes for provider configurations
   - Allow dynamic parameter passing

## PROXY SERVER (LLM GATEWAY)

The proxy server is a critical component that provides:
- Authentication and authorization
- Rate limiting and budget management
- Load balancing across multiple models/deployments
- Observability and logging
- Admin dashboard UI
- Enterprise features

Key files:
- `litellm/proxy/proxy_server.py` - Main server implementation
- `litellm/proxy/auth/` - Authentication logic
- `litellm/proxy/management_endpoints/` - Admin API endpoints

## MCP (MODEL CONTEXT PROTOCOL) SUPPORT

LiteLLM supports MCP for agent workflows:
- MCP server integration for tool calling
- Transformation between OpenAI and MCP tool formats
- Support for external MCP servers (Zapier, Jira, Linear, etc.)
- See `litellm/experimental_mcp_client/` and `litellm/proxy/_experimental/mcp_server/`

## TESTING CONSIDERATIONS

1. **Provider Tests**: Test against real provider APIs when possible
2. **Proxy Tests**: Include authentication, rate limiting, and routing tests
3. **Performance Tests**: Load testing for high-throughput scenarios
4. **Integration Tests**: End-to-end workflows including tool calling

## DOCUMENTATION

- Keep documentation in sync with code changes
- Update provider documentation when adding new providers
- Include code examples for new features
- Update changelog and release notes

## SECURITY CONSIDERATIONS

- Handle API keys securely
- Validate all inputs, especially for proxy endpoints
- Consider rate limiting and abuse prevention
- Follow security best practices for authentication

## ENTERPRISE FEATURES

- Some features are enterprise-only
- Check `enterprise/` directory for enterprise-specific code
- Maintain compatibility between open-source and enterprise versions

## COMMON PITFALLS TO AVOID

1. **Breaking Changes**: LiteLLM has many users - avoid breaking existing APIs
2. **Provider Specifics**: Each provider has unique quirks - handle them properly
3. **Rate Limits**: Respect provider rate limits in tests
4. **Memory Usage**: Be mindful of memory usage in streaming scenarios
5. **Dependencies**: Keep dependencies minimal and well-justified

## HELPFUL RESOURCES

- Main documentation: https://docs.litellm.ai/
- Provider-specific docs in `docs/my-website/docs/providers/`
- Admin UI for testing proxy features

## WHEN IN DOUBT

- Follow existing patterns in the codebase
- Check similar provider implementations
- Ensure comprehensive test coverage
- Update documentation appropriately
- Consider backward compatibility impact 