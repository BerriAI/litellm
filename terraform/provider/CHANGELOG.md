# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **guardrail**: New `litellm_guardrail` resource and data source to manage database-backed guardrails through the proxy's `/guardrails` API. Provider-specific parameters are supplied as a JSON `litellm_params` object; sensitive values are masked by the proxy on read, so that field is treated as write-mostly to avoid spurious diffs (#33392)

### Fixed

- **organization**: Send `PATCH` instead of `POST` to `/organization/update` and `/organization/member_update`, matching the methods the LiteLLM proxy serves; organization and organization member updates previously failed with a 405

### Changed

- The provider source of truth moved to `terraform/provider/` in [BerriAI/litellm](https://github.com/BerriAI/litellm); this repository is now a release mirror. CI in the monorepo statically audits every endpoint the provider calls against the proxy's OpenAPI schema on every change

## [0.2.2] - 2026-05-13

### Fixed

- **key**: Include `tags` in `UpdateKey` payload so tag changes on an existing `litellm_key` are applied on update instead of being silently dropped (#41)

## [0.2.1] - 2026-04-13

### Fixed

- **team, organization**: Use pointer types for `tpm_limit`, `rpm_limit`, and `max_budget` to prevent zero-value diffs on every `terraform plan` when these fields are not configured (#31)

## [0.2.0] - 2026-04-03

### ⚠️ Breaking Changes

#### `litellm_key`: API keys are no longer stored in Terraform state

**Why this change?** Storing raw API keys in Terraform state is a security risk — state files are often stored in S3, Terraform Cloud, or other backends where the key could be exposed even with encryption at rest. This release eliminates that risk entirely.

**What changed:**
- The `key` attribute is now **write-only** — available during `terraform apply` so you can pipe it to a secrets manager, but never persisted to state
- The resource ID has changed from the raw key value to its **SHA-256 hash (`token_id`)** — safe to store in state, cannot be used to authenticate
- **Requires Terraform 1.11+**

**Migration steps for existing `litellm_key` resources:**

1. Find the `token_id` for each key via the LiteLLM UI or `GET /key/info?key=<your-key>`
2. Remove the old resource from state:
   ```
   terraform state rm litellm_key.example
   ```
3. Re-import using the token_id:
   ```
   terraform import litellm_key.example <token_id>
   ```

> ⚠️ After upgrading, you cannot retrieve the raw key from state. Make sure you have the key value stored somewhere safe before migrating, or plan to rotate the key after re-import.

**Security best practice:** Since the key is only available during the initial `terraform apply`, pipe it directly to a secrets manager:

```hcl
resource "aws_ssm_parameter" "litellm_key" {
  name  = "/myapp/litellm-key"
  type  = "SecureString"
  value = litellm_key.example.key
}
```

### Fixed

- **key**: API key is no longer stored in Terraform state. The `key` attribute is now write-only and `token_id` is used as the resource ID (#27)
- **model**: Handle eventual consistency in model reads post-create (#26)

## [0.1.2] - 2026-02-17

### Added
- **Documentation**: Added RELEASING.md with comprehensive release process documentation
  - GPG key setup instructions
  - Step-by-step release workflow
  - Troubleshooting guide
  - Security best practices

## [0.1.1] - 2026-02-11

### Added
- **New Model Modes**: Added support for `audio_speech` and `rerank` model modes
  - `audio_speech`: For text-to-speech models (e.g., Gemini TTS, OpenAI TTS)
  - `rerank`: For reranking/semantic ranking models (e.g., Cohere Rerank, Vertex AI Semantic Ranker)

### Fixed
- Implemented exponential backoff for credential reads
- Only include cost fields when explicitly set in model resource
- Added litellm_credential_name support

## [0.3.14] - 2025-08-24

### Added
- **Enhanced JSON Parsing**: Added support for JSON string parsing in `additional_litellm_params`
  - JSON objects and arrays (starting with `{` or `[`) are now automatically parsed
  - Maintains backward compatibility with existing string-to-type conversion
  - Enables complex nested parameter configurations
- **Parameter Dropping Feature**: Added `additional_drop_params` special parameter
  - Allows removal of unwanted parameters from final `litellm_params` before API submission
  - Specified as JSON array string: `"additional_drop_params" = "[\"reasoningEffort\"]"`
  - Useful for overriding or removing built-in parameters when needed
- **Enhanced Examples**: Updated `examples/model_additional_params.tf` with comprehensive JSON parsing examples
  - Demonstrates all supported value types (boolean, integer, float, string, JSON objects/arrays)
  - Includes real-world Azure model configuration with parameter dropping
  - Shows both simple and complex use cases

### Changed
- **Documentation Enhancement**: Updated `docs/resources/model.md` with detailed JSON parsing documentation
  - Added comprehensive explanation of conversion rules and behavior
  - Included special `additional_drop_params` parameter documentation
  - Enhanced examples showing all supported parameter types and JSON parsing capabilities

### Technical Details
- Enhanced parameter processing logic in `createOrUpdateModel()` function
- Added JSON detection and parsing for string values starting with `[` or `{`
- Implemented parameter filtering system for `additional_drop_params`
- Maintains full backward compatibility with existing configurations

## [0.3.13] - 2025-08-24

### Changed
- Documentation: Performed a documentation audit and improvements across resources and data-sources. Added missing argument references, clarified types/defaults, documented implementation behaviors (e.g., additional_litellm_params parsing and state-preservation), and added an `examples/` directory with runnable HCL examples (starting with `examples/model_additional_params.tf`).
- Docs: Updated `docs/resources/model.md` with missing fields (`vertex_*`, pixel/second cost fields, and `additional_litellm_params`) and added conversion rules and an example.
- Docs Index: Added references to the new `examples/` directory in `docs/index.md`.

## [0.3.12] - 2025-08-13

### Added
- **New AWS Parameters**: Added `aws_session_name` and `aws_role_name` to model resource for cross-account access scenarios
  - Support for AWS session names in cross-account access configurations
  - Support for AWS IAM role names for cross-account access
  - Enhanced AWS Bedrock integration capabilities

### Changed
- **Documentation Overhaul**: Comprehensive update to all provider documentation
  - Updated provider source references from `bitop/litellm` to `registry.terraform.io/ncecere/litellm`
  - Consolidated all scattered example files into organized documentation structure
  - Enhanced all resource documentation with multiple real-world examples
  - Added comprehensive cross-resource integration examples
- **Vector Store Documentation**: Updated to reflect only officially supported LiteLLM providers
  - Removed unsupported providers (Pinecone, Weaviate, Chroma, Qdrant, Milvus, FAISS)
  - Added accurate examples for supported providers: AWS Bedrock Knowledge Bases, OpenAI Vector Stores, Azure Vector Stores, Vertex AI RAG Engine, PG Vector
  - Updated provider-specific parameters with correct configurations
  - Added references to official LiteLLM documentation
- **Project Organization**: Cleaned up project structure
  - Removed scattered example files from root directory
  - Consolidated all examples into comprehensive documentation
  - Updated README.md to reflect current capabilities and structure

### Fixed
- Corrected vector store provider documentation to match LiteLLM's official capabilities
- Updated all documentation links and references for accuracy

## [0.3.11] - 2025-08-10

### Added
- **New Resource**: `litellm_credential` - Manage credentials for secure authentication
  - Support for storing sensitive credential values (API keys, tokens, etc.)
  - Non-sensitive credential information storage
  - Model ID association for credentials
  - Secure handling of sensitive data with Terraform's sensitive attribute
- **New Resource**: `litellm_vector_store` - Manage vector stores for embeddings and RAG
  - Support for multiple vector store providers (Pinecone, Weaviate, Chroma, Qdrant, etc.)
  - Integration with credential management for secure authentication
  - Configurable metadata and provider-specific parameters
  - Full CRUD operations for vector store lifecycle management
- **New Data Source**: `litellm_credential` - Retrieve information about existing credentials
  - Read-only access to credential metadata (sensitive values excluded for security)
  - Support for model ID filtering
  - Cross-stack and cross-configuration referencing capabilities
- **New Data Source**: `litellm_vector_store` - Retrieve information about existing vector stores
  - Complete vector store information retrieval
  - Support for monitoring, validation, and cross-referencing use cases
  - Metadata-based conditional logic support
- Enhanced API response handling for credential and vector store operations
- Comprehensive documentation and examples for new resources and data sources
- Example Terraform configurations for common use cases

### Changed
- Extended `utils.go` with specialized API response handlers for credentials and vector stores
- Updated provider configuration to include new resources and data sources
- Enhanced error handling for credential and vector store not found scenarios

## [0.3.10] - 2025-08-10

### Added
- **New Resource**: `litellm_mcp_server` - Manage MCP (Model Context Protocol) servers
  - Support for HTTP, SSE, and stdio transport types
  - Configurable authentication types (none, bearer, basic)
  - MCP access groups for permission management
  - Cost tracking configuration for MCP tools
  - Environment variables and command arguments for stdio transport
  - Health check status monitoring
  - Comprehensive documentation and examples

### Changed
- Updated provider to support MCP server management functionality
- Enhanced API response handling for MCP-specific operations

## [0.3.9] - 2025-08-10

### Fixed
- Fixed issue where omitting `budget_duration` in key resource caused API error "Invalid duration format"
- Added missing `omitempty` JSON tag to `BudgetDuration` field in Key struct to prevent sending empty strings to API

## [0.3.8] - 2025-08-08

### Added
- Added `additional_litellm_params` field to model resource for custom parameters beyond standard ones
- Support for passing custom parameters like `drop_params`, `timeout`, `max_retries`, `organization`, etc.
- Automatic type conversion for string values to appropriate types (boolean, integer, float)
- Full backward compatibility with existing model configurations
- Comprehensive example demonstrating various use cases with different providers

## [0.3.7] - 2025-08-08

### Fixed
- Fixed issue where changing max_budget_in_team didn't update existing team members with new budget
- Added budget change detection using d.HasChange to update ALL existing members when budget changes
- Implemented tracking to avoid duplicate API calls for members already updated
- Enhanced debug logging for budget update operations

## [0.3.6] - 2025-08-08

### Fixed
- Fixed issue where models deleted from LiteLLM proxy caused terraform plan to fail instead of planning recreation
- Enhanced ErrorResponse struct to properly parse LiteLLM proxy error format with Detail field
- Improved isModelNotFoundError function to detect "not found on litellm proxy" messages in Detail.Error field

## [0.3.5] - 2025-08-08

### Fixed
- Fixed team member update behavior to use member_update endpoint instead of delete/re-add
- Restored team_member_permissions functionality to litellm_team resource
- Enhanced team resource with proper permissions management endpoints

## [0.3.0] - 2025-04-23

### Fixed
- Implemented retry mechanism with exponential backoff for model read operations
- Added detailed logging for retry attempts
- Improved error handling for "model not found" errors

## [0.2.9] - 2025-04-23

### Fixed
- Increased delay after model creation from 2 to 5 seconds to fix "model not found" errors
- Added logging to confirm delay is working properly

## [0.2.8] - 2025-04-23

### Fixed
- Added delay after model creation to fix "model not found" errors when the LiteLLM proxy hasn't fully registered the model yet

## [0.2.7] - 2025-04-23

### Fixed
- Fixed issue where `thinking_enabled` and `merge_reasoning_content_in_choices` values were not being preserved in state, causing Terraform to want to modify them on every run

## [0.2.6] - 2025-03-13

### Added
- Added new `merge_reasoning_content_in_choices` option to model resource

## [0.2.5] - 2025-03-13

### Fixed
- Fixed issue where `thinking_budget_tokens` was being added to models that don't have `thinking_enabled = true`

## [0.2.4] - 2025-03-13

### Added
- Added new `thinking` capability to model resource with configurable parameters:
  - `thinking_enabled` - Boolean to enable/disable thinking capability (default: false)
  - `thinking_budget_tokens` - Integer to set token budget for thinking (default: 1024)

## [0.2.2] - 2025-02-06

### Added
- Added new `reasoning_effort` parameter to model resource with values: "low", "medium", "high"
- Added "chat" mode to model resource

### Changed
- Updated model mode options to: "completion", "embedding", "image_generation", "chat", "moderation", "audio_transcription"

## [1.0.0] - 2024-01-17

### Added
- Initial release of the LiteLLM Terraform Provider
- Support for managing LiteLLM models
- Support for managing teams and team members
- Comprehensive documentation for all resources
