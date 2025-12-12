# Type Checking Failures to Fix

This document lists all type checking errors that need to be fixed in the codebase.

## 1. Signature Incompatibility: `get_chat_completion_prompt`

Several classes have `get_chat_completion_prompt` method signatures that don't match their supertypes. The missing parameter is `prompt_spec: PromptSpec | None = ...`.

### Files Affected:

1. **`integrations/humanloop.py:151`** ✅ **FIXED**
   - **Issue**: Missing `prompt_spec` parameter in signature
   - **Expected**: `prompt_spec: PromptSpec | None = ...` should be included
   - **Superclass**: `litellm.integrations.custom_logger.CustomLogger`

2. **`integrations/arize/arize_phoenix_prompt_manager.py:429`**
   - **Issue**: Missing `prompt_spec` parameter in signature
   - **Expected**: `prompt_spec: PromptSpec | None = ...` should be included
   - **Superclasses**: 
     - `litellm.integrations.custom_prompt_management.CustomPromptManagement`
     - `litellm.integrations.custom_logger.CustomLogger`
     - `litellm.integrations.prompt_management_base.PromptManagementBase`

3. **`proxy/custom_prompt_management.py:10`**
   - **Issue**: Missing `prompt_spec` parameter in signature
   - **Expected**: `prompt_spec: PromptSpec | None = ...` should be included
   - **Superclasses**:
     - `litellm.integrations.custom_prompt_management.CustomPromptManagement`
     - `litellm.integrations.custom_logger.CustomLogger`
     - `litellm.integrations.prompt_management_base.PromptManagementBase`

## 2. Signature Incompatibility: `async_get_chat_completion_prompt`

Multiple classes have `async_get_chat_completion_prompt` method signatures that don't match their supertypes. Common issues include:
- Missing `ignore_prompt_manager_model: bool | None = ...` parameter
- Missing `ignore_prompt_manager_optional_params: bool | None = ...` parameter
- Wrong type for `litellm_logging_obj` (should be `Logging` not `Any`)

### Files Affected:

1. **`integrations/anthropic_cache_control_hook.py:193`**
   - **Issue**: 
     - Missing `ignore_prompt_manager_model` and `ignore_prompt_manager_optional_params` parameters
     - Wrong type for `litellm_logging_obj` (has `Any`, should be `Logging`)
   - **Superclasses**:
     - `litellm.integrations.custom_logger.CustomLogger`
     - `litellm.integrations.prompt_management_base.PromptManagementBase`

2. **`integrations/dotprompt/dotprompt_manager.py:219`**
   - **Issue**: 
     - Missing `ignore_prompt_manager_model` and `ignore_prompt_manager_optional_params` parameters
     - Wrong type for `litellm_logging_obj` (has `Any`, should be `Logging`)
   - **Superclasses**:
     - `litellm.integrations.custom_logger.CustomLogger`
     - `litellm.integrations.prompt_management_base.PromptManagementBase`

3. **`integrations/langfuse/langfuse_prompt_management.py:178`**
   - **Issue**: Missing `ignore_prompt_manager_model` and `ignore_prompt_manager_optional_params` parameters
   - **Superclasses**:
     - `litellm.integrations.prompt_management_base.PromptManagementBase`
     - `litellm.integrations.custom_logger.CustomLogger`

4. **`integrations/vector_store_integrations/vector_store_pre_call_hook.py:43`**
   - **Issue**: 
     - Missing `ignore_prompt_manager_model` and `ignore_prompt_manager_optional_params` parameters
     - Wrong parameter order: `prompt_spec` appears before `tools` in superclass, but after in subclass
   - **Superclass**: `litellm.integrations.custom_logger.CustomLogger`

5. **`integrations/gitlab/gitlab_prompt_manager.py:566`**
   - **Issue**: 
     - Missing `ignore_prompt_manager_model` and `ignore_prompt_manager_optional_params` parameters
     - Wrong type for `litellm_logging_obj` (has `Any`, should be `Logging`)
   - **Superclasses**:
     - `litellm.integrations.custom_logger.CustomLogger`
     - `litellm.integrations.prompt_management_base.PromptManagementBase`

6. **`integrations/bitbucket/bitbucket_prompt_manager.py:545`**
   - **Issue**: 
     - Missing `ignore_prompt_manager_model` and `ignore_prompt_manager_optional_params` parameters
     - Wrong type for `litellm_logging_obj` (has `Any`, should be `Logging`)
   - **Superclasses**:
     - `litellm.integrations.custom_logger.CustomLogger`
     - `litellm.integrations.prompt_management_base.PromptManagementBase`

## 3. Signature Incompatibility: `should_run_prompt_management`

### File Affected:

**`integrations/arize/arize_phoenix_prompt_manager.py:363`**
- **Issue**: 
  - Missing `prompt_spec: PromptSpec | None` parameter in signature
  - `prompt_id` parameter should be `str | None` not `str`
- **Expected Signature**: `should_run_prompt_management(self, prompt_id: str | None, prompt_spec: PromptSpec | None, dynamic_callback_params: StandardCallbackDynamicParams) -> bool`
- **Superclasses**:
  - `litellm.integrations.custom_prompt_management.CustomPromptManagement`
  - `litellm.integrations.prompt_management_base.PromptManagementBase`

## 4. Signature Incompatibility: `_compile_prompt_helper`

### File Affected:

**`integrations/arize/arize_phoenix_prompt_manager.py:376`**
- **Issue**: 
  - Missing `prompt_spec: PromptSpec | None` parameter in signature
  - `prompt_id` parameter should be `str | None` not `str`
- **Expected Signature**: `_compile_prompt_helper(self, prompt_id: str | None, prompt_spec: PromptSpec | None, prompt_variables: dict[Any, Any] | None, dynamic_callback_params: StandardCallbackDynamicParams, prompt_label: str | None = ..., prompt_version: int | None = ...) -> PromptManagementClient`
- **Superclasses**:
  - `litellm.integrations.custom_prompt_management.CustomPromptManagement`
  - `litellm.integrations.prompt_management_base.PromptManagementBase`

## 5. Argument Type Errors in `arize_phoenix_prompt_manager.py`

### File: `integrations/arize/arize_phoenix_prompt_manager.py:453-454`

- **Line 453**: Argument 8 to `get_chat_completion_prompt` has incompatible type `str | None`; expected `PromptSpec | None`
- **Line 454**: Argument 9 to `get_chat_completion_prompt` has incompatible type `int | None`; expected `str | None`

These appear to be parameter order issues when calling the superclass method.

## 6. Attribute Errors

### File: `proxy/db/db_transaction_queue/redis_update_buffer.py:231`

- **Issue**: `type[ServiceTypes]` has no attribute `REDIS_DAILY_AGENT_SPEND_UPDATE_QUEUE`
- **Fix**: Add the missing attribute to `ServiceTypes` enum or use the correct attribute name

## 7. Abstract Class Instantiation Errors

### Files Affected:

1. **`integrations/arize/__init__.py:34`**
   - **Issue**: Cannot instantiate abstract class `ArizePhoenixPromptManager` with abstract attribute `async_compile_prompt_helper`
   - **Fix**: Implement the `async_compile_prompt_helper` method in `ArizePhoenixPromptManager`

2. **`proxy/custom_prompt_management.py:40`**
   - **Issue**: Cannot instantiate abstract class `X42PromptManagement` with abstract attribute `async_compile_prompt_helper`
   - **Fix**: Implement the `async_compile_prompt_helper` method in `X42PromptManagement`

## 8. Type Incompatibility: Azure Files Handler

### File: `llms/azure/files/handler.py`

1. **Line 33**: Argument `expires_after` to `create` of `AsyncFiles` has incompatible type `FileExpiresAfter | None`; expected `ExpiresAfter | Omit`
2. **Line 72**: Argument `expires_after` to `create` of `Files` has incompatible type `FileExpiresAfter | None`; expected `ExpiresAfter | Omit`

**Fix**: Convert `FileExpiresAfter | None` to `ExpiresAfter | Omit` or use the correct type.

## 9. Callable Type Errors

### File: `integrations/langfuse/langfuse.py`

1. **Line 913**: `callable?` not callable `[misc]`
2. **Line 928**: `callable?` not callable `[misc]`

**Fix**: Ensure the objects being called are properly typed as callable or add proper type guards.

## 10. Union Attribute Errors

### File: `proxy/guardrails/guardrail_hooks/presidio.py`

1. **Line 727**: Item `StreamingChoices` of `Choices | StreamingChoices` has no attribute `message`
2. **Line 731**: Item `StreamingChoices` of `Choices | StreamingChoices` has no attribute `message`

**Fix**: Add type guards to check if the object is `Choices` before accessing the `message` attribute, or handle `StreamingChoices` separately.

## 11. TypedDict and Argument Type Errors

### File: `proxy/openai_files_endpoints/files_endpoints.py`

1. **Line 373**: Incompatible types (expression has type `UploadFile | str`, TypedDict item "anchor" has type `Literal['created_at']`)
2. **Line 374**: Argument 1 to `int` has incompatible type `UploadFile | str`; expected `str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc`

**Fix**: Ensure proper type checking and conversion before using these values.

## 12. Type Incompatibility: Storage Backend Service

### File: `proxy/openai_files_endpoints/storage_backend_service.py:136`

- **Issue**: Argument `purpose` to `OpenAIFileObject` has incompatible type `str`; expected `Literal['assistants', 'assistants_output', 'batch', 'batch_output', 'fine-tune', 'fine-tune-results', 'vision', 'user_data']`

**Fix**: Use the correct literal type or add proper type casting/validation.

---

## Summary

**Total Errors**: 36 errors across 16 files

**Error Categories**:
- Signature incompatibility: 24 errors
- Abstract class instantiation: 2 errors
- Type incompatibility: 6 errors
- Attribute errors: 1 error
- Union attribute errors: 2 errors
- Callable type errors: 2 errors

**Priority Fixes**:
1. Fix signature incompatibilities (most common issue)
2. Implement missing abstract methods
3. Fix type mismatches in Azure files handler
4. Add proper type guards for union types
5. Fix attribute and callable errors

