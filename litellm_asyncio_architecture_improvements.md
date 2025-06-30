# LiteLLM Asyncio Error Handling: Architectural Improvements

## Current Problems

### 1. **Inconsistent Exception Conversion**
```python
# Azure handler - converts ALL CancelledError
except asyncio.CancelledError as e:
    raise AzureOpenAIError(status_code=500, message=str(e))

# OpenAI handler - lets some exceptions through
except Exception as e:
    if isinstance(e, OpenAIError):
        raise e
```

### 2. **No Asyncio-Aware Base Classes**
Base classes like `BaseLLM` don't provide asyncio-specific error handling patterns.

### 3. **Duplicated Error Handling Logic**
Every provider implements its own asyncio exception handling with copy-pasted patterns.

### 4. **Lost Exception Context**
Converting exceptions breaks asyncio semantics and loses important debugging context.

## Proposed Architectural Solutions

### Solution 1: Asyncio-Aware Exception Hierarchy

Create exception classes that preserve asyncio semantics while adding provider context:

```python
# litellm/exceptions.py
import asyncio
from typing import Optional, Type, Any

class LiteLLMAsyncError(Exception):
    """Base class for LiteLLM async errors that preserves asyncio semantics"""
    
    def __init__(
        self, 
        message: str, 
        original_exception: Optional[Exception] = None,
        provider: Optional[str] = None,
        status_code: int = 500,
        **kwargs
    ):
        super().__init__(message)
        self.original_exception = original_exception
        self.provider = provider
        self.status_code = status_code
        self.extra_context = kwargs
        
    def should_preserve_asyncio_semantics(self) -> bool:
        """Check if this error should preserve asyncio cancellation semantics"""
        return isinstance(self.original_exception, asyncio.CancelledError)
        
    def reraise_if_asyncio_semantic(self):
        """Re-raise original exception if it has important asyncio semantics"""
        if self.should_preserve_asyncio_semantics():
            raise self.original_exception from self

class AsyncCancellationError(LiteLLMAsyncError):
    """Error that wraps cancellations while preserving semantics"""
    
    def __init__(self, cancellation_type: str, **kwargs):
        self.cancellation_type = cancellation_type  # 'user', 'timeout', 'service'
        super().__init__(**kwargs)

class AsyncTimeoutError(LiteLLMAsyncError):
    """Timeout error that preserves asyncio timeout semantics"""
    pass
```

### Solution 2: Centralized Asyncio Error Handler

Create a central error handler that intelligently routes asyncio exceptions:

```python
# litellm/async_error_handler.py
import asyncio
import inspect
from typing import Optional, Dict, Any, Type, Callable
from contextlib import asynccontextmanager

class AsyncErrorContext:
    """Context for tracking the source and intent of async operations"""
    
    def __init__(
        self, 
        provider: str,
        operation: str,
        allow_user_cancellation: bool = True,
        timeout_source: Optional[str] = None
    ):
        self.provider = provider
        self.operation = operation
        self.allow_user_cancellation = allow_user_cancellation
        self.timeout_source = timeout_source

class AsyncioErrorHandler:
    """Centralized handler for asyncio exceptions with intelligent routing"""
    
    @staticmethod
    def is_user_cancellation() -> bool:
        """Check if the current task was cancelled by user/timeout"""
        current_task = asyncio.current_task()
        return current_task and (
            current_task.cancelled() or 
            current_task.cancelling() > 0
        )
    
    @staticmethod 
    def handle_asyncio_exception(
        exception: Exception,
        context: AsyncErrorContext,
        logging_obj: Optional[Any] = None,
        **log_context
    ) -> Exception:
        """
        Intelligently handle asyncio exceptions based on context
        
        Returns the exception that should be raised
        """
        
        if isinstance(exception, asyncio.CancelledError):
            return AsyncioErrorHandler._handle_cancellation(
                exception, context, logging_obj, **log_context
            )
        elif isinstance(exception, asyncio.TimeoutError):
            return AsyncioErrorHandler._handle_timeout(
                exception, context, logging_obj, **log_context
            )
        else:
            return AsyncioErrorHandler._handle_other_async_error(
                exception, context, logging_obj, **log_context
            )
    
    @staticmethod
    def _handle_cancellation(
        exception: asyncio.CancelledError,
        context: AsyncErrorContext, 
        logging_obj: Optional[Any],
        **log_context
    ) -> Exception:
        """Handle cancellation with intelligent source detection"""
        
        is_user_cancellation = AsyncioErrorHandler.is_user_cancellation()
        
        if is_user_cancellation and context.allow_user_cancellation:
            # User cancellation - preserve asyncio semantics
            if logging_obj:
                logging_obj.post_call(
                    **log_context,
                    additional_args={
                        **log_context.get("additional_args", {}),
                        "cancellation_type": "user",
                        "provider": context.provider
                    },
                    original_response=str(exception)
                )
            return exception  # Return original to preserve semantics
        else:
            # Service cancellation - convert with context
            if logging_obj:
                logging_obj.post_call(
                    **log_context,
                    additional_args={
                        **log_context.get("additional_args", {}),
                        "cancellation_type": "service", 
                        "provider": context.provider
                    },
                    original_response=str(exception)
                )
            return AsyncCancellationError(
                message=f"{context.provider} service cancellation",
                cancellation_type="service",
                original_exception=exception,
                provider=context.provider
            )

@asynccontextmanager
async def asyncio_error_context(
    provider: str,
    operation: str,
    logging_obj: Optional[Any] = None,
    **log_context
):
    """Context manager for handling asyncio errors elegantly"""
    context = AsyncErrorContext(provider, operation)
    try:
        yield context
    except Exception as e:
        # Handle the exception using centralized logic
        handled_exception = AsyncioErrorHandler.handle_asyncio_exception(
            e, context, logging_obj, **log_context
        )
        raise handled_exception
```

### Solution 3: Provider Base Class with Asyncio Support

Create an asyncio-aware base class that all providers inherit from:

```python
# litellm/llms/base_async_llm.py
import asyncio
from typing import Any, Dict, Optional, Callable
from abc import ABC, abstractmethod

class BaseAsyncLLM(ABC):
    """Base class for async LLM providers with built-in asyncio error handling"""
    
    def __init__(self):
        self.provider_name = self.__class__.__name__.lower()
    
    async def safe_acompletion(
        self,
        logging_obj: Any,
        **kwargs
    ):
        """Wrapper around acompletion with standardized asyncio error handling"""
        
        log_context = {
            "input": kwargs.get("data", {}).get("messages", []),
            "api_key": kwargs.get("api_key"),
            "additional_args": {"complete_input_dict": kwargs.get("data", {})},
        }
        
        async with asyncio_error_context(
            provider=self.provider_name,
            operation="completion", 
            logging_obj=logging_obj,
            **log_context
        ) as context:
            return await self._acompletion_impl(**kwargs)
    
    @abstractmethod
    async def _acompletion_impl(self, **kwargs):
        """Provider-specific completion implementation"""
        pass
    
    def create_provider_error(
        self, 
        exception: Exception, 
        status_code: int = 500,
        **kwargs
    ):
        """Create provider-specific error while preserving asyncio semantics"""
        # This would be overridden by each provider
        pass
```

### Solution 4: Decorator-Based Asyncio Error Handling

Create decorators that standardize asyncio error handling:

```python
# litellm/decorators.py
import asyncio
import functools
from typing import Callable, Type, Any, Optional

def handle_asyncio_errors(
    provider: str,
    error_class: Type[Exception],
    preserve_cancellation: bool = True,
    preserve_timeout: bool = True
):
    """Decorator for standardized asyncio error handling"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except asyncio.CancelledError as e:
                if preserve_cancellation and AsyncioErrorHandler.is_user_cancellation():
                    raise  # Preserve user cancellation
                else:
                    # Convert service cancellation
                    raise error_class(
                        status_code=500,
                        message=f"{provider} service cancelled the request: {str(e)}"
                    ) from e
            except asyncio.TimeoutError as e:
                if preserve_timeout:
                    raise  # Preserve timeout semantics
                else:
                    raise error_class(
                        status_code=408,
                        message=f"{provider} timeout: {str(e)}"
                    ) from e
            except Exception as e:
                # Let provider-specific exceptions through
                if isinstance(e, error_class):
                    raise
                # Convert unknown exceptions
                raise error_class(
                    status_code=getattr(e, 'status_code', 500),
                    message=str(e)
                ) from e
        
        return wrapper
    return decorator

# Usage in Azure provider:
class AzureChatCompletion(BaseAsyncLLM):
    
    @handle_asyncio_errors(
        provider="azure",
        error_class=AzureOpenAIError,
        preserve_cancellation=True
    )
    async def _acompletion_impl(self, **kwargs):
        # Original completion logic here
        pass
```

### Solution 5: Smart Exception Chaining

Instead of converting exceptions, chain them to preserve context:

```python
# Enhanced exception classes with chaining
class AzureOpenAIError(LiteLLMAsyncError):
    """Azure error that can preserve asyncio semantics via chaining"""
    
    @classmethod
    def from_asyncio_exception(
        cls, 
        asyncio_exception: Exception,
        context: str = "",
        preserve_semantics: bool = True
    ):
        """Create Azure error while optionally preserving asyncio semantics"""
        
        if preserve_semantics and isinstance(asyncio_exception, asyncio.CancelledError):
            if AsyncioErrorHandler.is_user_cancellation():
                # Don't convert user cancellations - just re-raise
                raise asyncio_exception
        
        # Create chained exception with context
        azure_error = cls(
            status_code=500,
            message=f"Azure operation failed: {context} - {str(asyncio_exception)}",
            original_exception=asyncio_exception
        )
        # Chain the exceptions
        raise azure_error from asyncio_exception
```

## Implementation Strategy

### Phase 1: Central Error Handler
1. Implement `AsyncioErrorHandler` and `AsyncErrorContext`
2. Create `asyncio_error_context` context manager
3. Update Azure handler to use new system

### Phase 2: Base Class Migration  
1. Create `BaseAsyncLLM` with standardized error handling
2. Migrate Azure and OpenAI providers
3. Add provider-specific error factory methods

### Phase 3: Decorator System
1. Implement `@handle_asyncio_errors` decorator
2. Apply to all async provider methods
3. Standardize error responses across providers

### Phase 4: Smart Exception Chaining
1. Enhance exception classes with chaining support
2. Implement semantic preservation logic
3. Add debugging context preservation

## Benefits

### 1. **Consistent Behavior**
All providers handle asyncio errors the same way, making LiteLLM predictable.

### 2. **Preserves Asyncio Semantics**
User timeouts and cancellations work correctly across all providers.

### 3. **Better Debugging**
Exception chaining preserves full context while adding provider information.

### 4. **Maintainable**
Centralized logic means one place to fix asyncio handling issues.

### 5. **Backward Compatible**
New system can be implemented gradually without breaking existing code.

### 6. **Standards Compliant** 
Follows asyncio best practices for library code.

## Example Usage

```python
# Before (current Azure handler)
except asyncio.CancelledError as e:
    raise AzureOpenAIError(status_code=500, message=str(e))

# After (with new system)
@handle_asyncio_errors(provider="azure", error_class=AzureOpenAIError)
async def acompletion(self, **kwargs):
    # Original logic - error handling is automatic
    pass

# Or with context manager approach:
async with asyncio_error_context("azure", "completion", logging_obj, **log_context):
    # Original logic
    pass
```

This architectural approach makes asyncio error handling elegant, consistent, and semantically correct across all LiteLLM providers while maintaining backward compatibility.