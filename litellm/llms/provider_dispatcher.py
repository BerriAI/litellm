"""
Provider Dispatcher - O(1) provider routing for completion()

Replaces the O(n) if/elif chain in main.py with a fast dispatch table.
This allows adding providers without modifying the main completion() function.

Usage:
    response = ProviderDispatcher.dispatch(
        custom_llm_provider="azure",
        model=model,
        messages=messages,
        ...
    )
"""

from typing import Union
from litellm.types.utils import ModelResponse
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper


class ProviderDispatcher:
    """
    Fast O(1) provider routing using a dispatch table.
    
    Starting with OpenAI as proof of concept, then incrementally add remaining 46 providers.
    """
    
    _dispatch_table = None  # Lazy initialization
    
    @classmethod
    def _initialize_dispatch_table(cls):
        """Initialize dispatch table on first use"""
        if cls._dispatch_table is not None:
            return
        
        # All OpenAI-compatible providers use the same handler
        cls._dispatch_table = {
            "openai": cls._handle_openai,
            "custom_openai": cls._handle_openai,
            "deepinfra": cls._handle_openai,
            "perplexity": cls._handle_openai,
            "nvidia_nim": cls._handle_openai,
            "cerebras": cls._handle_openai,
            "baseten": cls._handle_openai,
            "sambanova": cls._handle_openai,
            "volcengine": cls._handle_openai,
            "anyscale": cls._handle_openai,
            "together_ai": cls._handle_openai,
            "nebius": cls._handle_openai,
            "wandb": cls._handle_openai,
            # TODO: Add remaining providers incrementally
            # "azure": cls._handle_azure,
            # "anthropic": cls._handle_anthropic,
            # "bedrock": cls._handle_bedrock,
            # ... etc
        }
    
    @classmethod
    def dispatch(cls, custom_llm_provider: str, **context) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Dispatch to the appropriate provider handler.
        
        Args:
            custom_llm_provider: Provider name (e.g., 'azure', 'openai')
            **context: All parameters from completion() - model, messages, api_key, etc.
        
        Returns:
            ModelResponse or CustomStreamWrapper for streaming
        
        Raises:
            ValueError: If provider not in dispatch table (use old if/elif as fallback)
        """
        cls._initialize_dispatch_table()
        
        # _dispatch_table is guaranteed to be initialized after _initialize_dispatch_table()
        assert cls._dispatch_table is not None, "Dispatch table should be initialized"
        
        handler = cls._dispatch_table.get(custom_llm_provider)
        if handler is None:
            raise ValueError(
                f"Provider '{custom_llm_provider}' not yet migrated to dispatch table. "
                f"Available providers: {list(cls._dispatch_table.keys())}"
            )
        
        return handler(**context)
    
    @staticmethod  
    def _handle_openai(**ctx) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Handle OpenAI completions.
        
        Complete logic extracted from main.py lines 2029-2135
        """
        # CIRCULAR IMPORT WORKAROUND:
        # We cannot directly import OpenAIChatCompletion class here because:
        # 1. main.py imports from provider_dispatcher.py (this file)
        # 2. provider_dispatcher.py would import from openai.py
        # 3. openai.py might import from main.py -> circular dependency
        #
        # SOLUTION: Use the module-level instances that are already created in main.py
        # These instances are created at module load time (lines 235, 265) and are
        # available via litellm.main module reference.
        #
        # This is "hacky" but necessary because:
        # - We're refactoring a 6,000+ line file incrementally
        # - Breaking circular imports requires careful ordering
        # - Using existing instances avoids recreating handler objects
        # - Future refactoring can move these to a proper registry pattern
        
        import litellm
        from litellm.secret_managers.main import get_secret, get_secret_bool
        from litellm.utils import add_openai_metadata
        import openai
        
        # Access pre-instantiated handlers from main.py (created at lines 235, 265)
        from litellm import main as litellm_main
        openai_chat_completions = litellm_main.openai_chat_completions
        base_llm_http_handler = litellm_main.base_llm_http_handler
        
        # Extract context
        model = ctx['model']
        messages = ctx['messages']
        api_key = ctx.get('api_key')
        api_base = ctx.get('api_base')
        headers = ctx.get('headers')
        model_response = ctx['model_response']
        optional_params = ctx['optional_params']
        litellm_params = ctx['litellm_params']
        logging = ctx['logging_obj']
        acompletion = ctx.get('acompletion', False)
        timeout = ctx.get('timeout')
        client = ctx.get('client')
        extra_headers = ctx.get('extra_headers')
        print_verbose = ctx.get('print_verbose')
        logger_fn = ctx.get('logger_fn')
        custom_llm_provider = ctx.get('custom_llm_provider', 'openai')
        shared_session = ctx.get('shared_session')
        custom_prompt_dict = ctx.get('custom_prompt_dict')
        encoding = ctx.get('encoding')
        stream = ctx.get('stream')
        provider_config = ctx.get('provider_config')
        metadata = ctx.get('metadata')
        organization = ctx.get('organization')
        
        # Get API base with fallbacks
        api_base = (
            api_base
            or litellm.api_base
            or get_secret("OPENAI_BASE_URL")
            or get_secret("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )
        
        # Get organization
        organization = (
            organization
            or litellm.organization
            or get_secret("OPENAI_ORGANIZATION")
            or None
        )
        openai.organization = organization
        
        # Get API key
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret("OPENAI_API_KEY")
        )
        
        headers = headers or litellm.headers
        
        if extra_headers is not None:
            optional_params["extra_headers"] = extra_headers
        
        # PREVIEW: Allow metadata to be passed to OpenAI
        if litellm.enable_preview_features and metadata is not None:
            optional_params["metadata"] = add_openai_metadata(metadata)
        
        # Load config
        config = litellm.OpenAIConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v
        
        # Check if using experimental base handler
        use_base_llm_http_handler = get_secret_bool(
            "EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER"
        )
        
        try:
            if use_base_llm_http_handler:
                # Type checking disabled - complex handler signatures
                response = base_llm_http_handler.completion(  # type: ignore
                    model=model,
                    messages=messages,
                    api_base=api_base,  # type: ignore
                    custom_llm_provider=custom_llm_provider,
                    model_response=model_response,
                    encoding=encoding,
                    logging_obj=logging,
                    optional_params=optional_params,
                    timeout=timeout,  # type: ignore
                    litellm_params=litellm_params,
                    shared_session=shared_session,
                    acompletion=acompletion,
                    stream=stream,
                    api_key=api_key,  # type: ignore
                    headers=headers,
                    client=client,
                    provider_config=provider_config,
                )
            else:
                # Type checking disabled - complex handler signatures
                response = openai_chat_completions.completion(  # type: ignore
                    model=model,
                    messages=messages,
                    headers=headers,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    api_key=api_key,  # type: ignore
                    api_base=api_base,  # type: ignore
                    acompletion=acompletion,
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,  # type: ignore
                    custom_prompt_dict=custom_prompt_dict,  # type: ignore
                    client=client,
                    organization=organization,  # type: ignore
                    custom_llm_provider=custom_llm_provider,
                    shared_session=shared_session,
                )
        except Exception as e:
            # Log the original exception
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=str(e),
                additional_args={"headers": headers},
            )
            raise e
        
        # Post-call logging for streaming
        if optional_params.get("stream", False):
            logging.post_call(
                input=messages,
                api_key=api_key,
                original_response=response,
                additional_args={"headers": headers},
            )
        
        # Type ignore: Handler methods have broad return types (ModelResponse | CustomStreamWrapper | Coroutine | etc)
        # but in practice for chat completions, we only get ModelResponse or CustomStreamWrapper
        return response  # type: ignore

