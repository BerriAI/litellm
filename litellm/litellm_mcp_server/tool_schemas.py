"""
JSON Schema definitions for each MCP tool exposed by the LiteLLM MCP server.
"""

CHAT_COMPLETION_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "The LLM model to use (e.g. 'gpt-4o', 'claude-sonnet-4-20250514', 'anthropic/claude-sonnet-4-20250514'). Use provider/model format for non-OpenAI providers.",
        },
        "messages": {
            "type": "array",
            "description": "A list of messages comprising the conversation.",
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "user", "assistant", "tool"],
                        "description": "The role of the message author.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the message.",
                    },
                },
                "required": ["role", "content"],
            },
        },
        "temperature": {
            "type": "number",
            "description": "Sampling temperature between 0 and 2. Higher values make output more random.",
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum number of tokens to generate in the response.",
        },
        "top_p": {
            "type": "number",
            "description": "Nucleus sampling parameter. Consider tokens with top_p probability mass.",
        },
        "stop": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 4 sequences where the API will stop generating further tokens.",
        },
        "presence_penalty": {
            "type": "number",
            "description": "Penalize new tokens based on whether they appear in the text so far (-2.0 to 2.0).",
        },
        "frequency_penalty": {
            "type": "number",
            "description": "Penalize new tokens based on their existing frequency in the text (-2.0 to 2.0).",
        },
        "api_base": {
            "type": "string",
            "description": "Override the default API base URL for the provider.",
        },
        "api_key": {
            "type": "string",
            "description": "Override the default API key for the provider.",
        },
        "api_version": {
            "type": "string",
            "description": "API version to use (relevant for Azure OpenAI).",
        },
    },
    "required": ["model", "messages"],
}

EMBEDDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "The embedding model to use (e.g. 'text-embedding-3-small', 'text-embedding-ada-002').",
        },
        "input": {
            "type": ["string", "array"],
            "description": "Input text to embed. Can be a string or array of strings.",
            "items": {"type": "string"},
        },
        "api_base": {
            "type": "string",
            "description": "Override the default API base URL for the provider.",
        },
        "api_key": {
            "type": "string",
            "description": "Override the default API key for the provider.",
        },
    },
    "required": ["model", "input"],
}

IMAGE_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "The image generation model to use (e.g. 'dall-e-3', 'dall-e-2').",
        },
        "prompt": {
            "type": "string",
            "description": "A text description of the desired image(s).",
        },
        "n": {
            "type": "integer",
            "description": "The number of images to generate (1-10).",
        },
        "size": {
            "type": "string",
            "description": "The size of the generated images (e.g. '1024x1024', '512x512').",
        },
        "quality": {
            "type": "string",
            "description": "The quality of the image ('standard' or 'hd').",
        },
        "api_base": {
            "type": "string",
            "description": "Override the default API base URL for the provider.",
        },
        "api_key": {
            "type": "string",
            "description": "Override the default API key for the provider.",
        },
    },
    "required": ["prompt"],
}

TEXT_COMPLETION_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "The model to use for text completion (e.g. 'gpt-3.5-turbo-instruct', 'text-completion-openai/gpt-3.5-turbo-instruct').",
        },
        "prompt": {
            "type": "string",
            "description": "The prompt to generate completions for.",
        },
        "temperature": {
            "type": "number",
            "description": "Sampling temperature between 0 and 2.",
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum number of tokens to generate.",
        },
        "top_p": {
            "type": "number",
            "description": "Nucleus sampling parameter.",
        },
        "stop": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Up to 4 sequences where generation stops.",
        },
        "api_base": {
            "type": "string",
            "description": "Override the default API base URL for the provider.",
        },
        "api_key": {
            "type": "string",
            "description": "Override the default API key for the provider.",
        },
    },
    "required": ["model", "prompt"],
}

TRANSCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "The transcription model to use (e.g. 'whisper-1').",
        },
        "file": {
            "type": "string",
            "description": "Path to the audio file to transcribe.",
        },
        "language": {
            "type": "string",
            "description": "The language of the input audio (ISO-639-1 code).",
        },
        "prompt": {
            "type": "string",
            "description": "Optional text to guide the model's style or continue a previous audio segment.",
        },
        "response_format": {
            "type": "string",
            "description": "The format of the transcript output ('json', 'text', 'srt', 'verbose_json', 'vtt').",
        },
        "temperature": {
            "type": "number",
            "description": "Sampling temperature between 0 and 1.",
        },
        "api_base": {
            "type": "string",
            "description": "Override the default API base URL for the provider.",
        },
        "api_key": {
            "type": "string",
            "description": "Override the default API key for the provider.",
        },
    },
    "required": ["model", "file"],
}

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "The reranking model to use (e.g. 'cohere/rerank-english-v3.0').",
        },
        "query": {
            "type": "string",
            "description": "The search query to rerank documents against.",
        },
        "documents": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The list of documents to rerank.",
        },
        "top_n": {
            "type": "integer",
            "description": "The number of most relevant documents to return.",
        },
        "api_base": {
            "type": "string",
            "description": "Override the default API base URL for the provider.",
        },
        "api_key": {
            "type": "string",
            "description": "Override the default API key for the provider.",
        },
    },
    "required": ["model", "query", "documents"],
}

LIST_MODELS_SCHEMA = {
    "type": "object",
    "properties": {
        "provider": {
            "type": "string",
            "description": "Filter models by provider name (e.g. 'openai', 'anthropic', 'cohere'). If not specified, returns all available models.",
        },
    },
}
