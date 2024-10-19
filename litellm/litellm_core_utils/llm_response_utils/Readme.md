# Helper functions to convert OpenAI Spec Dictionary Responses to corresponding litellm Pydantic Objects

Note: Rerank Follows the Cohere API Spec

## Folder Structure:
```
llm_response_utils/
├── init.py
├── convert_dict_to_chat_completion_response.py
├── convert_dict_to_embedding_response.py
├── convert_dict_to_image_generation_response.py
├── convert_dict_to_rerank_response.py
├── convert_dict_to_streaming_response.py
├── convert_dict_to_transcription_response.py
├── handle_parallel_tool_calls.py
└── response_converter.py
```