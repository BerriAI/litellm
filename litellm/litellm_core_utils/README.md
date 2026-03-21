## Folder Contents

This folder contains general-purpose utilities that are used in multiple places in the codebase. 

Core files:
- `streaming_handler.py`: The core streaming logic + streaming related helper utils 
- `core_helpers.py`: code used in `types/` - e.g. `map_finish_reason`. 
- `exception_mapping_utils.py`: utils for mapping exceptions to openai-compatible error types. 
- `default_encoding.py`: code for loading the default encoding (tiktoken)
- `get_llm_provider_logic.py`: code for inferring the LLM provider from a given model name. 
- `duration_parser.py`: code for parsing durations - e.g. "1d", "1mo", "10s"
- `api_route_to_call_types.py`: mapping of API routes to their corresponding CallTypes (e.g., `/chat/completions` -> [acompletion, completion])

