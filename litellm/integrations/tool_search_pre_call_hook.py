"""
Tool Search Pre-Call Hook

Client-side tool search (BM25 + regex) for providers that don't support
Anthropic's server-side tool search API.

Automatically handles the tool_search agentic loop - when the model calls
tool_search, this hook executes the search and continues the conversation
with expanded tools until the model stops calling tool_search.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes, LLMResponseTypes


class ToolSearchPreCallHook(CustomLogger):
    """
    Hook that handles client-side tool search automatically.

    When a model calls tool_search, this hook:
    1. Executes local BM25/regex search on deferred tools
    2. Expands discovered tools into the tools list
    3. Automatically continues the conversation
    4. Returns final response when model stops calling tool_search
    """

    SUPPORTED_PROVIDERS = {"anthropic"}
    MAX_TOOL_SEARCH_ITERATIONS = 5  # Prevent infinite loops

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        """
        Transform request before sending to provider.

        - Remove deferred tools (don't send many tools to provider)
        - Replace tool_search_tool with regular function tool
        - Store deferred tools for later search execution
        """
        if call_type != CallTypes.anthropic_messages:
            return None

        tools = kwargs.get("tools")
        if not tools:
            return None

        # Detect tool search tool
        tool_search_config = self._detect_tool_search(tools)
        if not tool_search_config:
            return None

        # Check provider support
        model = kwargs.get("model", "")
        try:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model,
                custom_llm_provider=kwargs.get("custom_llm_provider"),
            )
        except Exception:
            custom_llm_provider = None

        if custom_llm_provider in self.SUPPORTED_PROVIDERS:
            return None  # Pass through to server-side

        verbose_logger.debug(
            f"ToolSearchPreCallHook: Client-side tool search for provider={custom_llm_provider}"
        )

        modified_tools, deferred_tools = self._prepare_tools(tools, tool_search_config)
        kwargs["tools"] = modified_tools
        kwargs["_tool_search_config"] = tool_search_config
        kwargs["_deferred_tools"] = deferred_tools
        kwargs["_tool_search_iteration"] = kwargs.get("_tool_search_iteration", 0)
        kwargs["_original_tools"] = tools  # Keep original for reference

        return kwargs

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict,
        response: LLMResponseTypes,
        call_type: Optional[CallTypes],
    ) -> Optional[LLMResponseTypes]:
        """
        Handle tool_search calls automatically.

        When model calls tool_search:
        1. Execute local BM25/regex search
        2. Expand discovered tools
        3. Continue conversation automatically
        4. Return final response
        """
        if call_type != CallTypes.anthropic_messages:
            return None

        tool_search_config = request_data.get("_tool_search_config")
        deferred_tools = request_data.get("_deferred_tools")
        iteration = request_data.get("_tool_search_iteration", 0)

        if not tool_search_config or not deferred_tools:
            return None

        # Check iteration limit
        if iteration >= self.MAX_TOOL_SEARCH_ITERATIONS:
            verbose_logger.warning(
                f"ToolSearchPreCallHook: Max iterations ({self.MAX_TOOL_SEARCH_ITERATIONS}) reached"
            )
            return None

        tool_search_name = tool_search_config.get("name", "tool_search")
        search_type = tool_search_config.get("search_type", "bm25")

        # Get content from response
        content = self._get_response_content(response)
        if not content:
            return None

        # Find tool_search calls and execute searches
        tool_search_calls = []
        expanded_tool_names = set()

        for block in content:
            block_type, block_name, block_id, block_input = self._parse_block(block)

            if block_type == "tool_use" and block_name == tool_search_name:
                query = block_input.get("query", "") if isinstance(block_input, dict) else ""

                # Execute local search
                search_engine = ClientSideToolSearch(deferred_tools)
                results = search_engine.search(query, search_type)

                verbose_logger.debug(
                    f"ToolSearchPreCallHook: Search '{query}' found {len(results)} tool(s)"
                )

                tool_search_calls.append({
                    "tool_use_id": block_id,
                    "query": query,
                    "results": results,
                })

                # Track which tools to expand
                for ref in results:
                    expanded_tool_names.add(ref.get("tool_name"))

        # If no tool_search calls, return as-is
        if not tool_search_calls:
            return None

        # Build expanded tools list
        current_tools = request_data.get("tools", [])
        deferred_dict = {t.get("name"): t for t in deferred_tools}

        for tool_name in expanded_tool_names:
            if tool_name in deferred_dict:
                # Add tool without defer_loading flag
                tool = deferred_dict[tool_name].copy()
                tool.pop("defer_loading", None)
                # Only add if not already present
                if not any(t.get("name") == tool_name for t in current_tools):
                    current_tools.append(tool)

        # Build messages for follow-up call
        messages = request_data.get("messages", [])

        # Add assistant response with tool_use
        assistant_content = self._build_assistant_content(content)
        messages = messages + [{"role": "assistant", "content": assistant_content}]

        # Add tool_results for each search
        tool_results = []
        for call in tool_search_calls:
            found_names = [r.get("tool_name") for r in call["results"]]
            result_text = f"Found {len(found_names)} tool(s): {', '.join(found_names)}. These tools are now available."
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call["tool_use_id"],
                "content": result_text,
            })

        messages = messages + [{"role": "user", "content": tool_results}]

        # Make follow-up call with expanded tools
        verbose_logger.debug(
            f"ToolSearchPreCallHook: Continuing with {len(current_tools)} tools (iteration {iteration + 1})"
        )

        # Import here to avoid circular imports
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages,
        )

        # Prepare follow-up request
        follow_up_kwargs = {
            "model": request_data.get("model"),
            "messages": messages,
            "max_tokens": request_data.get("max_tokens", 1024),
            "tools": current_tools,
            "_tool_search_config": tool_search_config,
            "_deferred_tools": deferred_tools,
            "_tool_search_iteration": iteration + 1,
        }

        # Copy other relevant params
        for key in ["temperature", "top_p", "stop_sequences", "stream"]:
            if key in request_data:
                follow_up_kwargs[key] = request_data[key]

        # Make recursive call
        return await anthropic_messages(**follow_up_kwargs)

    async def async_post_call_streaming_deployment_hook(
        self,
        request_data: dict,
        response_chunk: Any,
        call_type: Optional[CallTypes],
    ) -> Optional[Any]:
        """Handle streaming responses."""
        # Streaming with tool_search is complex - pass through for now
        return None

    def _detect_tool_search(self, tools: List[Dict]) -> Optional[Dict]:
        """Detect tool_search_tool in tools list."""
        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type == "tool_search_tool_regex_20251119":
                return {"search_type": "regex", "name": tool.get("name", "tool_search")}
            elif tool_type == "tool_search_tool_bm25_20251119":
                return {"search_type": "bm25", "name": tool.get("name", "tool_search")}
        return None

    def _prepare_tools(
        self, tools: List[Dict], config: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """Separate deferred tools and create synthetic tool_search function."""
        deferred: List[Dict] = []
        non_deferred: List[Dict] = []

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type in [
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
            ]:
                continue
            elif tool.get("defer_loading", False):
                deferred.append(tool)
            else:
                non_deferred.append(tool)

        tool_search_func = self._create_tool_search_function(config)
        non_deferred.append(tool_search_func)

        return non_deferred, deferred

    def _create_tool_search_function(self, config: Dict) -> Dict:
        """Create synthetic tool_search function in Anthropic format."""
        name = config["name"]
        search_type = config["search_type"]

        if search_type == "regex":
            description = "Search for available tools using regex patterns. Returns tool references that can be used in subsequent requests."
        else:
            description = "Search for available tools using natural language. Returns tool references that can be used in subsequent requests."

        return {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant tools.",
                    }
                },
                "required": ["query"],
            },
        }

    def _get_response_content(self, response: LLMResponseTypes) -> Optional[List]:
        """Extract content from response."""
        if isinstance(response, dict):
            return response.get("content")
        elif hasattr(response, "content"):
            return getattr(response, "content", None)
        return None

    def _parse_block(self, block: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Any]:
        """Parse a content block to extract type, name, id, and input."""
        if isinstance(block, dict):
            return (
                block.get("type"),
                block.get("name"),
                block.get("id"),
                block.get("input", {}),
            )
        return (
            getattr(block, "type", None),
            getattr(block, "name", None),
            getattr(block, "id", None),
            getattr(block, "input", {}),
        )

    def _build_assistant_content(self, content: List) -> List[Dict]:
        """Build assistant content for follow-up message."""
        result = []
        for block in content:
            block_type, block_name, block_id, block_input = self._parse_block(block)

            if block_type == "text":
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                result.append({"type": "text", "text": text})
            elif block_type == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block_id,
                    "name": block_name,
                    "input": block_input,
                })

        return result


class ClientSideToolSearch:
    """
    BM25 and regex search algorithms for tool discovery.

    Searches tool names, descriptions, and parameter information.
    """

    def __init__(self, tools: List[Dict]):
        self.tools = tools
        self._build_index()

    def search(self, query: str, search_type: str, max_results: int = 5) -> List[Dict]:
        """Execute search based on type."""
        if search_type == "regex":
            return self.search_regex(query, max_results)
        return self.search_bm25(query, max_results)

    def _build_index(self) -> None:
        """Build BM25 search index from tools."""
        self._tool_docs: List[Tuple[Dict, str]] = []
        self._idf_cache: Dict[str, float] = {}

        for tool in self.tools:
            doc = self._tool_to_text(tool)
            self._tool_docs.append((tool, doc.lower()))

        if self._tool_docs:
            all_terms: set = set()
            for _, doc in self._tool_docs:
                all_terms.update(self._tokenize(doc))

            n_docs = len(self._tool_docs)
            for term in all_terms:
                doc_freq = sum(
                    1 for _, doc in self._tool_docs if term in self._tokenize(doc)
                )
                self._idf_cache[term] = math.log(
                    (n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1
                )

    def _tool_to_text(self, tool: Dict) -> str:
        """Convert tool definition to searchable text."""
        parts: List[str] = []

        name = tool.get("name") or tool.get("function", {}).get("name", "")
        if name:
            parts.extend([name] * 3)

        desc = tool.get("description") or tool.get("function", {}).get("description", "")
        if desc:
            parts.append(desc)

        schema = tool.get("input_schema") or tool.get("function", {}).get("parameters", {})
        if isinstance(schema, dict):
            for prop_name, prop_info in schema.get("properties", {}).items():
                parts.append(prop_name)
                if isinstance(prop_info, dict) and prop_info.get("description"):
                    parts.append(prop_info["description"])

        return " ".join(parts)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r"\w+", text.lower())

    def search_bm25(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search tools using BM25 algorithm."""
        if not query or not self._tool_docs:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        k1, b = 1.2, 0.75
        avg_doc_len = sum(len(self._tokenize(doc)) for _, doc in self._tool_docs) / len(self._tool_docs)

        scores: List[Tuple[Dict, float]] = []

        for tool, doc_text in self._tool_docs:
            doc_terms = self._tokenize(doc_text)
            doc_len = len(doc_terms)
            term_freqs = Counter(doc_terms)

            score = 0.0
            for term in query_terms:
                if term not in self._idf_cache:
                    continue
                tf = term_freqs.get(term, 0)
                idf = self._idf_cache[term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                score += idf * (numerator / denominator) if denominator > 0 else 0

            if score > 0:
                scores.append((tool, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {"type": "tool_reference", "tool_name": t.get("name") or t.get("function", {}).get("name")}
            for t, _ in scores[:max_results]
        ]

    def search_regex(self, pattern: str, max_results: int = 5) -> List[Dict]:
        """Search tools using regex pattern matching."""
        if not pattern or len(pattern) > 200:
            return []

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []

        matches: List[Tuple[Dict, int]] = []

        for tool in self.tools:
            text = self._tool_to_text(tool)
            match_count = len(regex.findall(text))
            if match_count > 0:
                matches.append((tool, match_count))

        matches.sort(key=lambda x: x[1], reverse=True)

        return [
            {"type": "tool_reference", "tool_name": t.get("name") or t.get("function", {}).get("name")}
            for t, _ in matches[:max_results]
        ]
