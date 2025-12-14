"""Global registry for semantic MCP filter state."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._experimental.mcp_server.utils import split_server_prefix_from_name
from litellm.proxy.mcp_semantic_filter.stores.base import ToolVectorStore

from .settings import SemanticFilterConfig
from .stores import ToolVectorRecord, create_vector_store


@dataclass
class ToolDocument:
    tool_id: str
    server_label: str
    text: str


class SemanticMCPFilterRegistry:
    def __init__(self) -> None:
        self._config: Optional[SemanticFilterConfig] = None
        self._store: Optional[ToolVectorStore] = None
        self._documents: Dict[str, ToolDocument] = {}
        self._server_index: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._config and self._config.enabled and self._store)

    def reset(self) -> None:
        self._config = None
        self._store = None
        self._documents = {}
        self._server_index = {}

    def configure(self, config: SemanticFilterConfig) -> None:
        store = create_vector_store(config.vector_store)
        if not store:
            verbose_proxy_logger.warning(
                "semantic_mcp_filter disabled: unable to initialize backend"
            )
            self.reset()
            return

        self._config = config
        self._store = store
        self._documents = {}
        self._server_index = {}

    async def rebuild_index(self, mcp_server_manager: Any) -> None:
        if not self.enabled:
            return
        async with self._lock:
            tools = await mcp_server_manager.list_all_registered_tools()
            docs = self._build_documents(tools)
            store = self._store
            if store is None:
                return

            if not docs:
                if self._documents:
                    store.remove_records(self._documents.keys())
                self._documents = {}
                self._server_index = {}
                return

            new_docs_map = {doc.tool_id: doc for doc in docs}
            current_ids = set(self._documents.keys())
            new_ids = set(new_docs_map.keys())
            removed_ids = current_ids - new_ids

            changed_docs: List[ToolDocument] = []
            for doc in docs:
                existing = self._documents.get(doc.tool_id)
                if existing is None:
                    changed_docs.append(doc)
                    continue
                if (
                    existing.text != doc.text
                    or existing.server_label != doc.server_label
                ):
                    changed_docs.append(doc)

            if changed_docs:
                records = await self._embed_documents(changed_docs)
                if records:
                    store.upsert_records(records)

            if removed_ids:
                store.remove_records(removed_ids)

            self._documents = new_docs_map
            self._server_index = {}
            for doc in docs:
                self._server_index.setdefault(doc.server_label, set()).add(doc.tool_id)

            verbose_proxy_logger.debug(
                "semantic_mcp_filter indexed %s tools", len(docs)
            )

    async def handle_server_refresh(self, mcp_server_manager: Any) -> None:
        await self.rebuild_index(mcp_server_manager)

    async def query(
        self,
        *,
        prompt: str,
        allowed_servers: Optional[Iterable[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        if not self.enabled:
            return []
        if not prompt.strip():
            return []

        vector = await self._embed_query(prompt)
        if vector is None:
            return []

        config_top_k = self._config.vector_store.top_k if self._config else 5
        top_k = top_k or config_top_k

        allowed = self._normalize_allowed_servers(allowed_servers)
        store = self._store
        if store is None:
            return []
        results = store.query(vector, top_k * 3)
        filtered: List[str] = []
        for tool_id, _ in results:
            doc = self._documents.get(tool_id)
            if not doc:
                continue
            if allowed and doc.server_label not in allowed:
                continue
            filtered.append(tool_id)
            if len(filtered) >= top_k:
                break
        return filtered

    def _normalize_allowed_servers(
        self, allowed_servers: Optional[Iterable[str]]
    ) -> Optional[Set[str]]:
        config_allowed = None
        if self._config and self._config.include_servers:
            config_allowed = set(self._config.include_servers)

        if allowed_servers:
            allowed_set = set(allowed_servers)
            if config_allowed is not None:
                return allowed_set & config_allowed
            return allowed_set

        return config_allowed

    def get_tool_server(self, tool_id: str) -> Optional[str]:
        doc = self._documents.get(tool_id)
        return doc.server_label if doc else None

    def _build_documents(self, tools: Sequence[Any]) -> List[ToolDocument]:
        docs: List[ToolDocument] = []
        for tool in tools:
            tool_name = getattr(tool, "name", None)
            if not tool_name and isinstance(tool, dict):
                tool_name = tool.get("name")
            if not tool_name:
                continue
            _, server_label = split_server_prefix_from_name(tool_name)
            if not server_label:
                server_label = ""
            if self._config and self._config.include_servers:
                if server_label not in self._config.include_servers:
                    continue
            text = self._build_tool_text(tool_name, tool)
            docs.append(
                ToolDocument(
                    tool_id=tool_name,
                    server_label=server_label,
                    text=text,
                )
            )
        return docs

    def _build_tool_text(self, tool_name: str, tool: Any) -> str:
        description = getattr(tool, "description", None)
        if description is None and isinstance(tool, dict):
            description = tool.get("description")
        schema = getattr(tool, "inputSchema", None)
        if schema is None and isinstance(tool, dict):
            schema = tool.get("inputSchema") or tool.get("input_schema")

        schema_text = ""
        if schema:
            try:
                schema_text = json.dumps(schema, sort_keys=True)
            except TypeError:
                schema_text = str(schema)

        parts = [tool_name, description or "", schema_text]
        return "\n".join(part for part in parts if part)

    async def _embed_documents(
        self, docs: Sequence[ToolDocument]
    ) -> List[ToolVectorRecord]:
        texts = [doc.text for doc in docs]
        vectors = await self._batch_embed(texts)
        return [
            ToolVectorRecord(tool_id=doc.tool_id, vector=vector)
            for doc, vector in zip(docs, vectors)
        ]

    async def _embed_query(self, text: str) -> Optional[Sequence[float]]:
        vectors = await self._batch_embed([text])
        return vectors[0] if vectors else None

    async def _batch_embed(self, texts: Sequence[str]) -> List[Sequence[float]]:
        if not texts:
            return []
        if not self._config:
            return []
        model = self._config.embedding.model
        params = dict(self._config.embedding.parameters)

        loop = asyncio.get_running_loop()
        batches: List[List[str]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batches.append(list(texts[i : i + batch_size]))

        vectors: List[Sequence[float]] = []
        for batch in batches:
            current_batch = list(batch)
            embed_partial = partial(
                litellm.embedding,
                model=model,
                input=current_batch,
                **params,
            )
            response = await loop.run_in_executor(None, embed_partial)
            response_data = getattr(response, "data", None)
            if response_data is None:
                try:
                    response_data = response.get("data")  # type: ignore[attr-defined]
                except AttributeError:
                    response_data = None
            if not response_data:
                continue
            for item in response_data:
                vectors.append(item["embedding"])
        return vectors


semantic_mcp_filter_registry = SemanticMCPFilterRegistry()
