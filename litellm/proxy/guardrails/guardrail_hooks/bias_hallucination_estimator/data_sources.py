from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from importlib.util import find_spec as _find_spec
from pathlib import Path
from typing import Any, Optional, cast

_AIOHTTP_AVAILABLE: bool = _find_spec("aiohttp") is not None


class DataSourceResult:
    def __init__(
        self,
        text: str,
        source: str,
        confidence: float = 0.8,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.source = source
        self.confidence = min(max(confidence, 0.0), 1.0)
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"DataSourceResult(source={self.source}, confidence={self.confidence:.2f})"
        )


class DataSource(ABC):
    def __init__(self, name: str = "", enabled: bool = True, priority: int = 0) -> None:
        self.name = name
        self.enabled = enabled
        self.priority = priority

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[DataSourceResult]:
        pass

    async def verify_fact(self, claim: str) -> tuple[bool, Optional[str]]:
        results = await self.search(claim, limit=1)
        if results:
            return True, results[0].text
        return False, None


def _build_keyword_index(documents: list[str | dict[str, Any]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for idx, doc in enumerate(documents):
        text = _get_doc_text(doc).lower()
        for word in set(re.findall(r"\b\w+\b", text)):
            index.setdefault(word, []).append(idx)
    return index


def _get_doc_text(doc: str | dict[str, Any]) -> str:
    if isinstance(doc, str):
        return doc
    return doc.get("text", str(doc))


def _keyword_search(
    query: str,
    documents: list[str | dict[str, Any]],
    index: dict[str, list[int]],
    source_name: str,
    limit: int,
) -> list[DataSourceResult]:
    if not documents or not query:
        return []
    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    if not query_words:
        return []
    doc_indices: set[int] = set()
    for word in query_words:
        if word in index:
            doc_indices.update(index[word])
    scored: list[tuple[float, DataSourceResult]] = []
    for idx in doc_indices:
        doc = documents[idx]
        text = _get_doc_text(doc)
        matching = len(set(re.findall(r"\b\w+\b", text.lower())) & query_words)
        score = matching / len(query_words)
        scored.append(
            (score, DataSourceResult(text=text, source=source_name, confidence=score))
        )
    scored.sort(reverse=True, key=lambda x: x[0])
    return [result for _, result in scored[:limit]]


def _parse_json_docs(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return cast(list[dict[str, Any]], raw)
    return [cast(dict[str, Any], raw)]


class FileDataSource(DataSource):
    def __init__(
        self,
        file_path: str,
        name: str = "",
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        super().__init__(
            name=name or f"file_{Path(file_path).stem}",
            enabled=enabled,
            priority=priority,
        )
        self.file_path = file_path
        self._documents: list[str | dict[str, Any]] = self._load_documents(file_path)
        self._index: dict[str, list[int]] = _build_keyword_index(self._documents)

    @staticmethod
    def _load_documents(file_path: str) -> list[str | dict[str, Any]]:
        try:
            path = Path(file_path)
            if not path.exists():
                return []
            if path.suffix == ".json":
                with open(path) as f:
                    return cast(
                        list[str | dict[str, Any]], _parse_json_docs(json.load(f))
                    )
            if path.suffix in {".csv", ".txt"}:
                with open(path) as f:
                    return cast(
                        list[str | dict[str, Any]],
                        [{"text": line.strip()} for line in f if line.strip()],
                    )
        except Exception:
            pass
        return []

    async def search(self, query: str, limit: int = 5) -> list[DataSourceResult]:
        return _keyword_search(query, self._documents, self._index, self.name, limit)


class URLDataSource(DataSource):
    def __init__(
        self,
        urls: list[str],
        name: str = "url_source",
        enabled: bool = True,
        priority: int = 0,
        cache_ttl: int = 3600,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(name=name, enabled=enabled, priority=priority)
        self.urls = urls
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._documents: list[str | dict[str, Any]] = []
        self._index: dict[str, list[int]] = {}
        self._fetched = False
        self._fetch_lock = asyncio.Lock()

    async def search(self, query: str, limit: int = 5) -> list[DataSourceResult]:
        if not self._fetched:
            async with self._fetch_lock:
                if not self._fetched:
                    self._documents = await self._fetch_all()
                    self._index = _build_keyword_index(self._documents)
                    self._fetched = True
        return _keyword_search(query, self._documents, self._index, self.name, limit)

    async def _fetch_all(self) -> list[str | dict[str, Any]]:
        if not _AIOHTTP_AVAILABLE:
            return []
        results = await asyncio.gather(
            *[self._fetch_url(u) for u in self.urls], return_exceptions=True
        )
        return [doc for batch in results if isinstance(batch, list) for doc in batch]

    async def _fetch_url(self, url: str) -> list[str | dict[str, Any]]:
        try:
            import aiohttp  # type: ignore[import-untyped]

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_content(content)
        except Exception:
            pass
        return []

    @staticmethod
    def _parse_content(content: str) -> list[str | dict[str, Any]]:
        try:
            return cast(
                list[str | dict[str, Any]], _parse_json_docs(json.loads(content))
            )
        except json.JSONDecodeError:
            return cast(list[str | dict[str, Any]], [{"text": content}])


class ContextDocumentDataSource(DataSource):
    def __init__(
        self,
        documents: list[dict[str, Any]],
        name: str = "context_documents",
        enabled: bool = True,
        priority: int = 100,
    ) -> None:
        super().__init__(name=name, enabled=enabled, priority=priority)
        self._documents: list[str | dict[str, Any]] = cast(
            list[str | dict[str, Any]], documents
        )
        self._index = _build_keyword_index(self._documents)

    async def search(self, query: str, limit: int = 5) -> list[DataSourceResult]:
        return _keyword_search(query, self._documents, self._index, self.name, limit)


class VectorStoreDataSource(DataSource):
    def __init__(
        self,
        provider: str,
        name: str = "",
        enabled: bool = True,
        priority: int = 0,
        client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        **config: Any,
    ) -> None:
        super().__init__(
            name=name or f"vectorstore_{provider}", enabled=enabled, priority=priority
        )
        self.provider = provider
        self.config = config
        self.client: Optional[Any] = (
            client if client is not None else self._initialize_client(provider, config)
        )
        self.embedding_model: Optional[Any] = (
            embedding_model
            if embedding_model is not None
            else self._load_embedding_model()
        )

    @staticmethod
    def _initialize_client(provider: str, config: dict[str, Any]) -> Optional[Any]:
        if provider == "pinecone":
            try:
                import pinecone  # type: ignore[import-untyped]

                api_key = config.get("api_key")
                index_name = config.get("index_name")
                if api_key and index_name:
                    return pinecone.Index(index_name)  # type: ignore[no-untyped-call]
            except ImportError:
                pass
        elif provider == "weaviate":
            try:
                import weaviate  # type: ignore[import-untyped]

                url = config.get("url")
                if url:
                    return weaviate.Client(url)  # type: ignore[no-untyped-call]
            except ImportError:
                pass
        return None

    async def search(self, query: str, limit: int = 5) -> list[DataSourceResult]:
        if not self.client:
            return []
        try:
            embedding = await self._get_embedding(query)
            if not embedding:
                return []
            if self.provider == "pinecone":
                results = self.client.query(
                    embedding, top_k=limit, include_metadata=True
                )
                return [
                    DataSourceResult(
                        text=match.get("metadata", {}).get("text", ""),
                        source=self.name,
                        confidence=float(match.get("score", 0.5)),
                    )
                    for match in results.get("matches", [])
                ]
            if self.provider == "weaviate":
                response = (
                    self.client.query.get(self.config.get("collection", "Document"))
                    .with_near_vector({"vector": embedding})
                    .with_limit(limit)
                    .do()
                )
                docs = response.get("data", {}).get("Get", {})
                return [
                    DataSourceResult(
                        text=doc.get("text", ""), source=self.name, confidence=0.8
                    )
                    for doc in docs
                ]
        except Exception:
            pass
        return []

    @staticmethod
    def _load_embedding_model() -> Optional[Any]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            return SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            return None

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        if self.embedding_model is None:
            return None
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


class KnowledgeGraphDataSource(DataSource):
    def __init__(
        self,
        endpoint: str = "https://query.wikidata.org/sparql",
        name: str = "knowledge_graph",
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        super().__init__(name=name, enabled=enabled, priority=priority)
        self.endpoint = endpoint

    async def search(self, query: str, limit: int = 5) -> list[DataSourceResult]:
        if not _AIOHTTP_AVAILABLE:
            return []
        sparql_query = self._build_sparql_query(query)
        try:
            import aiohttp  # type: ignore[import-untyped]

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.endpoint,
                    params={"query": sparql_query, "format": "json"},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            DataSourceResult(
                                text=text, source=self.name, confidence=0.9
                            )
                            for binding in data.get("results", {}).get("bindings", [])[
                                :limit
                            ]
                            for text in (self._extract_text(binding),)
                            if text
                        ]
        except Exception:
            pass
        return []

    @staticmethod
    def _build_sparql_query(query: str) -> str:
        safe_query = re.sub(r"[^a-zA-Z0-9 \-]", "", query)[:128]
        return f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item rdfs:label "{safe_query}"@en .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        LIMIT 10
        """

    @staticmethod
    def _extract_text(binding: dict[str, Any]) -> str:
        for key in ("itemLabel", "result", "label"):
            if key in binding:
                return binding[key].get("value", "")
        return ""


class FactCheckDataSource(DataSource):
    def __init__(
        self,
        provider: str = "snopes",
        name: str = "",
        enabled: bool = True,
        priority: int = 0,
        api_key: Optional[str] = None,
        **config: Any,
    ) -> None:
        super().__init__(
            name=name or f"factcheck_{provider}", enabled=enabled, priority=priority
        )
        self.provider = provider
        self.api_key = api_key
        self.config = config

    async def search(
        self, query: str, limit: int = 5
    ) -> list[DataSourceResult]:  # noqa: ARG002
        return []
