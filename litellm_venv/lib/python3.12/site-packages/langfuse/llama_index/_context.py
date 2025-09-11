from contextvars import ContextVar
from typing import Optional, Any, List, Dict
from ._types import InstrumentorContextData


default_context: InstrumentorContextData = {
    "trace_id": None,
    "parent_observation_id": None,
    "update_parent": True,
    "trace_name": None,
    "root_llama_index_span_id": None,
    "is_user_managed_trace": None,
    "user_id": None,
    "session_id": None,
    "version": None,
    "release": None,
    "metadata": None,
    "tags": None,
    "public": None,
}

langfuse_instrumentor_context: ContextVar[InstrumentorContextData] = ContextVar(
    "langfuse_instrumentor_context",
    default={**default_context},
    # The spread operator (**) is used here to create a new dictionary
    # that is a shallow copy of default_trace_attributes.
    # This ensures that each ContextVar instance gets its own copy of the default attributes,
    # preventing accidental shared state between different contexts.
    # If we didn't use the spread operator, all contexts would reference the same dictionary,
    # which could lead to unexpected behavior if the dictionary is modified.
)


class InstrumentorContext:
    @staticmethod
    def _get_context():
        return langfuse_instrumentor_context.get()

    @property
    def trace_id(self) -> Optional[str]:
        return self._get_context()["trace_id"]

    @property
    def parent_observation_id(self) -> Optional[str]:
        return self._get_context()["parent_observation_id"]

    @property
    def root_llama_index_span_id(self) -> Optional[str]:
        return self._get_context()["root_llama_index_span_id"]

    @property
    def is_user_managed_trace(self) -> Optional[bool]:
        return self._get_context()["is_user_managed_trace"]

    @property
    def update_parent(self) -> Optional[bool]:
        return self._get_context()["update_parent"]

    @property
    def trace_name(self) -> Optional[str]:
        return self._get_context()["trace_name"]

    @property
    def trace_data(self):
        return {
            "user_id": self._get_context()["user_id"],
            "session_id": self._get_context()["session_id"],
            "version": self._get_context()["version"],
            "release": self._get_context()["release"],
            "metadata": self._get_context()["metadata"],
            "tags": self._get_context()["tags"],
            "public": self._get_context()["public"],
        }

    @staticmethod
    def reset():
        langfuse_instrumentor_context.set({**default_context})

    def reset_trace_id(self):
        previous_context = self._get_context()

        langfuse_instrumentor_context.set(
            {**previous_context, "trace_id": None, "root_llama_index_span_id": None}
        )

    @staticmethod
    def update(
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        update_parent: Optional[bool] = None,
        root_llama_index_span_id: Optional[str] = None,
        is_user_managed_trace: Optional[bool] = None,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        public: Optional[bool] = None,
    ):
        updates = {}

        if trace_id is not None:
            updates["trace_id"] = trace_id
        if parent_observation_id is not None:
            updates["parent_observation_id"] = parent_observation_id
        if update_parent is not None:
            updates["update_parent"] = update_parent
        if trace_name is not None:
            updates["trace_name"] = trace_name
        if root_llama_index_span_id is not None:
            updates["root_llama_index_span_id"] = root_llama_index_span_id
        if is_user_managed_trace is not None:
            updates["is_user_managed_trace"] = is_user_managed_trace
        if user_id is not None:
            updates["user_id"] = user_id
        if session_id is not None:
            updates["session_id"] = session_id
        if version is not None:
            updates["version"] = version
        if release is not None:
            updates["release"] = release
        if metadata is not None:
            updates["metadata"] = metadata
        if tags is not None:
            updates["tags"] = tags
        if public is not None:
            updates["public"] = public

        langfuse_instrumentor_context.get().update(updates)
