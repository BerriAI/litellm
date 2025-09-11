from typing import Optional, Dict, Any, List, TypedDict


class InstrumentorContextData(TypedDict):
    trace_id: Optional[str]
    parent_observation_id: Optional[str]
    root_llama_index_span_id: Optional[str]
    is_user_managed_trace: Optional[bool]
    update_parent: Optional[bool]
    trace_name: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    version: Optional[str]
    release: Optional[str]
    metadata: Optional[Dict[str, Any]]
    tags: Optional[List[str]]
    public: Optional[bool]
