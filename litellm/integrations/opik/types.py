from typing import TypedDict, Dict, List

class OpikSpan(TypedDict, total=False):
    project_name: str
    trace_id: str
    parent_span_id: str
    name: str
    type: str
    start_time: str
    end_time: str
    input: Dict
    output: Dict
    metadata: Dict
    tags: List[str]
    usage: Dict
    

class OpikTrace(TypedDict, total=False):
    trace_id: str
    project_name: str
    name: str
    start_time: str
    end_time: str
    input: Dict
    output: Dict
    metadata: Dict
    tags: List[str]
