# SPDX-License-Identifier: MIT OR Apache-2.0
# Type-only SDK interface from patronus-ark-core 0.1.14; runtime SDK is installed separately.
"""Python shapes of Core configuration and serialized reports."""

from typing import Literal, NotRequired, TypedDict

class EvidenceSpan(TypedDict):
    label: str
    text: str
    score: float
    start_byte: int
    end_byte: int
    start_char: int
    end_char: int

class LayerResult(TypedDict):
    level: str
    layer_type: str
    class_name: str
    confidence: float
    matched: bool
    duration_ms: float
    thresholds: dict[str, float]
    details: dict[str, object]

class LabelScore(TypedDict):
    label: str
    confidence: float
    matched: bool

class NativeResult(TypedDict):
    category: Literal["pii", "dlp", "injection"]
    class_name: str
    confidence: float
    level: str
    model: str
    duration_ms: float
    layers: list[LayerResult]
    evidence_spans: list[EvidenceSpan]
    label_scores: list[LabelScore]
    decision: dict[str, object] | None

class WolfResult(TypedDict):
    document_logit: float
    attack_probability: float
    threshold: float
    matched: bool
    windows: int
    model_id: str
    revision: str
    precision: str

class ScanReport(TypedDict):
    input_bytes: int
    native: list[NativeResult]
    wolf: WolfResult | None

class OnnxRuntimeOptions(TypedDict, total=False):
    intra_threads: int | None
    inter_threads: int | None
    spinning: bool | None

class WolfConfig(TypedDict):
    bundle_dir: str
    threshold: float
    precision: NotRequired[Literal["auto", "fp16", "int8_int4"]]
    backend: NotRequired[
        Literal["auto", "cpu", "gpu", "coreml", "core_ml", "cuda", "direct_ml", "tensorrt", "tensor_rt"]
    ]
    runtime_options: NotRequired[OnnxRuntimeOptions]

class ScannerConfig(TypedDict, total=False):
    categories: list[Literal["pii", "dlp", "injection"]]
    rules: dict[str, bool]
    models: dict[str, bool]
    pii_mode: Literal["strict", "balanced", "recall"]
    dlp_mode: Literal["strict", "balanced", "recall"]
    explain: bool
    max_input_bytes: int
    max_findings: int
    wolf: WolfConfig | None

class WarmupOptions(TypedDict, total=False):
    download: Literal["never", "missing"]
    download_timeout_ms: int

class WolfExecutionOptions(TypedDict, total=False):
    window_tokens: int
    overlap_tokens: int
    batch_size: int
