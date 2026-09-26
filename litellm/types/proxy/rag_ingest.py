"""``general_settings.rag_ingest``: the controls run on bytes before they land in a vector store."""

from pydantic import BaseModel, ConfigDict, Field


class RagIngestSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    malware_scanner: str | None = Field(
        default=None,
        description=(
            "The scanner every vector-store upload goes through, as <module>.<instance> where the module sits "
            "next to config.yaml or is importable, resolved the way custom_auth is. The instance must expose "
            "scan(content: bytes) -> ScanResult and be safe to call from several threads at once. Unset runs "
            "the EICAR test scanner, which flags only the EICAR test file"
        ),
    )
    files_api_controls: bool = Field(
        default=True,
        description=(
            "Whether /v1/files uploads with purpose assistants or user_data run the upload controls (format, size "
            "and malware checks) before reaching the provider. false lets those uploads through unchecked, for a "
            "deployment that sends the provider a format the controls reject on that route, such as DOCX or "
            "images. /v1/rag/ingest runs the controls whatever this is set to"
        ),
    )
