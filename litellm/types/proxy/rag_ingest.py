"""``general_settings.rag_ingest``: the controls run on bytes before they land in a vector store."""

from pydantic import BaseModel, ConfigDict, Field


class RagIngestSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    malware_scanner: str | None = Field(
        None,
        description=(
            "The scanner every vector-store upload goes through, as <module>.<instance> where the module sits "
            "next to config.yaml or is importable, resolved the way custom_auth is. The instance must expose "
            "scan(content: bytes) -> ScanResult and be safe to call from several threads at once. Unset runs "
            "the EICAR test scanner, which flags only the EICAR test file"
        ),
    )
