from typing import Final

ASSEMBLY_AI_POLLING_INTERVAL: Final = 10
ASSEMBLY_AI_MAX_POLLING_ATTEMPTS: Final = 180
ASSEMBLYAI_UPLOAD_ROUTES: Final = frozenset(
    (
        "/assemblyai/v2/upload",
        "/eu.assemblyai/v2/upload",
    )
)
