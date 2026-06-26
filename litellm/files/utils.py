from typing import Optional

from litellm.types.llms.openai import CreateFileRequest
from litellm.types.utils import ExtractedFileData

# MIME types a .jsonl batch upload is plausibly labeled with. Clients are
# inconsistent (text/plain, application/json, octet-stream, ndjson, ...), so a
# batch file must not silently bypass the streaming path just because of its
# declared type. ``purpose == "batch"`` is the authoritative signal; non-JSONL
# content still fails loudly when the rows are parsed.
_BATCH_JSONL_CONTENT_TYPES = frozenset(
    {
        "application/jsonl",
        "application/json",
        "application/octet-stream",
        "application/x-ndjson",
        "application/x-jsonlines",
        "text/plain",
    }
)


class FilesAPIUtils:
    """
    Utils for files API interface on litellm
    """

    @staticmethod
    def is_batch_jsonl_file(
        create_file_data: CreateFileRequest, extracted_file_data: ExtractedFileData
    ) -> bool:
        """
        Check if the file is a batch jsonl file
        """
        return (
            create_file_data.get("purpose") == "batch"
            and FilesAPIUtils.valid_content_type(
                extracted_file_data.get("content_type")
            )
            and extracted_file_data.get("content") is not None
        )

    @staticmethod
    def is_batch_jsonl_request(
        create_file_data: CreateFileRequest, content_type: Optional[str]
    ) -> bool:
        """
        Batch-jsonl check from metadata only, so the body can stay a streamable
        Path/handle instead of being read into memory.
        """
        return (
            create_file_data.get("purpose") == "batch"
            and FilesAPIUtils.valid_content_type(content_type)
            and create_file_data.get("file") is not None
        )

    @staticmethod
    def valid_content_type(content_type: Optional[str]) -> bool:
        """
        Whether the upload's MIME type is one a batch JSONL file is plausibly
        sent as (see ``_BATCH_JSONL_CONTENT_TYPES``).
        """
        return content_type in _BATCH_JSONL_CONTENT_TYPES
