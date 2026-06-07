from typing import Optional

from litellm.types.llms.openai import CreateFileRequest
from litellm.types.utils import ExtractedFileData


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
        Check if the content type is valid
        """
        return content_type in set(["application/jsonl", "application/octet-stream"])
