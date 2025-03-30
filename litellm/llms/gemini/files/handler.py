"""
Supports writing files to Google AI Studio Files API.

For vertex ai, check out the vertex_ai/files/handler.py file.
"""
from litellm.llms.base_llm.files.transformation import BaseFilesConfig
from litellm.types.llms.openai import CreateFileRequest, FileObject


class GoogleAIStudioFilesHandler(BaseFilesConfig):
    def __init__(self):
        pass

    def create_file(
        self, create_file_data: CreateFileRequest, litellm_params: dict
    ) -> FileObject:
        raise NotImplementedError("Google AI Studio Files API is not yet implemented")

    def list_files(self):
        pass

    def delete_file(self):
        pass

    def retrieve_file(self):
        pass

    def retrieve_file_content(self):
        pass
