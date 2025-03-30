from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from litellm.types.llms.openai import CreateFileRequest, FileObject

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseFilesConfig(ABC):
    @abstractmethod
    def create_file(
        self, create_file_data: CreateFileRequest, litellm_params: dict
    ) -> FileObject:
        """
        Creates a file
        """
        pass

    @abstractmethod
    def list_files(self):
        """
        Lists all files
        """
        pass

    @abstractmethod
    def delete_file(self):
        """
        Deletes a file
        """
        pass

    @abstractmethod
    def retrieve_file(self):
        """
        Returns the metadata of the file
        """
        pass

    @abstractmethod
    def retrieve_file_content(self):
        """
        Returns the content of the file
        """
        pass
