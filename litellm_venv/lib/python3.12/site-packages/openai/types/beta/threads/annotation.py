# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from ...._utils import PropertyInfo
from .file_path_annotation import FilePathAnnotation
from .file_citation_annotation import FileCitationAnnotation

__all__ = ["Annotation"]

Annotation: TypeAlias = Annotated[
    Union[FileCitationAnnotation, FilePathAnnotation],
    PropertyInfo(discriminator="type"),
]
