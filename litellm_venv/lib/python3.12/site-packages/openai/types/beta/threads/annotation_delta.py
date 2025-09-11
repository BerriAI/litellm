# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, TypeAlias

from ...._utils import PropertyInfo
from .file_path_delta_annotation import FilePathDeltaAnnotation
from .file_citation_delta_annotation import FileCitationDeltaAnnotation

__all__ = ["AnnotationDelta"]

AnnotationDelta: TypeAlias = Annotated[
    Union[FileCitationDeltaAnnotation, FilePathDeltaAnnotation],
    PropertyInfo(discriminator="type"),
]
