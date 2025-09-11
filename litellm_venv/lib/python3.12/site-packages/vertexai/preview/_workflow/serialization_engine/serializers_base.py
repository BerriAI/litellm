# -*- coding: utf-8 -*-

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=line-too-long, bad-continuation,protected-access
"""Defines the Serializer classes."""

import abc
import dataclasses
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from google.cloud.aiplatform.utils import gcs_utils
from vertexai.preview._workflow.shared import data_structures

T = TypeVar("T")
SERIALIZATION_METADATA_FILENAME = "serialization_metadata"
SERIALIZATION_METADATA_SERIALIZER_KEY = "serializer"
SERIALIZATION_METADATA_DEPENDENCIES_KEY = "dependencies"
SERIALIZATION_METADATA_CUSTOM_COMMANDS_KEY = "custom_commands"


SerializerArgs = data_structures.IdAsKeyDict


@dataclasses.dataclass
class SerializationMetadata:
    """Metadata of Serializer classes.

    This is supposed to be a class attribute named `_metadata` of the Serializer
    class.

    Example Usage:
        ```
        import vertexai

        # define a custom Serializer
        class KerasCustomSerializer(
                vertexai.preview.developer.Serializer):
            # make a metadata
            _metadata = vertexai.preview.developer.SerializationMetadata()

            def serialize(self, to_serialize, gcs_path):
                ...
            def deserialize(self, gcs_path):
                ...
        ```
    """

    serializer: Optional[str] = None
    dependencies: List[str] = dataclasses.field(default_factory=list)
    custom_commands: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self):
        return {
            SERIALIZATION_METADATA_SERIALIZER_KEY: self.serializer,
            SERIALIZATION_METADATA_DEPENDENCIES_KEY: self.dependencies,
            SERIALIZATION_METADATA_CUSTOM_COMMANDS_KEY: self.custom_commands,
        }

    def to_jsonable_dict(self):
        return self.to_dict()


class SerializationError(Exception):
    """Raised when the object fails to be serialized."""

    pass


def write_and_upload_data(data: bytes, gcs_filename: str):
    """Writes data to a local temp file and uploads the file to gcs.

    Args:
        data (bytes):
            Required. Bytes data to write.
        gcs_filename (str):
            Required. A gcs file path.
    """
    if gcs_filename.startswith("gs://"):
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(data)
            temp_file.flush()
            temp_file.seek(0)

            gcs_utils.upload_to_gcs(temp_file.name, gcs_filename)
    else:
        dirname = os.path.dirname(gcs_filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(gcs_filename, mode="wb") as f:
            f.write(data)


def _get_uri_prefix(gcs_uri: str) -> str:
    """Gets the directory of the gcs_uri.

    Example:
      1) file uri:
        _get_uri_prefix("gs://<bucket>/directory/file.extension") == "gs://
        <bucket>/directory/"
      2) folder uri:
        _get_uri_prefix("gs://<bucket>/parent_dir/dir") == "gs://<bucket>/
        parent_dir/"
    Args:
        gcs_uri: A string starting with "gs://" that represent a gcs uri.
    Returns:
        The parent gcs directory in string format.
    """
    # For tensorflow, the uri may be "gs://my-bucket/saved_model/"
    if gcs_uri.endswith("/"):
        gcs_uri = gcs_uri[:-1]
    gcs_pathlibpath = pathlib.Path(gcs_uri)
    file_name = gcs_pathlibpath.name
    return gcs_uri[: -len(file_name)]


def _get_metadata_path_from_file_gcs_uri(gcs_uri: str) -> str:
    gcs_pathlibpath = pathlib.Path(gcs_uri)
    prefix = _get_uri_prefix(gcs_uri=gcs_uri)
    return os.path.join(
        prefix,
        f"{SERIALIZATION_METADATA_FILENAME}_{gcs_pathlibpath.stem}.json",
    )


def _get_custom_serializer_path_from_file_gcs_uri(
    gcs_uri: str, serializer_name: str
) -> str:
    prefix = _get_uri_prefix(gcs_uri=gcs_uri)
    return os.path.join(prefix, f"{serializer_name}")


class Serializer(metaclass=abc.ABCMeta):
    """Abstract class of serializers.

    custom Serializers should be subclasses of this class.
    Example Usage:

        ```
        import vertexai

        # define a custom Serializer
        class KerasCustomSerializer(
                vertexai.preview.developer.Serializer):
            _metadata = vertexai.preview.developer.SerializationMetadata()

            def serialize(self, to_serialize, gcs_path):
                ...
            def deserialize(self, gcs_path):
                ...

        KerasCustomSerializer.register_requirements(
                ['library1==1.0.0', 'library2<2.0'])
        vertexai.preview.developer.register_serializer(
                keras.models.Model, KerasCustomSerializer)
        ```
    """

    _serialization_scheme: Dict[Type[Any], Optional[Type["Serializer"]]] = {}
    _custom_serialization_scheme: Dict[Type[Any], Optional[Type["Serializer"]]] = {}
    # _instances holds the instance of each Serializer for each type.
    _instances: Dict[Type["Serializer"], "Serializer"] = {}
    _metadata: SerializationMetadata = SerializationMetadata()

    def __new__(cls):
        try:
            import cloudpickle  # noqa:F401
        except ImportError as e:
            raise ImportError(
                "cloudpickle is not installed. Please call `pip install google-cloud-aiplatform[preview]`."
            ) from e

        if cls not in Serializer._instances:
            Serializer._instances[cls] = super().__new__(cls)
            if cls._metadata.serializer is None:
                cls._metadata.serializer = cls.__name__
        return Serializer._instances[cls]

    @abc.abstractmethod
    def serialize(
        self,
        to_serialize: T,
        gcs_path: str,
        **kwargs,
    ) -> Union[Dict[str, Any], str]:  # pytype: disable=invalid-annotation
        raise NotImplementedError

    @abc.abstractmethod
    def deserialize(
        self,
        serialized_gcs_path: str,
        **kwargs,
    ) -> T:  # pytype: disable=invalid-annotation
        raise NotImplementedError

    @classmethod
    def _register(
        cls, to_serialize_type: Type[Any], serializer_cls: Type["Serializer"]
    ):
        cls._serialization_scheme[to_serialize_type] = serializer_cls

    @classmethod
    def register_custom(
        cls, to_serialize_type: Type[Any], serializer_cls: Type["Serializer"]
    ):
        """Registers a custom serializer for a specific type.

        Example Usage:
            ```
            # define a custom Serializer
            class KerasCustomSerializer(serialization_engine.Serializer):
                _metadata = serialization_engine.SerializationMetadata()
                def serialize(self, to_serialize, gcs_path):
                    ...
                def deserialize(self, gcs_path):
                    ...

            any_serializer = serialization_engine.AnySerializer()
            any_serializer.register_custom(keras.models.Model, KerasCustomSerializer)
            ```
        Args:
            to_serialize_type: The class that is supposed to be serialized with
                the to-be-registered custom Serializer.
            serializer_cls: The custom Serializer to be registered.
        """
        cls._custom_serialization_scheme[to_serialize_type] = serializer_cls

    @classmethod
    def get_instance(cls) -> "Serializer":
        if cls not in Serializer._instances:
            Serializer._instances[cls] = cls()
        return Serializer._instances[cls]

    @classmethod
    def _dedupe_deps(cls):
        # TODO(b/282719450): Consider letting the later specifier to overwrite
        # earlier specifier for the same package, and automatically detecting
        # the version if version is not specified.
        cls._metadata.dependencies = list(dict.fromkeys(cls._metadata.dependencies))

    @classmethod
    def _dedupe_custom_commands(cls):
        cls._metadata.custom_commands = list(
            dict.fromkeys(cls._metadata.custom_commands)
        )

    @classmethod
    def register_requirement(cls, required_package: str):
        # TODO(b/280648121) Consider allowing the user to register the
        # installation command so that we support installing packages not
        # covered by PyPI in the remote machine.
        cls._metadata.dependencies.append(required_package)
        cls._dedupe_deps()

    @classmethod
    def register_requirements(cls, requirements: List[str]):
        cls._metadata.dependencies.extend(requirements)
        cls._dedupe_deps()

    @classmethod
    def register_custom_command(cls, custom_command: str):
        cls._metadata.custom_commands.append(custom_command)
        cls._dedupe_custom_commands()
