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
import collections
import dataclasses
import importlib
import json
import os
import sys
import tempfile
from typing import Any, Dict, Union, List, TypeVar, Type, Optional

from google.cloud.aiplatform import base
from google.cloud.aiplatform.utils import gcs_utils
from vertexai.preview._workflow.serialization_engine import (
    serializers,
    serializers_base,
)
from vertexai.preview._workflow.shared import (
    supported_frameworks,
)

from packaging import requirements


T = TypeVar("T")

_LOGGER = base.Logger("vertexai.serialization_engine")

SERIALIZATION_METADATA_SERIALIZER_KEY = "serializer"
SERIALIZATION_METADATA_DEPENDENCIES_KEY = "dependencies"
SERIALIZATION_ARGS_DIRNAME = "serialization_args"
GLOBAL_SERIALIZATION_METADATA = "global_serialization_metadata.json"

_LIGHTNING_ROOT_DIR = "/vertex_lightning_root_dir/"
_JSONABLE_TYPES = Union[int, float, bytes, bool, str, None]

# This is a collection of all the predefined serializers and the fully qualified
# class names that these serializers are intended to be used on.
_PREDEFINED_SERIALIZERS = frozenset(
    [
        ("sklearn.base.BaseEstimator", serializers.SklearnEstimatorSerializer),
        ("tensorflow.keras.models.Model", serializers.KerasModelSerializer),
        (
            "tensorflow.keras.callbacks.History",
            serializers.KerasHistoryCallbackSerializer,
        ),
        ("tensorflow.data.Dataset", serializers.TFDatasetSerializer),
        ("torch.nn.Module", serializers.TorchModelSerializer),
        ("torch.utils.data.DataLoader", serializers.TorchDataLoaderSerializer),
        ("lightning.pytorch.Trainer", serializers.LightningTrainerSerializer),
        ("bigframes.dataframe.DataFrame", serializers.BigframeSerializer),
        ("pandas.DataFrame", serializers.PandasDataSerializer),
    ]
)


def get_arg_path_from_file_gcs_uri(gcs_uri: str, arg_name: str) -> str:
    """Gets the argument gcs path from the to-be-serialized object's gcs uri."""
    # TODO(b/306392189): add an intermediate directory to differentiate
    # arguments for different objects.
    prefix = serializers.get_uri_prefix(gcs_uri=gcs_uri)
    return os.path.join(
        prefix,
        SERIALIZATION_ARGS_DIRNAME,
        arg_name,
    )


def _is_the_same_gcs_path(gcs_path_form1, gcs_path_form2) -> bool:
    if gcs_path_form1 in (
        gcs_path_form2,
        gcs_path_form2.replace("gs://", "/gcs/"),
        gcs_path_form2.replace("/gcs/", "gs://"),
    ):
        return True
    return False


@dataclasses.dataclass
class SerializerArg:
    value: _JSONABLE_TYPES = None
    gcs_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        if d.get("value", None) is not None and d.get("gcs_path", None) is not None:
            raise ValueError("Only one of value or gcs_path should be provided.")
        value = d.get("value", None)
        if sys.version_info < (3, 10):
            # in Python <=3.9, we couldn't use subscriptable generics for instance
            # checks.
            if value is not None and type(value) not in (int, float, bytes, bool, str):
                raise ValueError(
                    "Only string, int, float, bool, bytes and None are supported "
                    f"while a {type(value)} {value} is provided."
                )
        else:
            if value is not None and not isinstance(value, _JSONABLE_TYPES):
                raise ValueError(
                    "Only string, int, float, bool, bytes and None are supported "
                    f"while a {type(value)} {value} is provided."
                )
        return cls(value, d.get("gcs_path", None))

    def to_dict(self):
        return {"value": self.value, "gcs_path": self.gcs_path}

    def to_jsonable_dict(self):
        return self.to_dict()


@dataclasses.dataclass
class SerializedEntryMetadata:
    # TODO(b/307272556): consider deprecate either serialization_id or obj.
    serialization_id: str
    serializer_args: Dict[str, SerializerArg] = dataclasses.field(
        default_factory=collections.defaultdict
    )
    obj: Any = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            d.get("serialization_id", None),
            {
                key: SerializerArg.from_dict(value)
                for key, value in d["serializer_args"].items()
            },
            d.get("obj", None),
        )

    def to_dict(self):
        return {
            "serialization_id": self.serialization_id,
            "serializer_args": {
                key: value.to_dict() for key, value in self.serializer_args.items()
            },
            "obj": self.obj,
        }

    def to_jsonable_dict(self):
        # We'll not save the object to jsonized data
        return {
            "serialization_id": self.serialization_id,
            "serializer_args": {
                key: value.to_jsonable_dict()
                for key, value in self.serializer_args.items()
            },
        }


class SerializedDict(dict):
    """A dict that ensures all the gcs_path keys are starting with gs://"""

    def __getitem__(self, __key, /):
        if __key.startswith("/gcs/") and __key in self.keys():
            value = super().__getitem__(__key)
            new_key = __key.replace("/gcs/", "gs://")
            super().__setitem__(new_key, value)
            super().__delitem__(__key)
            return super().__getitem__(new_key)
        elif __key.startswith("/gcs/"):
            value = super().__getitem__(__key.replace("/gcs/", "gs://"))
            return value
        return super().__getitem__(__key)

    def __setitem__(self, __key, __value, /):
        if __key.startswith("/gcs/"):
            super().__setitem__(__key.replace("/gcs/", "gs://"), __value)
        super().__setitem__(__key, __value)

    def __delitem__(self, __key, /):
        if __key.startswith("/gcs/") and __key not in self.keys():
            super().__delitem__(__key.replace("/gcs/", "gs://"))
        super().__delitem__(__key)

    def get(self, key, default=None):
        new_key = key.replace("/gcs/", "gs://")
        return super().get(new_key, default)


@dataclasses.dataclass
class AnySerializationMetadata(serializers_base.SerializationMetadata):
    """Metadata of AnySerializer class."""

    # serialized is a dict from the gcs path of the serialized to its serialization metadata
    serialized: SerializedDict = dataclasses.field(default_factory=SerializedDict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(
            serializer=d.get("serializer", None),
            dependencies=d.get("dependencies", None),
            serialized=SerializedDict(
                {
                    key: SerializedEntryMetadata.from_dict(value)
                    for key, value in d["serialized"].items()
                }
            ),
        )

    def to_dict(self):
        dct = super().to_dict()
        dct.update(
            {
                "serialized": {
                    key: value.to_dict() for key, value in self.serialized.items()
                }
            }
        )
        return dct

    def to_jsonable_dict(self):
        dct = super().to_jsonable_dict()
        dct.update(
            {
                "serialized": {
                    key: value.to_jsonable_dict()
                    for key, value in self.serialized.items()
                }
            }
        )
        return dct


def _check_dependency_versions(required_packages: List[str]):
    for package in required_packages:
        requirement = requirements.Requirement(package)
        package_name = requirement.name
        current_version = supported_frameworks._get_version_for_package(package_name)
        if not requirement.specifier.contains(current_version):
            _LOGGER.warning(
                "%s's version is %s, while the required version is %s",
                package_name,
                current_version,
                requirement.specifier,
            )


def _get_custom_serializer_path_from_file_gcs_uri(
    gcs_uri: str, serializer_name: str
) -> str:
    prefix = serializers.get_uri_prefix(gcs_uri=gcs_uri)
    return os.path.join(prefix, f"{serializer_name}")


class AnySerializer(serializers_base.Serializer):
    """A serializer that can routes any object to their own serializer."""

    _metadata: AnySerializationMetadata = AnySerializationMetadata(
        serializer="AnySerializer"
    )

    def __init__(self):
        super().__init__()
        # Register with default serializers
        AnySerializer._register(object, serializers.CloudPickleSerializer)

        for args in _PREDEFINED_SERIALIZERS:
            AnySerializer._register_predefined_serializer(*args)

    @classmethod
    def _get_custom_serializer(cls, type_cls):
        return cls._custom_serialization_scheme.get(type_cls)

    @classmethod
    def _get_predefined_serializer(cls, type_cls):
        return cls._serialization_scheme.get(type_cls)

    @classmethod
    def _register_predefined_serializer(
        cls,
        full_class_name: str,
        serializer: serializers_base.Serializer,
    ):
        """Registers a predefined serializer to AnySerializer."""
        try:
            module_name, class_name = full_class_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            to_serialize_class = getattr(module, class_name)

            AnySerializer._register(to_serialize_class, serializer)
            _LOGGER.debug(f"Successfully registered {serializer}")

        except Exception as e:
            _LOGGER.debug(f"Failed to register {serializer} due to: {e}")

    def _gcs_path_in_metadata(self, obj) -> Optional[str]:
        """Checks if an object has been (de-)serialized before."""
        for key, value in self._metadata.serialized.items():
            if obj is value.obj:
                return key

    def _update_metadata_for_obj(
        self,
        to_serialize: T,
        new_gcs_path: str,
        serializer_args: Optional[Dict[str, SerializerArg]] = None,
    ):
        for key, value in self._metadata.serialized.items():
            if to_serialize is value.obj and not _is_the_same_gcs_path(
                key, new_gcs_path
            ):
                self._metadata.serialized[new_gcs_path] = value
                del self._metadata.serialized[key]
                return

        new_value = SerializedEntryMetadata(
            serialization_id=id(to_serialize),
            serializer_args=serializer_args,
            obj=to_serialize,
        )

        self._metadata.serialized[new_gcs_path] = new_value

    def save_global_metadata(self, gcs_path: str):
        """Saves the current global metadata to the specified gcs_path."""
        if gcs_path.startswith("gs://"):
            with tempfile.NamedTemporaryFile(mode="wt") as temp_file:
                json.dump(self._metadata.to_jsonable_dict(), temp_file)
                temp_file.flush()
                temp_file.seek(0)

                gcs_utils.upload_to_gcs(temp_file.name, gcs_path)
        else:
            # In distributed training, one worker could have written this global
            # dataset and keep opening it will raise FileExistsError.
            # TODO(b/306434083): Find the right error type to catch and put the
            # `with open` in a try clause. This is because, even with the
            # os.path.exists() check, it can still happen that during the check,
            # the file doesn't exist but it exists while we are writing.
            if os.path.exists(gcs_path):
                _LOGGER.info("%s already exists, returning", gcs_path)
                return
            try:
                with open(gcs_path, "w") as f:
                    json.dump(self._metadata.to_jsonable_dict(), f)
            except Exception as e:
                _LOGGER.warning(
                    "Failed to save global metadata to %s due to error %s", gcs_path, e
                )

    def load_global_metadata(self, gcs_path: str) -> Dict[str, Any]:
        """Loads the current global metadata from the specified gcs_path."""
        if gcs_path.startswith("gs://"):
            with tempfile.NamedTemporaryFile() as temp_file:
                gcs_utils.download_file_from_gcs(gcs_path, temp_file.name)
                with open(temp_file.name, mode="rb") as f:
                    metadata = json.load(f)
        else:
            with open(gcs_path, "rb") as f:
                metadata = json.load(f)

        self._metadata = AnySerializationMetadata.from_dict(metadata)

    def serialize(self, to_serialize: T, gcs_path: str, **kwargs) -> Dict[str, Any]:
        """Simplified version of serialize()."""
        metadata_path = serializers.get_metadata_path_from_file_gcs_uri(gcs_path)
        gcs_path_in_metadata = self._gcs_path_in_metadata(to_serialize)
        # The object has been serialized, this likely happens when this code
        # is run on the remote side (CustomJob)
        if gcs_path_in_metadata and not kwargs:
            serializer_args = self._metadata.serialized[
                gcs_path_in_metadata
            ].serializer_args
        else:
            serializer_args = kwargs.copy()
        _LOGGER.debug("serializer_args is %s", serializer_args)

        for i, step_type in enumerate(
            to_serialize.__class__.__mro__ + to_serialize.__class__.__mro__
        ):
            # Iterate through the custom serialization scheme first.
            if (
                i < len(to_serialize.__class__.__mro__)
                and step_type not in AnySerializer._custom_serialization_scheme
            ) or (
                i >= len(to_serialize.__class__.__mro__)
                and step_type not in AnySerializer._serialization_scheme
            ):
                continue
            elif i < len(to_serialize.__class__.__mro__):
                serializer = AnySerializer._get_custom_serializer(
                    step_type
                ).get_instance()  # pytype: disable=attribute-error
                # If the Serializer is a custom Serializer, serialize the
                # Custom Serializer first.
                serializer_path = _get_custom_serializer_path_from_file_gcs_uri(
                    gcs_path, serializer.__class__.__name__
                )
                serializers.CloudPickleSerializer().serialize(
                    serializer, serializer_path
                )
            else:
                serializer = AnySerializer._get_predefined_serializer(
                    step_type
                ).get_instance()

            try:
                # Sometimes the returned gcs_path can be different from the
                # passed-in gcs_path. The serialize() could add a suffix, for
                # example.
                gcs_path_returned = serializer.serialize(
                    to_serialize=to_serialize, gcs_path=gcs_path, **serializer_args
                )
                # Don't fail if the gcs_path_returned is None, we'll keep using
                # the original gcs_path.
                gcs_path = gcs_path_returned or gcs_path
            except Exception as e:  # pylint: disable=broad-exception-caught
                if serializer.__class__.__name__ != "CloudPickleSerializer":
                    _LOGGER.warning(
                        "Failed to serialize %s with %s due to error %s",
                        to_serialize.__class__.__name__,
                        serializer.__class__.__name__,
                        e,
                    )
                    # Falling back to Serializers of super classes
                    continue
                else:
                    raise serializers_base.SerializationError from e

            local_metadata = serializer._metadata.to_dict()
            serializers_base.write_and_upload_data(
                json.dumps(local_metadata).encode(), metadata_path
            )

            # Serialize the parameters if needed.
            # TODO(b/296584472): remove the iteration once the serialization of
            # nested objects can be automatically detected.
            for arg_name, arg_value in kwargs.items():
                if type(arg_value) not in (int, float, bool, bytes, str, list, dict):
                    arg_serialized_gcs_path = get_arg_path_from_file_gcs_uri(
                        gcs_path, arg_name
                    )
                    self.serialize(arg_value, arg_serialized_gcs_path)
                    serializer_args[arg_name] = SerializerArg(
                        gcs_path=arg_serialized_gcs_path
                    )
                else:
                    serializer_args[arg_name] = SerializerArg(value=arg_value)

            self._update_metadata_for_obj(
                to_serialize, gcs_path, serializer_args=serializer_args
            )

            return local_metadata

    def deserialize(self, serialized_gcs_path: str, **kwargs) -> T:
        """Routes the corresponding Serializer based on the metadata."""
        _LOGGER.debug("deserializing from %s.", serialized_gcs_path)
        # Note: do not use "in" to check the key. Use "get()".
        # This is because the "serialized" field is not of the built-in dict
        # type.
        if self._metadata.serialized.get(serialized_gcs_path, None) is None:
            _LOGGER.warning(
                "gcs_path %s not found in the metadata. "
                "Make sure global serialization metadata is loaded.",
                serialized_gcs_path,
            )
            serializer_args = {}
        else:
            serializer_args = self._metadata.serialized[
                serialized_gcs_path
            ].serializer_args

        for arg_name, serializer_arg in serializer_args.items():
            if serializer_arg.value is not None:
                kwargs[arg_name] = serializer_arg.value
            else:
                kwargs[arg_name] = self.deserialize(
                    serialized_gcs_path=serializer_arg.gcs_path
                )

        local_metadata = serializers._get_metadata(serialized_gcs_path)

        _LOGGER.debug(
            "deserializing from %s, metadata is %s", serialized_gcs_path, local_metadata
        )

        serializer_cls_name = local_metadata[SERIALIZATION_METADATA_SERIALIZER_KEY]
        packages = local_metadata[SERIALIZATION_METADATA_DEPENDENCIES_KEY]
        _check_dependency_versions(packages)
        serializer_class = getattr(
            serializers, serializer_cls_name, None
        ) or globals().get(serializer_cls_name)
        if not serializer_class:
            # Serializer is an unregistered custom Serializer.
            # Deserialize serializer.
            serializer_path = _get_custom_serializer_path_from_file_gcs_uri(
                serialized_gcs_path, serializer_cls_name
            )
            serializer = serializers.CloudPickleSerializer().deserialize(
                serialized_gcs_path=serializer_path
            )
        else:
            serializer = serializer_class.get_instance()

        for key, value in local_metadata.items():
            setattr(serializer.__class__._metadata, key, value)

        _LOGGER.debug(
            "using serializer %s to deserialize from path %s, w/ kwargs %s",
            serializer.__class__.__name__,
            serialized_gcs_path,
            kwargs,
        )
        obj = serializer.deserialize(serialized_gcs_path=serialized_gcs_path, **kwargs)
        if not serializer_class:
            # Register the serializer
            AnySerializer.register_custom(obj.__class__, serializer.__class__)
            AnySerializer._instances[serializer.__class__] = serializer
        if (
            self._metadata.serialized.get(serialized_gcs_path, None) is not None
        ):  # don't use "in"
            self._metadata.serialized[serialized_gcs_path].obj = obj
        else:
            _LOGGER.warning(
                "the gcs_path %s doesn't exist in the metadata."
                " Please make sure the global metadata is loaded.",
                serialized_gcs_path,
            )
            self._metadata.serialized[serialized_gcs_path] = SerializedEntryMetadata(
                serialization_id=id(obj), obj=obj
            )
        return obj


def register_serializer(
    to_serialize_type: Type[Any], serializer_cls: Type[serializers_base.Serializer]
):
    """Registers a Serializer for a specific type.

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

    Args:
        to_serialize_type: The class that is supposed to be serialized with
            the to-be-registered custom Serializer.
        serializer_cls: The custom Serializer to be registered.
    """
    any_serializer = AnySerializer()
    any_serializer.register_custom(
        to_serialize_type=to_serialize_type, serializer_cls=serializer_cls
    )
