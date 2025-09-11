# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Set
import collections


_ProtoModule = collections.namedtuple(
    "ProtoModule",
    ["package", "marshal", "manifest"],
)


def define_module(
    *, package: str, marshal: str = None, manifest: Set[str] = frozenset()
) -> _ProtoModule:
    """Define a protocol buffers module.

    The settings defined here are used for all protobuf messages
    declared in the module of the given name.

    Args:
        package (str): The proto package name.
        marshal (str): The name of the marshal to use. It is recommended
            to use one marshal per Python library (e.g. package on PyPI).
        manifest (Set[str]): A set of messages and enums to be created. Setting
            this adds a slight efficiency in piecing together proto
            descriptors under the hood.
    """
    if not marshal:
        marshal = package
    return _ProtoModule(
        package=package,
        marshal=marshal,
        manifest=frozenset(manifest),
    )


__all__ = ("define_module",)
