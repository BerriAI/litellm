# Copyright 2018 Google LLC
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

import sys

from proto.marshal import Marshal


def compile(name, attrs):
    """Return the package and marshal to use.

    Args:
        name (str): The name of the new class, as sent to ``type.__new__``.
        attrs (Mapping[str, Any]): The attrs for a new class, as sent
            to ``type.__new__``

    Returns:
        Tuple[str, ~.Marshal]:
            - The proto package, if any (empty string otherwise).
            - The marshal object to use.
    """
    # Pull a reference to the module where this class is being
    # declared.
    module = sys.modules.get(attrs.get("__module__"))
    module_name = module.__name__ if hasattr(module, __name__) else ""
    proto_module = getattr(module, "__protobuf__", object())

    # A package should be present; get the marshal from there.
    # TODO: Revert to empty string as a package value after protobuf fix.
    # When package is empty, upb based protobuf fails with an
    # "TypeError: Couldn't build proto file into descriptor pool: invalid name: empty part ()' means"
    # during an attempt to add to descriptor pool.
    package = getattr(
        proto_module, "package", module_name if module_name else "_default_package"
    )
    marshal = Marshal(name=getattr(proto_module, "marshal", package))

    # Done; return the data.
    return (package, marshal)
