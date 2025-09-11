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


class IdAsKeyDict(dict):
    """Customized dict that maps each key to its id before storing the data.

    This subclass of dict still allows one to use the original key during
    subscription ([] operator) or via `get()` method. But under the hood, the
    keys are the ids of the original keys.

    Example:
        # add some hashable objects (key1 and key2) to the dict
        id_as_key_dict = IdAsKeyDict({key1: value1, key2: value2})
        # add a unhashable object (key3) to the dict
        id_as_key_dict[key3] = value3

        # can access the value via subscription using the original key
        assert id_as_key_dict[key1] == value1
        assert id_as_key_dict[key2] == value2
        assert id_as_key_dict[key3] == value3
        # can access the value via get method using the original key
        assert id_as_key_dict.get(key1) == value1
        assert id_as_key_dict.get(key2) == value2
        assert id_as_key_dict.get(key3) == value3
        # but the original keys are not in the dict - the ids are
        assert id(key1) in id_as_key_dict
        assert id(key2) in id_as_key_dict
        assert id(key3) in id_as_key_dict
        assert key1 not in id_as_key_dict
        assert key2 not in id_as_key_dict
        assert key3 not in id_as_key_dict
    """

    def __init__(self, *args, **kwargs):
        internal_dict = {}
        for arg in args:
            for k, v in arg.items():
                internal_dict[id(k)] = v
        for k, v in kwargs.items():
            internal_dict[id(k)] = v
        super().__init__(internal_dict)

    def __getitem__(self, _key):
        internal_key = id(_key)
        return super().__getitem__(internal_key)

    def __setitem__(self, _key, _value):
        internal_key = id(_key)
        return super().__setitem__(internal_key, _value)

    def get(self, key, default=None):
        internal_key = id(key)
        return super().get(internal_key, default)
