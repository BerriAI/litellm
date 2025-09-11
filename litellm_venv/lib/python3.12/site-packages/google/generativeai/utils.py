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
from __future__ import annotations


def flatten_update_paths(updates):
    new_updates = {}
    for key, value in updates.items():
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_update_paths(value).items():
                new_updates[f"{key}.{sub_key}"] = sub_value
        else:
            new_updates[key] = value

    return new_updates
