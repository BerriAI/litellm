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

import dataclasses
import pprint
import re
import reprlib
import textwrap


def set_doc(doc):
    """A decorator to set the docstring of a function."""

    def inner(f):
        f.__doc__ = doc
        return f

    return inner


def strip_oneof(docstring):
    lines = docstring.splitlines()
    lines = [line for line in lines if ".. _oneof:" not in line]
    lines = [line for line in lines if "This field is a member of `oneof`_" not in line]
    return "\n".join(lines)


def prettyprint(cls):
    cls.__str__ = _prettyprint
    cls.__repr__ = _prettyprint
    return cls


repr = reprlib.Repr()


@reprlib.recursive_repr()
def _prettyprint(self):
    """A dataclass prettyprint function you can use in __str__or __repr__.

    Note: You can't set `__str__ = pprint.pformat` because it causes a recursion error.

    Mostly identical to pprint but:

    * This will contract long lists and dicts (> 10lines) to [...] and {...}.
    * This will contract long object reprs to ClassName(...).
    """
    fields = []
    for f in dataclasses.fields(self):
        s = pprint.pformat(getattr(self, f.name))
        class_re = r"^(\w+)\(.*\)$"
        if s.count("\n") >= 10:
            if s.startswith("["):
                s = "[...]"
            elif s.startswith("{"):
                s = "{...}"
            elif re.match(class_re, s, flags=re.DOTALL):
                s = re.sub(class_re, r"\1(...)", s, flags=re.DOTALL)
            else:
                s = "..."
        else:
            width = len(f.name) + 1
            s = textwrap.indent(s, " " * width).lstrip(" ")
        fields.append(f"{f.name}={s}")
    attrs = ",\n".join(fields)

    name = self.__class__.__name__
    width = len(name) + 1

    attrs = textwrap.indent(attrs, " " * width).lstrip(" ")
    return f"{name}({attrs})"
