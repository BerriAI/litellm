# Copyright 2020 Google LLC
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

"""This module contains classes that traverse AST and convert it to something else.

If the parser successfully accepts a valid input (the bigquery cell magic arguments),
the result is an Abstract Syntax Tree (AST) that represents the input as a tree
with notes containing various useful metadata.

Node visitors can process such tree and convert it to something else that can
be used for further processing, for example:

 * An optimized version of the tree with redundancy removed/simplified (not used here).
 * The same tree, but with semantic errors checked, because an otherwise syntactically
   valid input might still contain errors (not used here, semantic errors are detected
   elsewhere).
 * A form that can be directly handed to the code that operates on the input. The
   ``QueryParamsExtractor`` class, for instance, splits the input arguments into
   the "--params <...>" part and everything else.
   The "everything else" part can be then parsed by the default Jupyter argument parser,
   while the --params option is processed separately by the Python evaluator.

More info on the visitor design pattern:
https://en.wikipedia.org/wiki/Visitor_pattern

"""

from __future__ import print_function


class NodeVisitor(object):
    """Base visitor class implementing the dispatch machinery."""

    def visit(self, node):
        method_name = "visit_{}".format(type(node).__name__)
        visitor_method = getattr(self, method_name, self.method_missing)
        return visitor_method(node)

    def method_missing(self, node):
        raise Exception("No visit_{} method".format(type(node).__name__))


class QueryParamsExtractor(NodeVisitor):
    """A visitor that extracts the "--params <...>" part from input line arguments."""

    def visit_InputLine(self, node):
        params_dict_parts = []
        other_parts = []

        dest_var_parts = self.visit(node.destination_var)
        params, other_options = self.visit(node.option_list)

        if dest_var_parts:
            other_parts.extend(dest_var_parts)

        if dest_var_parts and other_options:
            other_parts.append(" ")
        other_parts.extend(other_options)

        params_dict_parts.extend(params)

        return "".join(params_dict_parts), "".join(other_parts)

    def visit_DestinationVar(self, node):
        return [node.name] if node.name is not None else []

    def visit_CmdOptionList(self, node):
        params_opt_parts = []
        other_parts = []

        for i, opt in enumerate(node.options):
            option_parts = self.visit(opt)
            list_to_extend = params_opt_parts if opt.name == "params" else other_parts

            if list_to_extend:
                list_to_extend.append(" ")
            list_to_extend.extend(option_parts)

        return params_opt_parts, other_parts

    def visit_CmdOption(self, node):
        result = ["--{}".format(node.name)]

        if node.value is not None:
            result.append(" ")
            value_parts = self.visit(node.value)
            result.extend(value_parts)

        return result

    def visit_CmdOptionValue(self, node):
        return [node.value]

    def visit_ParamsOption(self, node):
        value_parts = self.visit(node.value)
        return value_parts

    def visit_PyVarExpansion(self, node):
        return [node.raw_value]

    def visit_PyDict(self, node):
        result = ["{"]

        for i, item in enumerate(node.items):
            if i > 0:
                result.append(", ")
            item_parts = self.visit(item)
            result.extend(item_parts)

        result.append("}")
        return result

    def visit_PyDictItem(self, node):
        result = self.visit(node.key)  # key parts
        result.append(": ")
        value_parts = self.visit(node.value)
        result.extend(value_parts)
        return result

    def visit_PyDictKey(self, node):
        return [node.key_value]

    def visit_PyScalarValue(self, node):
        return [node.raw_value]

    def visit_PyTuple(self, node):
        result = ["("]

        for i, item in enumerate(node.items):
            if i > 0:
                result.append(", ")
            item_parts = self.visit(item)
            result.extend(item_parts)

        result.append(")")
        return result

    def visit_PyList(self, node):
        result = ["["]

        for i, item in enumerate(node.items):
            if i > 0:
                result.append(", ")
            item_parts = self.visit(item)
            result.extend(item_parts)

        result.append("]")
        return result
