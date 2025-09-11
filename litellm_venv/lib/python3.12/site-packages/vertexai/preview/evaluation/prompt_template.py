# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
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
import string
from typing import Set


class PromptTemplate:
    """A prompt template for creating prompts with placeholders.

    The `PromptTemplate` class allows users to define a template string with
    placeholders represented in curly braces `{placeholder}`. The placeholder
    names cannot contain spaces. These placeholders can be replaced with specific
    values using the `assemble` method, providing flexibility in generating
    dynamic prompts.

    Example Usage:

        ```
            template_str = "Hello, {name}! Today is {day}. How are you?"
            prompt_template = PromptTemplate(template_str)
            completed_prompt = prompt_template.assemble(name="John", day="Monday")
            print(completed_prompt)
        ```

    Attributes:
        template: The template string containing placeholders for replacement.
        placeholders: A set of placeholder names from the template string.
    """

    def __init__(self, template: str):
        """Initializes the PromptTemplate with a given template.

        Args:
            template: The template string with placeholders. Placeholders should be
              represented in curly braces `{placeholder}`.
        """
        self.template = str(template)
        self.placeholders = self._get_placeholders()

    def _get_placeholders(self) -> Set[str]:
        """Extracts and return a set of placeholder names from the template."""
        return set(
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.template)
            if field_name is not None
        )

    def assemble(self, **kwargs) -> "PromptTemplate":
        """Replaces only the provided placeholders in the template with specific values.

        Args:
            **kwargs: Keyword arguments where keys are placeholder names and values
              are the replacements.

        Returns:
            A new PromptTemplate instance with the updated template string.
        """
        replaced_values = {
            key: kwargs.get(key, "{" + key + "}") for key in self.placeholders
        }
        new_template = self.template.format(**replaced_values)
        return PromptTemplate(new_template)

    def __str__(self) -> str:
        """Returns the template string."""
        return self.template

    def __repr__(self) -> str:
        """Returns a string representation of the PromptTemplate."""
        return f"PromptTemplate('{self.template}')"
