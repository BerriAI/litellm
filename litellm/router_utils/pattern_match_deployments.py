"""
Class to handle llm wildcard routing and regex pattern matching
"""

import re


class PatternMatchRouter:
    def __init__(self):
        self.patterns = {}

    def add_pattern(self, pattern, handler):
        # Convert the pattern to a regex
        regex = self._pattern_to_regex(pattern)
        if regex not in self.patterns:
            self.patterns[regex] = []
        if isinstance(handler, list):
            self.patterns[regex].extend(handler)
        else:
            self.patterns[regex].append(handler)

    def _pattern_to_regex(self, pattern):
        # Replace '*' with '.*' for regex matching
        regex = pattern.replace("*", ".*")
        # Escape other special characters
        regex = re.escape(regex).replace(r"\.\*", ".*")
        return f"^{regex}$"

    def route(self, request):
        for pattern, handlers in self.patterns.items():
            if re.match(pattern, request):
                return handlers
        return None  # No matching pattern found


# Example usage:
# router = PatternRouter()
# router.add_pattern('openai/*', ['openai_handler1', 'openai_handler2'])
# router.add_pattern('openai/fo::*::static::*', 'openai_fo_static_handler')
# print(router.route('openai/gpt-4'))  # Output: ['openai_handler1', 'openai_handler2']
# print(router.route('openai/fo::hi::static::hi'))  # Output: ['openai_fo_static_handler']
# print(router.route('something/else'))  # Output: None
