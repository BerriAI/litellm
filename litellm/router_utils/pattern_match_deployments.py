"""
Class to handle llm wildcard routing and regex pattern matching
"""

import re
from typing import Dict, List, Optional

from litellm._logging import verbose_router_logger


class PatternMatchRouter:
    """
    Class to handle llm wildcard routing and regex pattern matching

    doc: https://docs.litellm.ai/docs/proxy/configs#provider-specific-wildcard-routing

    This class will store a mapping for regex pattern: List[Deployments]
    """

    def __init__(self):
        self.patterns: Dict[str, List] = {}

    def add_pattern(self, pattern: str, llm_deployment: Dict):
        """
        Add a regex pattern and the corresponding llm deployments to the patterns

        Args:
            pattern: str
            llm_deployment: str or List[str]
        """
        # Convert the pattern to a regex
        regex = self._pattern_to_regex(pattern)
        if regex not in self.patterns:
            self.patterns[regex] = []
        if isinstance(llm_deployment, list):
            self.patterns[regex].extend(llm_deployment)
        else:
            self.patterns[regex].append(llm_deployment)

    def _pattern_to_regex(self, pattern: str) -> str:
        """
        Convert a wildcard pattern to a regex pattern

        example:
        pattern: openai/*
        regex: openai/.*

        pattern: openai/fo::*::static::*
        regex: openai/fo::.*::static::.*

        Args:
            pattern: str

        Returns:
            str: regex pattern
        """
        # Replace '*' with '.*' for regex matching
        regex = pattern.replace("*", ".*")
        # Escape other special characters
        regex = re.escape(regex).replace(r"\.\*", ".*")
        return f"^{regex}$"

    def route(self, request: Optional[str]) -> Optional[List[Dict]]:
        """
        Route a requested model to the corresponding llm deployments based on the regex pattern

        loop through all the patterns and find the matching pattern
        if a pattern is found, return the corresponding llm deployments
        if no pattern is found, return None

        Args:
            request: Optional[str]

        Returns:
            Optional[List[Deployment]]: llm deployments
        """
        try:
            if request is None:
                return None
            for pattern, llm_deployments in self.patterns.items():
                if re.match(pattern, request):
                    return llm_deployments
        except Exception as e:
            verbose_router_logger.debug(f"Error in PatternMatchRouter.route: {str(e)}")

        return None  # No matching pattern found


# Example usage:
# router = PatternRouter()
# router.add_pattern('openai/*', [Deployment(), Deployment()])
# router.add_pattern('openai/fo::*::static::*', Deployment())
# print(router.route('openai/gpt-4'))  # Output: [Deployment(), Deployment()]
# print(router.route('openai/fo::hi::static::hi'))  # Output: [Deployment()]
# print(router.route('something/else'))  # Output: None
