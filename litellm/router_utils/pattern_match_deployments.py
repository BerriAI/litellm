"""
Class to handle llm wildcard routing and regex pattern matching
"""

import copy
import re
from re import Match
from typing import Dict, List, Optional, Tuple

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm._logging import verbose_router_logger


class PatternUtils:
    @staticmethod
    def calculate_pattern_specificity(pattern: str) -> Tuple[int, int]:
        """
        Calculate pattern specificity based on length and complexity.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            Tuple of (length, complexity) for sorting
        """
        complexity_chars = ["*", "+", "?", "\\", "^", "$", "|", "(", ")"]
        ret_val = (
            len(pattern),  # Longer patterns more specific
            sum(
                pattern.count(char) for char in complexity_chars
            ),  # More regex complexity
        )
        return ret_val

    @staticmethod
    def sorted_patterns(
        patterns: Dict[str, List[Dict]]
    ) -> List[Tuple[str, List[Dict]]]:
        """
        Cached property for patterns sorted by specificity.

        Returns:
            Sorted list of pattern-deployment tuples
        """
        return sorted(
            patterns.items(),
            key=lambda x: PatternUtils.calculate_pattern_specificity(x[0]),
            reverse=True,
        )


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
        # # Replace '*' with '.*' for regex matching
        # regex = pattern.replace("*", ".*")
        # # Escape other special characters
        # regex = re.escape(regex).replace(r"\.\*", ".*")
        # return f"^{regex}$"
        return re.escape(pattern).replace(r"\*", "(.*)")

    def _return_pattern_matched_deployments(
        self, matched_pattern: Match, deployments: List[Dict]
    ) -> List[Dict]:
        new_deployments = []
        for deployment in deployments:
            new_deployment = copy.deepcopy(deployment)
            new_deployment["litellm_params"][
                "model"
            ] = PatternMatchRouter.set_deployment_model_name(
                matched_pattern=matched_pattern,
                litellm_deployment_litellm_model=deployment["litellm_params"]["model"],
            )
            new_deployments.append(new_deployment)

        return new_deployments

    def route(
        self, request: Optional[str], filtered_model_names: Optional[List[str]] = None
    ) -> Optional[List[Dict]]:
        """
        Route a requested model to the corresponding llm deployments based on the regex pattern

        loop through all the patterns and find the matching pattern
        if a pattern is found, return the corresponding llm deployments
        if no pattern is found, return None

        Args:
            request: str - the received model name from the user (can be a wildcard route). If none, No deployments will be returned.
            filtered_model_names: Optional[List[str]] - if provided, only return deployments that match the filtered_model_names
        Returns:
            Optional[List[Deployment]]: llm deployments
        """
        try:
            if request is None:
                return None

            sorted_patterns = PatternUtils.sorted_patterns(self.patterns)
            regex_filtered_model_names = (
                [self._pattern_to_regex(m) for m in filtered_model_names]
                if filtered_model_names is not None
                else []
            )
            for pattern, llm_deployments in sorted_patterns:
                if (
                    filtered_model_names is not None
                    and pattern not in regex_filtered_model_names
                ):
                    continue
                pattern_match = re.match(pattern, request)
                if pattern_match:
                    return self._return_pattern_matched_deployments(
                        matched_pattern=pattern_match, deployments=llm_deployments
                    )
        except Exception as e:
            verbose_router_logger.debug(f"Error in PatternMatchRouter.route: {str(e)}")

        return None  # No matching pattern found

    @staticmethod
    def set_deployment_model_name(
        matched_pattern: Match,
        litellm_deployment_litellm_model: str,
    ) -> str:
        """
        Set the model name for the matched pattern llm deployment

        E.g.:

        Case 1:
        model_name: llmengine/* (can be any regex pattern or wildcard pattern)
        litellm_params:
            model: openai/*

        if model_name = "llmengine/foo" -> model = "openai/foo"

        Case 2:
        model_name: llmengine/fo::*::static::*
        litellm_params:
            model: openai/fo::*::static::*

        if model_name = "llmengine/foo::bar::static::baz" -> model = "openai/foo::bar::static::baz"

        Case 3:
        model_name: *meta.llama3*
        litellm_params:
            model: bedrock/meta.llama3*

        if model_name = "hello-world-meta.llama3-70b" -> model = "bedrock/meta.llama3-70b"
        """

        ## BASE CASE: if the deployment model name does not contain a wildcard, return the deployment model name
        if "*" not in litellm_deployment_litellm_model:
            return litellm_deployment_litellm_model

        wildcard_count = litellm_deployment_litellm_model.count("*")

        # Extract all dynamic segments from the request
        dynamic_segments = matched_pattern.groups()

        if len(dynamic_segments) > wildcard_count:
            return (
                matched_pattern.string
            )  # default to the user input, if unable to map based on wildcards.
        # Replace the corresponding wildcards in the litellm model pattern with extracted segments
        for segment in dynamic_segments:
            litellm_deployment_litellm_model = litellm_deployment_litellm_model.replace(
                "*", segment, 1
            )

        return litellm_deployment_litellm_model

    def get_pattern(
        self, model: str, custom_llm_provider: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Check if a pattern exists for the given model and custom llm provider

        Args:
            model: str
            custom_llm_provider: Optional[str]

        Returns:
            bool: True if pattern exists, False otherwise
        """
        if custom_llm_provider is None:
            try:
                (
                    _,
                    custom_llm_provider,
                    _,
                    _,
                ) = get_llm_provider(model=model)
            except Exception:
                # get_llm_provider raises exception when provider is unknown
                pass
        return self.route(model) or self.route(f"{custom_llm_provider}/{model}")

    def get_deployments_by_pattern(
        self, model: str, custom_llm_provider: Optional[str] = None
    ) -> List[Dict]:
        """
        Get the deployments by pattern

        Args:
            model: str
            custom_llm_provider: Optional[str]

        Returns:
            List[Dict]: llm deployments matching the pattern
        """
        pattern_match = self.get_pattern(model, custom_llm_provider)
        if pattern_match:
            return pattern_match
        return []


# Example usage:
# router = PatternRouter()
# router.add_pattern('openai/*', [Deployment(), Deployment()])
# router.add_pattern('openai/fo::*::static::*', Deployment())
# print(router.route('openai/gpt-4'))  # Output: [Deployment(), Deployment()]
# print(router.route('openai/fo::hi::static::hi'))  # Output: [Deployment()]
# print(router.route('something/else'))  # Output: None
