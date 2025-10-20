"""
AWS Bedrock AgentCore Runtime Provider

This module provides support for AWS Bedrock AgentCore Runtime API.
"""

from .handler import AgentCoreConfig, completion, acompletion

__all__ = ["AgentCoreConfig", "completion", "acompletion"]
